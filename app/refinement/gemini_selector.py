"""Question categorisation and segment selection implemented with Gemini.

One text-and-screenshot call per question: the model gets the full screenshot,
the segment manifest as JSON and the transcript. It returns the question's
response category (with a confidence and a one-sentence reason) and only the
segments that matter, each with a score. No crop is sent here, and no
per-segment prose is requested, so the answer stays short.

The result is saved next to the screenshot as ``relevant_ss_manifest.json``.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app import config
from app.refinement.manifest import segment_entry, write_relevant_manifest
from app.refinement.provider import (
    RefinementResult,
    ResponseCategory,
    SegmentSelectionError,
    SegmentSelector,
    SelectionResult,
)

logger = logging.getLogger(__name__)

# Long OCR blocks are cut in the prompt only; the manifest files keep the full text.
PROMPT_TEXT_CHARS = 200
MAX_SCORE = 10

SYSTEM_INSTRUCTION = (
    "You help a screen assistant understand a user's spoken question about their screen. "
    "You get the full screenshot, the question, and a JSON manifest of numbered segments "
    "found on that screenshot: each has an id, a kind, a pixel box (left, top, width, "
    "height) and noisy OCR text that may be garbled or empty. Do two things.\n\n"
    "1. Categorise the question as exactly one response_category:\n"
    "- teaching: the user wants to understand an idea or concept; the answer is an "
    "explanation with nothing for the user to do. Examples: 'Explain how linear "
    "regression works from this video', 'Why does the line on this graph slope down?', "
    "'What is a layer mask?', 'Teach me what this function does'.\n"
    "- instructing: the user wants to complete a task; the answer is the full sequence "
    "of actions to perform. Examples: 'How do I export this as a PNG?', 'Walk me "
    "through creating a new branch', 'How do I add a column to this table?', 'Show me "
    "how to change the font size'.\n"
    "- guiding: the user wants to find something or know the next move on screen; the "
    "answer points at one or a few elements, with no full procedure. Examples: 'Where "
    "is the export button?', 'What should I click next?', 'Which of these tools is the "
    "lasso?', 'Is this the right setting?'.\n"
    "- informative: the user asks for a fact or information read from what is on screen; "
    "the answer is the information itself. Examples: 'How many folders are there on my "
    "screen and what are their names?', 'What does the error in the terminal say?', "
    "'Which branch am I on?', 'Read this text', 'What is open in this tab?'.\n"
    "- fallback: anything else - greetings and small talk, questions unrelated to the "
    "screen, and an unclear or empty question.\n"
    "Tie-breaks: 'why', 'what is' and 'explain' (understanding a concept, not just "
    "reading a value) are teaching; 'how do I X' is instructing; guiding is when the "
    "answer is an element to interact with ('where', 'which tool or button to use', "
    "'what next'); informative is when the answer is a fact shown on screen ('what', "
    "'how many', 'what does it say'); for a mixed question follow its main verb. Give a "
    "confidence from 0 to 1 and a one-sentence reason for the category.\n\n"
    "2. Return ONLY the segments that help answer the question, each with a score from 1 "
    "(marginally useful) to 10 (essential). A segment can matter because of what it "
    "shows even if its OCR text does not repeat the question's words (for example, a "
    "file tree answers a question about folders); use the box to find it on the "
    "screenshot. Leave irrelevant segments out. Return an empty list if nothing is "
    "relevant."
)


class _Scored(BaseModel):
    segment_id: int
    score: int


class _Selection(BaseModel):
    # Field order is the order the model writes them: it decides the category first.
    response_category: ResponseCategory
    confidence: float
    reason: str
    segments: list[_Scored]


@dataclass(frozen=True)
class _Assessment:
    response_category: ResponseCategory
    confidence: float
    reason: str
    scores: dict[int, int]


def _prompt_manifest(refinement: RefinementResult):
    entries = []
    for segment in refinement.segments:
        entry = segment_entry(segment, refinement.folder)
        entry.pop("file")
        entry["text"] = " ".join(entry["text"].split())[:PROMPT_TEXT_CHARS]
        entries.append(entry)
    return json.dumps({"segments": entries}, ensure_ascii=False, separators=(",", ":"))


class GeminiSegmentSelector(SegmentSelector):
    """Scores the segments of a screenshot against the user's transcript."""

    name = "gemini"

    def __init__(self, api_key=None, model=None, timeout_seconds=None, thinking_budget=None, client=None):
        self._api_key = config.GEMINI_API_KEY if api_key is None else api_key
        self._model = model or config.SEGMENT_SELECTION_MODEL or config.GEMINI_MODEL
        self._timeout_seconds = (
            config.GEMINI_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self._thinking_budget = (
            config.SEGMENT_SELECTION_THINKING_BUDGET if thinking_budget is None else thinking_budget
        )
        # An injected client is used as-is (tests hand in a fake SDK).
        self._client = client if client is not None else self._build_client()

    def _build_client(self):
        if not self._api_key:
            return None
        return genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self._timeout_seconds * 1000)),
        )

    @property
    def model(self):
        return self._model

    def select(self, refinement: RefinementResult, transcript: str) -> SelectionResult:
        started = time.perf_counter()

        assessment = self._assess(refinement, transcript)
        scores = assessment.scores

        seconds = time.perf_counter() - started
        try:
            path = write_relevant_manifest(
                refinement,
                transcript,
                self._model,
                scores,
                seconds,
                assessment.response_category.value,
                assessment.confidence,
                assessment.reason,
            )
        except OSError as exc:
            raise SegmentSelectionError(f"Could not write the relevant segment manifest: {exc}") from exc

        logger.info(
            "[Selection] category=%s (confidence %.2f): %s",
            assessment.response_category.value,
            assessment.confidence,
            assessment.reason,
        )
        logger.info(
            "[Selection] %s of %s segment(s) relevant in %.2fs: %s -> %s",
            len(scores),
            refinement.segment_count,
            seconds,
            {segment_id: score for segment_id, score in sorted(scores.items(), key=lambda item: -item[1])},
            path,
        )
        return SelectionResult(
            manifest_path=str(path),
            scores=scores,
            seconds=seconds,
            response_category=assessment.response_category,
            confidence=assessment.confidence,
            category_reason=assessment.reason,
        )

    def _assess(self, refinement, transcript):
        if self._client is None:
            raise SegmentSelectionError("GEMINI_API_KEY is not set; add it to .env or the environment")

        try:
            screenshot = Path(refinement.screenshot_path).read_bytes()
        except OSError as exc:
            raise SegmentSelectionError(f"Could not read the screenshot: {exc}") from exc

        prompt = f"User question: {transcript}\n\nSegment manifest (JSON):\n{_prompt_manifest(refinement)}"
        parts = [
            types.Part.from_bytes(data=screenshot, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
        thinking = (
            types.ThinkingConfig(thinking_budget=self._thinking_budget)
            if self._thinking_budget is not None and self._thinking_budget >= 0
            else None
        )

        logger.info(
            "[Selection] asking %s to categorise the question and score %s segment(s) for %r",
            self._model,
            refinement.segment_count,
            transcript,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=_Selection,
                    thinking_config=thinking,
                ),
            )
        except Exception as exc:
            raise SegmentSelectionError(f"Gemini request failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            logger.info(
                "[Selection] tokens: prompt=%s output=%s thinking=%s",
                usage.prompt_token_count,
                usage.candidates_token_count,
                getattr(usage, "thoughts_token_count", None),
            )

        return self._decode(response, {segment.index for segment in refinement.segments})

    @staticmethod
    def _decode(response, known_ids):
        """Validate the answer; unknown ids are dropped, scores clamped to 1-10 and confidence to 0-1."""
        try:
            parsed = getattr(response, "parsed", None)
            selection = (
                _Selection.model_validate(parsed)
                if parsed is not None
                else _Selection.model_validate_json(response.text)
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise SegmentSelectionError(f"Gemini returned an invalid selection: {exc}") from exc

        scores = {}
        for item in selection.segments:
            if item.segment_id not in known_ids:
                logger.warning("[Selection] ignoring unknown segment id %s", item.segment_id)
                continue
            score = min(item.score, MAX_SCORE)
            if score < 1:
                continue
            scores[item.segment_id] = max(score, scores.get(item.segment_id, 0))

        return _Assessment(
            response_category=selection.response_category,
            confidence=min(max(selection.confidence, 0.0), 1.0),
            reason=" ".join(selection.reason.split()),
            scores=scores,
        )
