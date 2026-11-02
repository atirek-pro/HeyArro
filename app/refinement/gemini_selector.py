"""Segment selection implemented with Gemini.

One text-and-screenshot call per question: the model gets the full screenshot,
the segment manifest as JSON and the transcript, and returns only the segments
that matter, each with a score. No crop is sent here, and no per-segment prose is
requested, so the answer stays short.

The result is saved next to the screenshot as ``relevant_ss_manifest.json``.
"""

import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app import config
from app.refinement.manifest import segment_entry, write_relevant_manifest
from app.refinement.provider import (
    RefinementResult,
    SegmentSelectionError,
    SegmentSelector,
    SelectionResult,
)

logger = logging.getLogger(__name__)

# Long OCR blocks are cut in the prompt only; the manifest files keep the full text.
PROMPT_TEXT_CHARS = 200
MAX_SCORE = 10

SYSTEM_INSTRUCTION = (
    "You help a screen assistant decide which parts of a screenshot matter for a user's "
    "spoken question. You get the full screenshot, the question, and a JSON manifest of "
    "numbered segments found on that screenshot: each has an id, a kind, a pixel box "
    "(left, top, width, height) and noisy OCR text that may be garbled or empty. "
    "Return ONLY the segments that help answer the question, each with a score from 1 "
    "(marginally useful) to 10 (essential). A segment can matter because of what it "
    "shows even if its OCR text does not repeat the question's words (for example, a "
    "file tree answers a question about folders); use the box to find it on the "
    "screenshot. Leave irrelevant segments out. Return an empty list if nothing is "
    "relevant. Do not explain."
)


class _Scored(BaseModel):
    segment_id: int
    score: int


class _Selection(BaseModel):
    segments: list[_Scored]


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

        if refinement.segments:
            scores = self._score(refinement, transcript)
        else:
            logger.info("[Selection] no segments to choose from; skipping the model call")
            scores = {}

        seconds = time.perf_counter() - started
        try:
            path = write_relevant_manifest(refinement, transcript, self._model, scores, seconds)
        except OSError as exc:
            raise SegmentSelectionError(f"Could not write the relevant segment manifest: {exc}") from exc

        logger.info(
            "[Selection] %s of %s segment(s) relevant in %.2fs: %s -> %s",
            len(scores),
            refinement.segment_count,
            seconds,
            {segment_id: score for segment_id, score in sorted(scores.items(), key=lambda item: -item[1])},
            path,
        )
        return SelectionResult(manifest_path=str(path), scores=scores, seconds=seconds)

    def _score(self, refinement, transcript):
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
            "[Selection] asking %s to score %s segment(s) for %r",
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
        """Validate the answer; unknown ids are dropped and scores clamped to 1-10."""
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
        return scores
