"""Response planning implemented with Gemini.

One call per question: the model gets the full screenshot, the question, its
response category and the relevant segments from ``relevant_ss_manifest.json``
with their pixel boxes, followed by the saved close-up image of each segment. The system instruction is the shared planning brief plus
the section for that category. It returns a plan - goal, approach, ordered steps
and caveats - and never the answer's sentences.

The plan is saved as ``response_plan.json`` in the screenshot's own folder. Each
step lists segment ids; their boxes are copied from the manifest in code, so the
coordinates in the plan are never invented by the model.
"""

import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, ValidationError

from app import config
from app.planner.manifest import write_plan
from app.planner.prompts import system_instruction
from app.planner.provider import PlanResult, ResponsePlanner, ResponsePlanningError, VisualAction
from app.refinement.manifest import RELEVANT_MANIFEST_NAME
from app.refinement.provider import SelectionResult

logger = logging.getLogger(__name__)

# Long OCR blocks are cut in the prompt only; the manifests keep the full text.
PROMPT_TEXT_CHARS = 300


class _Step(BaseModel):
    purpose: str
    target: str
    segment_ids: list[int]
    visual_action: VisualAction
    key_points: list[str]


class _Plan(BaseModel):
    goal: str
    approach: str
    steps: list[_Step]
    caveats: list[str]


class GeminiResponsePlanner(ResponsePlanner):
    """Plans the response to a question with Gemini."""

    name = "gemini"

    def __init__(self, api_key=None, model=None, timeout_seconds=None, thinking_budget=None, client=None):
        self._api_key = config.GEMINI_API_KEY if api_key is None else api_key
        self._model = model or config.RESPONSE_PLANNING_MODEL or config.GEMINI_MODEL
        self._timeout_seconds = (
            config.GEMINI_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self._thinking_budget = (
            config.RESPONSE_PLANNING_THINKING_BUDGET if thinking_budget is None else thinking_budget
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

    def plan(self, selection: SelectionResult) -> PlanResult:
        if self._client is None:
            raise ResponsePlanningError("GEMINI_API_KEY is not set; add it to .env or the environment")

        category = selection.response_category
        try:
            instruction = system_instruction(category)
        except ValueError as exc:
            raise ResponsePlanningError(str(exc)) from exc

        started = time.perf_counter()

        manifest_path = Path(selection.manifest_path)
        manifest = self._read_manifest(manifest_path)
        folder = manifest_path.parent
        screenshot_path = folder / manifest["screenshot"]
        screenshot, size = self._read_screenshot(screenshot_path)

        segments = manifest["segments"]
        plan = self._ask(instruction, screenshot, size, manifest, category, folder)
        document = self._document(plan, manifest, segments, size, category, time.perf_counter() - started)

        try:
            path = write_plan(folder, document)
        except OSError as exc:
            raise ResponsePlanningError(f"Could not write the response plan: {exc}") from exc

        seconds = time.perf_counter() - started
        logger.info("[Plan] goal: %s", document["goal"])
        for step in document["steps"]:
            logger.info(
                "[Plan] step %s: %s (%s, segments %s)",
                step["order"],
                step["purpose"],
                step["visual_action"],
                step["segment_ids"],
            )
        logger.info("[Plan] %s step(s) in %.2fs -> %s", len(document["steps"]), seconds, path)
        return PlanResult(
            plan_path=str(path),
            response_category=category,
            step_count=len(document["steps"]),
            seconds=seconds,
        )

    # --- inputs -------------------------------------------------------------

    @staticmethod
    def _read_manifest(path):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["screenshot"], manifest["segments"], manifest["query"]
        except (OSError, ValueError, KeyError) as exc:
            raise ResponsePlanningError(f"Could not read {RELEVANT_MANIFEST_NAME}: {exc}") from exc
        return manifest

    @staticmethod
    def _read_screenshot(path):
        try:
            data = path.read_bytes()
            with Image.open(path) as image:
                size = image.size
        except (OSError, ValueError) as exc:
            raise ResponsePlanningError(f"Could not read the screenshot: {exc}") from exc
        return data, size

    @staticmethod
    def _read_crops(segments, folder):
        """The saved crop of each segment; a segment without a readable crop is sent as text only."""
        crops = []
        for segment in segments:
            if not segment.get("file"):
                continue
            try:
                crops.append((segment["id"], (folder / segment["file"]).read_bytes()))
            except OSError as exc:
                logger.warning("[Plan] could not read the crop of segment %s: %s", segment["id"], exc)
        return crops

    # --- the model call -----------------------------------------------------

    def _ask(self, instruction, screenshot, size, manifest, category, folder):
        crops = self._read_crops(manifest["segments"], folder)
        prompt = (
            f"User question: {manifest['query']}\n"
            f"Response category: {category.value} ({manifest.get('category_reason', '')})\n"
            f"Screenshot size: {size[0]}x{size[1]} pixels\n\n"
            f"Relevant segments (JSON):\n{self._prompt_segments(manifest['segments'])}\n\n"
            f"A close-up image of each segment follows, labelled with its id "
            f"({len(crops)} of {len(manifest['segments'])} available)."
        )
        parts = [
            types.Part.from_bytes(data=screenshot, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
        for segment_id, data in crops:
            parts.append(types.Part.from_text(text=f"Segment {segment_id}:"))
            parts.append(types.Part.from_bytes(data=data, mime_type="image/png"))
        thinking = (
            types.ThinkingConfig(thinking_budget=self._thinking_budget)
            if self._thinking_budget is not None and self._thinking_budget >= 0
            else None
        )

        logger.info(
            "[Plan] asking %s for a %s plan from %s segment(s)",
            self._model,
            category.value,
            len(manifest["segments"]),
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                    response_mime_type="application/json",
                    response_schema=_Plan,
                    thinking_config=thinking,
                ),
            )
        except Exception as exc:
            raise ResponsePlanningError(f"Gemini request failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            logger.info(
                "[Plan] tokens: prompt=%s output=%s thinking=%s",
                usage.prompt_token_count,
                usage.candidates_token_count,
                getattr(usage, "thoughts_token_count", None),
            )
        return self._decode(response)

    @staticmethod
    def _prompt_segments(segments):
        entries = [
            {
                "id": segment["id"],
                "kind": segment["kind"],
                "text": " ".join(segment["text"].split())[:PROMPT_TEXT_CHARS],
                "score": segment["score"],
                "box": segment["box"],
            }
            for segment in segments
        ]
        return json.dumps({"segments": entries}, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode(response):
        try:
            parsed = getattr(response, "parsed", None)
            plan = _Plan.model_validate(parsed) if parsed is not None else _Plan.model_validate_json(response.text)
        except (ValidationError, ValueError, TypeError) as exc:
            raise ResponsePlanningError(f"Gemini returned an invalid plan: {exc}") from exc

        if not plan.steps:
            raise ResponsePlanningError("Gemini returned a plan without steps")
        return plan

    # --- the saved document -------------------------------------------------

    def _document(self, plan, manifest, segments, size, category, seconds):
        boxes = {segment["id"]: segment for segment in segments}

        steps = []
        for order, step in enumerate(plan.steps, start=1):
            ids = []
            for segment_id in step.segment_ids:
                if segment_id not in boxes:
                    logger.warning("[Plan] step %s names unknown segment id %s; ignored", order, segment_id)
                elif segment_id not in ids:
                    ids.append(segment_id)
            steps.append(
                {
                    "order": order,
                    "purpose": " ".join(step.purpose.split()),
                    "target": " ".join(step.target.split()),
                    "visual_action": step.visual_action.value,
                    "key_points": [" ".join(point.split()) for point in step.key_points if point.strip()],
                    "segment_ids": ids,
                    "targets": [
                        {"id": segment_id, "kind": boxes[segment_id]["kind"], "box": boxes[segment_id]["box"]}
                        for segment_id in ids
                    ],
                }
            )

        return {
            "query": manifest["query"],
            "response_category": category.value,
            "category_confidence": manifest.get("confidence"),
            "category_reason": manifest.get("category_reason", ""),
            "screenshot": manifest["screenshot"],
            "screenshot_size": {"width": size[0], "height": size[1]},
            "source_manifest": RELEVANT_MANIFEST_NAME,
            "model": self._model,
            "planning_seconds": round(seconds, 3),
            "goal": " ".join(plan.goal.split()),
            "approach": " ".join(plan.approach.split()),
            "steps": steps,
            "caveats": [" ".join(caveat.split()) for caveat in plan.caveats if caveat.strip()],
        }
