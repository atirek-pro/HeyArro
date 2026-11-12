"""Final response composing implemented with Gemini.

One call per question: the model gets the screenshot, the question, its category
and the response plan (goal, approach, steps with their purpose, target, overlay
action and key points, and caveats). It returns the sentences to speak, each
naming the plan step it belongs to. It never sees or writes coordinates: the
pointer of every sentence is copied in code from ``plan_pointers.json`` for its
step.

The result is saved as ``final_response.json`` next to the plan.
"""

import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app import config
from app.final_response.manifest import write_final_response
from app.final_response.prompts import system_instruction
from app.final_response.provider import FinalResponseError, FinalResponseResult, ResponseComposer
from app.grounding.manifest import POINTERS_NAME
from app.grounding.provider import GroundingResult
from app.planner.manifest import PLAN_NAME
from app.refinement.provider import ResponseCategory

logger = logging.getLogger(__name__)

# Opening and closing sentences belong to no plan step.
NO_STEP = 0


class _Sentence(BaseModel):
    step: int
    text: str


class _Response(BaseModel):
    sentences: list[_Sentence]


class GeminiResponseComposer(ResponseComposer):
    """Writes the final response from a grounded plan with Gemini."""

    name = "gemini"

    def __init__(self, api_key=None, model=None, timeout_seconds=None, thinking_budget=None, client=None):
        self._api_key = config.GEMINI_API_KEY if api_key is None else api_key
        self._model = model or config.FINAL_RESPONSE_MODEL or config.GEMINI_MODEL
        self._timeout_seconds = (
            config.GEMINI_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self._thinking_budget = (
            config.FINAL_RESPONSE_THINKING_BUDGET if thinking_budget is None else thinking_budget
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

    def compose(self, grounding: GroundingResult) -> FinalResponseResult:
        if self._client is None:
            raise FinalResponseError("GEMINI_API_KEY is not set; add it to .env or the environment")

        started = time.perf_counter()

        folder = Path(grounding.pointers_path).parent
        plan = self._read_json(folder / PLAN_NAME)
        pointers = self._read_json(folder / POINTERS_NAME)
        try:
            category = ResponseCategory(plan["response_category"])
            instruction = system_instruction(category)
            steps = plan["steps"]
            pointers_by_step = {pointer["order"]: pointer for pointer in pointers["pointers"]}
            screenshot = (folder / plan["screenshot"]).read_bytes()
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalResponseError(f"Could not use {PLAN_NAME} and {POINTERS_NAME}: {exc}") from exc
        except OSError as exc:
            raise FinalResponseError(f"Could not read the screenshot: {exc}") from exc

        answer = self._ask(instruction, screenshot, plan, category)
        sentences, uncovered = self._attach_pointers(answer, steps, pointers_by_step)

        seconds = time.perf_counter() - started
        document = {
            "query": plan.get("query", ""),
            "response_category": category.value,
            "model": self._model,
            "screenshot": plan["screenshot"],
            "screenshot_size": plan.get("screenshot_size"),
            "source_plan": PLAN_NAME,
            "source_pointers": POINTERS_NAME,
            "composing_seconds": round(seconds, 3),
            "sentences": sentences,
            "uncovered_steps": uncovered,
        }
        try:
            path = write_final_response(folder, document)
        except OSError as exc:
            raise FinalResponseError(f"Could not write the final response: {exc}") from exc

        for sentence in sentences:
            logger.info(
                "[Response] %s. %s (step %s, %s)",
                sentence["index"],
                sentence["text"],
                sentence["step"],
                "pointer " + str(sentence["pointer"]["point"]) if sentence["pointer"] else "no pointer",
            )
        if uncovered:
            logger.warning("[Response] plan step(s) without a sentence: %s", uncovered)
        logger.info("[Response] %s sentence(s) in %.2fs -> %s", len(sentences), seconds, path)
        return FinalResponseResult(response_path=str(path), sentence_count=len(sentences), seconds=seconds)

    # --- inputs -------------------------------------------------------------

    @staticmethod
    def _read_json(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FinalResponseError(f"Could not read {path.name}: {exc}") from exc

    # --- the model call -----------------------------------------------------

    def _ask(self, instruction, screenshot, plan, category):
        prompt_plan = {
            "goal": plan.get("goal", ""),
            "approach": plan.get("approach", ""),
            "steps": [
                {
                    "order": step["order"],
                    "purpose": step.get("purpose", ""),
                    "target": step.get("target", ""),
                    "visual_action": step.get("visual_action", "none"),
                    "key_points": step.get("key_points", []),
                }
                for step in plan["steps"]
            ],
            "caveats": plan.get("caveats", []),
        }
        prompt = (
            f"User question: {plan.get('query', '')}\n"
            f"Response category: {category.value}\n\n"
            f"Response plan (JSON):\n{json.dumps(prompt_plan, ensure_ascii=False, separators=(',', ':'))}"
        )
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
            "[Response] asking %s for the %s response to a %s-step plan",
            self._model,
            category.value,
            len(plan["steps"]),
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                    response_mime_type="application/json",
                    response_schema=_Response,
                    thinking_config=thinking,
                ),
            )
        except Exception as exc:
            raise FinalResponseError(f"Gemini request failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            logger.info(
                "[Response] tokens: prompt=%s output=%s thinking=%s",
                usage.prompt_token_count,
                usage.candidates_token_count,
                getattr(usage, "thoughts_token_count", None),
            )

        try:
            parsed = getattr(response, "parsed", None)
            return _Response.model_validate(parsed) if parsed is not None else _Response.model_validate_json(response.text)
        except (ValidationError, ValueError, TypeError) as exc:
            raise FinalResponseError(f"Gemini returned an invalid response: {exc}") from exc

    # --- the saved document -------------------------------------------------

    @staticmethod
    def _attach_pointers(answer, steps, pointers_by_step):
        """Give each sentence its plan step's pointer; return the sentences and the uncovered steps."""
        steps_by_order = {step["order"]: step for step in steps}

        sentences = []
        for item in answer.sentences:
            text = " ".join(item.text.split())
            if not text:
                continue
            if item.step != NO_STEP and item.step not in steps_by_order:
                logger.warning("[Response] dropping a sentence for unknown plan step %s: %r", item.step, text)
                continue

            step = steps_by_order.get(item.step)
            pointer = pointers_by_step.get(item.step)
            has_pointer = (
                step is not None
                and step.get("visual_action", "none") != "none"
                and pointer is not None
                and pointer.get("found")
            )
            sentences.append(
                {
                    "index": len(sentences) + 1,
                    "text": text,
                    "step": item.step,
                    "visual_action": step.get("visual_action", "none") if step else "none",
                    "target": step.get("target", "") if step else "",
                    "pointer": {"box": pointer["box"], "point": pointer["point"]} if has_pointer else None,
                }
            )

        if not sentences:
            raise FinalResponseError("Gemini returned a response without sentences")

        spoken = {sentence["step"] for sentence in sentences}
        uncovered = [step["order"] for step in steps if step["order"] not in spoken]
        return sentences, uncovered
