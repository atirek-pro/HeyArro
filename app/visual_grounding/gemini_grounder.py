"""The Gemini refinement call: locate one target inside a crop.

This is a *visual grounding* request, not a teaching request. The model is given
a high-resolution crop and a description of one target, and answers with only a
bounding box and a confidence. Everything Gemini-specific about that call lives
here; the pipeline in ``grounder.py`` never sees the SDK.
"""

import logging
import time

from pydantic import ValidationError

from app import config
from app.llm.provider import (
    LLMError,
    image_size,
    validate_screenshot,
)
from app.visual_grounding.base import GroundingError
from app.visual_grounding.models import GroundedTarget

logger = logging.getLogger(__name__)

GROUNDING_SYSTEM_INSTRUCTION = (
    "You are a visual grounding system.\n"
    "You are given a high-resolution crop from a computer screenshot and a description of "
    "a target UI element or visual object. Locate the exact target described.\n"
    "\n"
    "Rules:\n"
    "1. Identify the actual visible target, not merely an approximate location.\n"
    "2. Use the surrounding UI context in the crop to tell the target apart from nearby "
    "elements.\n"
    "3. Return the bounding box that covers the target element.\n"
    "4. Coordinates are relative to the provided crop: (0,0) is the crop's top-left pixel. "
    "x and y are the TOP-LEFT corner of the box; width and height are its size.\n"
    "5. Never return coordinates outside the crop.\n"
    "6. Never invent a target that is not visibly present.\n"
    "7. If you cannot confidently identify the target, report a low confidence (below 0.7).\n"
    "8. Do not explain your reasoning.\n"
    "9. Return only the structured response."
)


def build_grounding_prompt(description, crop_size):
    """Build the user turn for one refinement request."""
    width, height = crop_size
    return (
        f'Target: "{description}"\n'
        "\n"
        f"Locate exactly this target in the attached crop. The crop is {width} by {height} "
        f"pixels: x runs from 0 to {width - 1} and y runs from 0 to {height - 1}. Give the "
        "bounding box in those crop pixels, never outside the crop."
    )


def _response_text(response):
    """Return the raw text of a Gemini response, if it has any."""
    try:
        text = response.text
    except Exception:
        text = None
    return str(text).strip() if text else ""


def _decode_grounded_target(response) -> GroundedTarget:
    """Validate the refinement into ``GroundedTarget``, or raise GroundingError.

    The SDK parses structured output into ``response.parsed``; the raw JSON text
    is only a fallback. Either way Pydantic does the validating, so a malformed
    answer becomes a controlled failure instead of a bad box.
    """
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        try:
            return GroundedTarget.model_validate(parsed)
        except ValidationError as exc:
            raise GroundingError(f"The refinement model returned an invalid box: {exc}") from exc

    text = _response_text(response)
    if not text:
        raise GroundingError("The refinement model returned no structured answer")

    try:
        return GroundedTarget.model_validate_json(text)
    except ValidationError as exc:
        raise GroundingError(f"The refinement model returned malformed output: {exc}") from exc


class GeminiTargetLocator:
    """Asks Gemini to locate one described target inside a crop."""

    name = "gemini"

    def __init__(self, api_key=None, model=None, timeout_seconds=None, client=None):
        self._api_key = config.GEMINI_API_KEY if api_key is None else api_key
        self._model = model or config.GEMINI_MODEL
        self._timeout_seconds = (
            config.GEMINI_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self._client = client
        self._sdk = None

    @property
    def model(self):
        return self._model

    def locate(self, crop_image, description, crop_size=None) -> GroundedTarget:
        """Return the target's bounding box inside ``crop_image`` (crop pixels)."""
        try:
            mime_type = validate_screenshot(crop_image)
        except LLMError as exc:
            raise GroundingError(f"The crop could not be sent for refinement: {exc}") from exc
        if mime_type is None:
            raise GroundingError("There is no crop to refine")

        size = crop_size or image_size(crop_image)
        if not size:
            raise GroundingError("The crop's pixel size is unknown")

        client = self._get_client()
        types = self._import_sdk()[1]

        parts = [
            types.Part.from_bytes(data=crop_image, mime_type=mime_type),
            types.Part.from_text(text=build_grounding_prompt(description, size)),
        ]

        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=GROUNDING_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=GroundedTarget,
                ),
            )
        except GroundingError:
            raise
        except Exception as exc:
            raise GroundingError(f"The refinement request failed: {exc}") from exc

        target = _decode_grounded_target(response)
        logger.debug(
            "[Grounding] %s answered in %.3fs", self._model, time.perf_counter() - started
        )
        return target

    def _import_sdk(self):
        if self._sdk is None:
            try:
                from google import genai
                from google.genai import types
            except Exception as exc:
                raise GroundingError(f"google-genai is not available: {exc}") from exc
            self._sdk = (genai, types)
        return self._sdk

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise GroundingError(
                "GEMINI_API_KEY is not set, so visual targets cannot be refined"
            )

        genai, types = self._import_sdk()
        try:
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=int(self._timeout_seconds * 1000)),
            )
        except Exception as exc:
            raise GroundingError(f"Could not create the Gemini client: {exc}") from exc

        logger.info(
            "Grounding client ready (model=%s, timeout=%ss)", self._model, self._timeout_seconds
        )
        return self._client
