"""Gemini vision provider - the first real VisionLLMProvider implementation.

Everything Gemini-specific lives here: the SDK import, the request payload, the
structured-output schema and the response decoding. Nothing outside this module
needs to know about them.
"""

import logging
import time

from pydantic import ValidationError

from app import config
from app.llm.provider import (
    LLMError,
    ProviderConfigurationError,
    SYSTEM_INSTRUCTION,
    VisionLLMProvider,
    VisionLLMRequest,
    build_user_prompt,
    image_size,
    validate_screenshot,
)
from app.llm.response import HeyArroResponse

logger = logging.getLogger(__name__)


def _response_text(response):
    """Decode a provider response's raw text without leaking SDK types outward."""
    try:
        text = response.text
    except Exception:
        text = None

    if text:
        return str(text).strip()

    pieces = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            piece = getattr(part, "text", None)
            if piece:
                pieces.append(str(piece))

    return " ".join(pieces).strip()


def _decode_structured(response, model, label):
    """Validate a structured Gemini response into ``model``, or raise LLMError.

    The SDK parses the structured output into ``response.parsed`` when a response
    schema is supplied; the raw JSON text is only used as a fallback. Either way
    Pydantic does the validating, so anything that does not validate becomes an
    LLMError rather than a plausible-looking result.
    """
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        try:
            return model.model_validate(parsed)
        except ValidationError as exc:
            raise LLMError(f"Gemini returned an invalid {label}: {exc}") from exc

    text = _response_text(response)
    if not text:
        raise LLMError("Gemini returned an empty response")

    try:
        return model.model_validate_json(text)
    except ValidationError as exc:
        raise LLMError(f"Gemini returned malformed {label}: {exc}") from exc


def _decode_response(response):
    """Turn a structured Gemini response into a validated HeyArroResponse."""
    return _decode_structured(response, HeyArroResponse, "structured response")


class GeminiVisionProvider(VisionLLMProvider):
    """Sends the transcript and the desktop screenshot to a Gemini model."""

    name = "gemini"

    def __init__(self, api_key=None, model=None, timeout_seconds=None, client=None):
        self._api_key = config.GEMINI_API_KEY if api_key is None else api_key
        self._model = model or config.GEMINI_MODEL
        self._timeout_seconds = (
            config.GEMINI_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self._client = client
        self._sdk = None

        if self._client is None and not self._api_key:
            logger.warning(
                "GEMINI_API_KEY is not set; the Gemini provider will fail until it is configured"
            )

    @property
    def model(self):
        return self._model

    def process(self, request: VisionLLMRequest) -> HeyArroResponse:
        # Validated before the client is built, so a bad screenshot never
        # reaches the API.
        mime_type = validate_screenshot(request.screenshot)

        client = self._get_client()
        types = self._import_sdk()[1]

        # The model can only place a visual target correctly if it is told the
        # screenshot's own pixel grid; fall back to the PNG header if the caller
        # did not supply it.
        screenshot_size = request.screenshot_size or image_size(request.screenshot)

        parts = []
        if mime_type is not None:
            parts.append(types.Part.from_bytes(data=request.screenshot, mime_type=mime_type))
        parts.append(
            types.Part.from_text(
                text=build_user_prompt(
                    request.transcript,
                    mime_type is not None,
                    screenshot_size,
                    request.guidance,
                )
            )
        )

        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=HeyArroResponse,
                ),
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        result = _decode_response(response)

        duration = round(time.perf_counter() - started, 3)
        logger.info(
            "Gemini responded in %ss using model %s (mode=%s, steps=%s)",
            duration,
            self._model,
            result.teaching.mode.value,
            len(result.teaching.steps),
        )

        return result

    def _import_sdk(self):
        if self._sdk is None:
            try:
                from google import genai
                from google.genai import types
            except Exception as exc:
                raise ProviderConfigurationError(f"google-genai is not available: {exc}") from exc
            self._sdk = (genai, types)
        return self._sdk

    # --- teaching plan generation -------------------------------------------

    def generate_teaching_plan(
        self, user_query, screenshot=None, screenshot_size=None, context=None, guidance=None
    ) -> "TeachingPlan":
        """Ask Gemini how to teach ``user_query``, given the screenshot.

        This is the teaching-plan generator: it decides *what* to teach and in
        what order, not where things are on screen. The visual targets it
        produces are approximate on purpose - the grounding pipeline refines
        them later - and it reuses this provider's client, model and
        structured-output mechanism rather than building its own.

        Returns a validated ``TeachingPlan``, or raises LLMError.
        """
        # Imported here so this module does not pull the teaching package in at
        # import time.
        from app.teaching.plan import TeachingPlan
        from app.teaching.plan_prompt import (
            TEACHING_PLAN_SYSTEM_INSTRUCTION,
            build_teaching_plan_prompt,
        )

        # Validated before the client is built, so a bad screenshot never
        # reaches the API.
        mime_type = validate_screenshot(screenshot)
        size = screenshot_size or image_size(screenshot)

        client = self._get_client()
        types = self._import_sdk()[1]

        parts = []
        if mime_type is not None:
            parts.append(types.Part.from_bytes(data=screenshot, mime_type=mime_type))
        parts.append(
            types.Part.from_text(
                text=build_teaching_plan_prompt(
                    user_query, mime_type is not None, size, context, guidance
                )
            )
        )

        logger.info("[TeachingPlan] Generation started")
        logger.info("[TeachingPlan] Request received (%s characters)", len((user_query or "").strip()))
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=TEACHING_PLAN_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=TeachingPlan,
                ),
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Teaching plan request failed: {exc}") from exc

        plan = _decode_structured(response, TeachingPlan, "teaching plan")
        logger.info("[TeachingPlan] Teaching plan generated: %s step(s)", len(plan.steps))
        logger.info("[TeachingPlan] Generation completed in %.1fs", time.perf_counter() - started)
        return plan

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise ProviderConfigurationError(
                "GEMINI_API_KEY is not set; set it or select another provider with LLM_PROVIDER"
            )

        genai, types = self._import_sdk()
        try:
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=int(self._timeout_seconds * 1000)),
            )
        except Exception as exc:
            raise ProviderConfigurationError(f"Could not create the Gemini client: {exc}") from exc

        logger.info(
            "Gemini client ready (model=%s, timeout=%ss)", self._model, self._timeout_seconds
        )
        return self._client
