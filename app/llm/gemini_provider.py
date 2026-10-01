"""Gemini vision provider - the real VisionLLMProvider implementation.

Everything Gemini-specific lives here: the SDK import, the request payload, the
structured-output schema and the response decoding. Nothing outside this module
needs to know about them.
"""

import logging
import time

from pydantic import ValidationError

from google import genai
from google.genai import types

from app import config
from app.llm.provider import (
    LLMError,
    ProviderConfigurationError,
    SYSTEM_INSTRUCTION,
    VisionLLMProvider,
    VisionLLMRequest,
    build_user_prompt,
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
        # An injected client is used as-is - tests hand in a fake SDK, and a
        # caller may manage its own connection; otherwise the SDK client is
        # built once, here.
        self._client = client if client is not None else self._build_client()

        if not self._api_key and client is None:
            logger.warning(
                "GEMINI_API_KEY is not set; the Gemini provider will fail until it is configured"
            )

    def _build_client(self):
        """Build the SDK client with the configured timeout, if there is a key.

        A missing key is reported when a request is made, not here, so an
        unconfigured provider can still be constructed and warned about instead
        of taking the application down at startup.
        """
        if not self._api_key:
            return None

        return genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self._timeout_seconds * 1000)),
        )

    def _client_or_raise(self):
        """Return the SDK client, or explain that no key is configured."""
        if self._client is None:
            raise ProviderConfigurationError(
                "GEMINI_API_KEY is not set; add it to .env or the environment"
            )
        return self._client

    @property
    def model(self):
        return self._model

    def process(self, request: VisionLLMRequest) -> HeyArroResponse:
        # Validated before anything is sent, so a bad screenshot never reaches
        # the API.
        mime_type = validate_screenshot(request.screenshot)

        client = self._client_or_raise()

        parts = []
        if mime_type is not None:
            parts.append(types.Part.from_bytes(data=request.screenshot, mime_type=mime_type))
        parts.append(
            types.Part.from_text(
                text=build_user_prompt(
                    request.transcript,
                    mime_type is not None,
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
            "Gemini responded in %ss using model %s (tone=%s, %s characters)",
            duration,
            self._model,
            result.response.tone.value,
            len(result.response.text),
        )

        return result

