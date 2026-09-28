"""Gemini vision provider - the first real VisionLLMProvider implementation.

Everything Gemini-specific lives here: the SDK import, the request payload and
the response decoding. Nothing outside this module needs to know about them.
"""

import logging
import time

from app import config
from app.llm.provider import (
    LLMError,
    ProviderConfigurationError,
    SYSTEM_INSTRUCTION,
    VisionLLMProvider,
    VisionLLMRequest,
    VisionLLMResponse,
    build_user_prompt,
    validate_screenshot,
)

logger = logging.getLogger(__name__)


def _response_text(response):
    """Decode a provider response without leaking SDK types outward."""
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

    def process(self, request: VisionLLMRequest) -> VisionLLMResponse:
        # Validated before the client is built, so a bad screenshot never
        # reaches the API.
        mime_type = validate_screenshot(request.screenshot)

        client = self._get_client()
        types = self._import_sdk()[1]

        parts = []
        if mime_type is not None:
            parts.append(types.Part.from_bytes(data=request.screenshot, mime_type=mime_type))
        parts.append(
            types.Part.from_text(
                text=build_user_prompt(request.transcript, mime_type is not None)
            )
        )

        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = _response_text(response)
        if not text:
            raise LLMError("Gemini returned an empty response")

        duration = round(time.perf_counter() - started, 3)
        logger.info("Gemini responded in %ss using model %s", duration, self._model)

        return VisionLLMResponse(text=text, model=self._model, duration=duration)

    def _import_sdk(self):
        if self._sdk is None:
            try:
                from google import genai
                from google.genai import types
            except Exception as exc:
                raise ProviderConfigurationError(f"google-genai is not available: {exc}") from exc
            self._sdk = (genai, types)
        return self._sdk

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
