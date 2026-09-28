"""Text-to-speech abstraction and centralized provider selection."""

import importlib
import logging

from app import config
from app.tts.provider import (
    TTSError,
    TTSNotImplementedError,
    TTSProvider,
    UnsupportedTTSProviderError,
    normalize_text,
)

logger = logging.getLogger(__name__)

# provider name -> (module, class). Modules are imported on demand, so no engine
# is loaded unless that provider is actually selected.
PROVIDER_REGISTRY = {
    "windows": ("app.tts.windows_provider", "WindowsTTSProvider"),
    "elevenlabs": ("app.tts.elevenlabs_provider", "ElevenLabsTTSProvider"),
    "mock": ("app.tts.mock_provider", "MockTTSProvider"),
}

__all__ = [
    "PROVIDER_REGISTRY",
    "TTSError",
    "TTSNotImplementedError",
    "TTSProvider",
    "UnsupportedTTSProviderError",
    "available_tts_providers",
    "get_tts_provider",
    "normalize_text",
]


def available_tts_providers():
    """Return the registered provider names."""
    return sorted(PROVIDER_REGISTRY)


def get_tts_provider(provider_name=None) -> TTSProvider:
    """Build the provider for ``provider_name``, defaulting to the configured one.

    This is the only place that knows which concrete providers exist, so the
    rest of the application can depend on TTSProvider alone.
    """
    name = (provider_name or config.TTS_PROVIDER or "").strip().lower()

    if name not in PROVIDER_REGISTRY:
        raise UnsupportedTTSProviderError(
            f"Unsupported TTS provider {name!r}. "
            f"Available providers: {', '.join(available_tts_providers())}"
        )

    module_name, class_name = PROVIDER_REGISTRY[name]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise TTSError(f"Could not load TTS provider {name!r}: {exc}") from exc

    provider = getattr(module, class_name)()
    logger.info("TTS provider selected: %s (%s)", name, class_name)
    return provider
