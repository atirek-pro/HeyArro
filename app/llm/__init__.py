"""Vision LLM abstraction and centralized provider selection."""

import importlib
import logging

from app import config
from app.llm.provider import (
    LLMError,
    ProviderConfigurationError,
    ProviderNotImplementedError,
    UnsupportedProviderError,
    VisionLLMProvider,
    VisionLLMRequest,
    VisionLLMResponse,
)

logger = logging.getLogger(__name__)

# provider name -> (module, class). Modules are imported on demand, so no
# vendor SDK is loaded unless that provider is actually selected.
PROVIDER_REGISTRY = {
    "gemini": ("app.llm.gemini_provider", "GeminiVisionProvider"),
    "openai": ("app.llm.openai_provider", "OpenAIVisionProvider"),
    "claude": ("app.llm.claude_provider", "ClaudeVisionProvider"),
    "mock": ("app.llm.mock_provider", "MockVisionLLMProvider"),
}

__all__ = [
    "LLMError",
    "PROVIDER_REGISTRY",
    "ProviderConfigurationError",
    "ProviderNotImplementedError",
    "UnsupportedProviderError",
    "VisionLLMProvider",
    "VisionLLMRequest",
    "VisionLLMResponse",
    "available_providers",
    "get_llm_provider",
]


def available_providers():
    """Return the registered provider names."""
    return sorted(PROVIDER_REGISTRY)


def get_llm_provider(provider_name=None) -> VisionLLMProvider:
    """Build the provider for ``provider_name``, defaulting to the configured one.

    This is the only place that knows which concrete providers exist, so the
    rest of the application can depend on VisionLLMProvider alone.
    """
    name = (provider_name or config.LLM_PROVIDER or "").strip().lower()

    if name not in PROVIDER_REGISTRY:
        raise UnsupportedProviderError(
            f"Unsupported LLM provider {name!r}. "
            f"Available providers: {', '.join(available_providers())}"
        )

    module_name, class_name = PROVIDER_REGISTRY[name]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ProviderConfigurationError(f"Could not load provider {name!r}: {exc}") from exc

    provider = getattr(module, class_name)()
    logger.info("LLM provider selected: %s (%s)", name, class_name)
    return provider
