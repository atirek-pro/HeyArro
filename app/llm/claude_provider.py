"""Placeholder Claude vision provider.

Satisfies the VisionLLMProvider abstraction without implementing the Anthropic
API, and deliberately does not pull in the Anthropic SDK.
"""

from app.llm.provider import (
    ProviderNotImplementedError,
    VisionLLMProvider,
    VisionLLMRequest,
    VisionLLMResponse,
)


class ClaudeVisionProvider(VisionLLMProvider):
    """Registered so the provider can be selected, but not implemented yet."""

    name = "claude"

    def process(self, request: VisionLLMRequest) -> VisionLLMResponse:
        raise ProviderNotImplementedError(
            "The Claude vision provider is not implemented yet; "
            "set LLM_PROVIDER to gemini or mock."
        )
