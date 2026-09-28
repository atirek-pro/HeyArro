"""Placeholder OpenAI vision provider.

Satisfies the VisionLLMProvider abstraction without implementing the OpenAI
API, and deliberately does not pull in the OpenAI SDK.
"""

from app.llm.provider import (
    ProviderNotImplementedError,
    VisionLLMProvider,
    VisionLLMRequest,
)
from app.llm.response import HeyArroResponse


class OpenAIVisionProvider(VisionLLMProvider):
    """Registered so the provider can be selected, but not implemented yet."""

    name = "openai"

    def process(self, request: VisionLLMRequest) -> HeyArroResponse:
        raise ProviderNotImplementedError(
            "The OpenAI vision provider is not implemented yet; "
            "set LLM_PROVIDER to gemini or mock."
        )
