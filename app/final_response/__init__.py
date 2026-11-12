"""Final response abstraction and provider selection."""

import logging

from app import config
from app.final_response.provider import FinalResponseError, FinalResponseResult, ResponseComposer

logger = logging.getLogger(__name__)

__all__ = [
    "FinalResponseError",
    "FinalResponseResult",
    "ResponseComposer",
    "create_response_composer",
]


def create_response_composer() -> ResponseComposer | None:
    """Build the response composer, or None when it is switched off or has no key."""
    if not config.FINAL_RESPONSE_ENABLED:
        return None

    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set; the final response is disabled")
        return None

    from app.final_response.gemini_composer import GeminiResponseComposer

    return GeminiResponseComposer()
