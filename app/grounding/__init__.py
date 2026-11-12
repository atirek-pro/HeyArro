"""Plan grounding abstraction and provider selection."""

import logging

from app import config
from app.grounding.provider import GroundingResult, PlanGrounder, PlanGroundingError

logger = logging.getLogger(__name__)

__all__ = [
    "GroundingResult",
    "PlanGrounder",
    "PlanGroundingError",
    "create_plan_grounder",
]


def create_plan_grounder() -> PlanGrounder | None:
    """Build the plan grounder, or None when it is switched off or has no key."""
    if not config.PLAN_GROUNDING_ENABLED:
        return None

    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set; plan grounding is disabled")
        return None

    from app.grounding.gemini_grounder import GeminiPlanGrounder

    return GeminiPlanGrounder()
