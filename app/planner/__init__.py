"""Response planning abstraction and provider selection."""

import logging

from app import config
from app.planner.provider import (
    PlanResult,
    ResponsePlanner,
    ResponsePlanningError,
    VisualAction,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PlanResult",
    "ResponsePlanner",
    "ResponsePlanningError",
    "VisualAction",
    "create_response_planner",
]


def create_response_planner() -> ResponsePlanner | None:
    """Build the response planner, or None when it is switched off or has no key."""
    if not config.RESPONSE_PLANNING_ENABLED:
        return None

    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set; response planning is disabled")
        return None

    from app.planner.gemini_planner import GeminiResponsePlanner

    return GeminiResponsePlanner()
