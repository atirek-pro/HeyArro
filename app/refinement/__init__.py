"""Screenshot refinement abstraction and provider selection."""

import logging

from app import config
from app.refinement.provider import (
    RefinementResult,
    ScreenRefinementError,
    ScreenRefinementProvider,
    ScreenSegment,
    SegmentSelectionError,
    SegmentSelector,
    SelectionResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RefinementResult",
    "ScreenRefinementError",
    "ScreenRefinementProvider",
    "ScreenSegment",
    "SegmentSelectionError",
    "SegmentSelector",
    "SelectionResult",
    "create_refinement_provider",
    "create_segment_selector",
]


def create_refinement_provider() -> ScreenRefinementProvider:
    """Build the configured provider.

    The application depends on the ScreenRefinementProvider interface, so the
    concrete library is chosen here and can be swapped without touching the app.
    """
    from app.refinement.unstructured_provider import UnstructuredRefinementProvider

    return UnstructuredRefinementProvider()


def create_segment_selector() -> SegmentSelector | None:
    """Build the segment selector, or None when it is switched off or has no key."""
    if not config.SEGMENT_SELECTION_ENABLED:
        return None

    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set; relevant segment selection is disabled")
        return None

    from app.refinement.gemini_selector import GeminiSegmentSelector

    return GeminiSegmentSelector()
