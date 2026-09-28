"""Visual grounding: turning approximate visual targets into precise ones.

The first vision pass decides what to show; this package decides where it is by
cropping a high-resolution region around the first pass's guess and asking the
vision model to locate the target inside that crop.

Nothing here knows about the overlay or about speech: grounding produces a
refined ``VisualTarget`` in screenshot pixels, exactly the representation the
overlay already consumes.
"""

import logging

from app import config
from app.visual_grounding.base import GroundingError, VisualGrounder
from app.visual_grounding.coordinate_mapper import GroundingCoordinateMapper
from app.visual_grounding.cropper import ScreenshotCropper, annotate, crop_region
from app.visual_grounding.grounder import (
    GroundingCache,
    VisualGroundingService,
    cache_key,
    ground_teaching_plan,
)
from app.visual_grounding.models import (
    CropResult,
    GroundedTarget,
    GroundingRequest,
    GroundingStatus,
    Rect,
    Refinement,
    rect_from_target,
    target_description,
    target_from_rect,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CropResult",
    "GroundedTarget",
    "GroundingCache",
    "GroundingCoordinateMapper",
    "GroundingError",
    "GroundingRequest",
    "GroundingStatus",
    "Rect",
    "Refinement",
    "ScreenshotCropper",
    "VisualGrounder",
    "VisualGroundingService",
    "annotate",
    "cache_key",
    "create_visual_grounder",
    "crop_region",
    "ground_teaching_plan",
    "rect_from_target",
    "target_description",
    "target_from_rect",
]


def create_visual_grounder() -> VisualGrounder | None:
    """Build the configured grounder, or None when grounding should not run.

    Grounding is skipped (and the application behaves exactly as it did before
    this phase) when it is switched off, when there is nothing visual to refine,
    or when there is no Gemini key to refine with.
    """
    if not config.VISUAL_GROUNDING_ENABLED:
        logger.info("Visual grounding off: VISUAL_GROUNDING_ENABLED=false")
        return None

    if not config.TEACHING_ENABLED:
        logger.info("Visual grounding off: visual teaching is disabled")
        return None

    if not config.GEMINI_API_KEY:
        logger.info("Visual grounding off: no GEMINI_API_KEY to refine targets with")
        return None

    from app.visual_grounding.gemini_grounder import GeminiTargetLocator

    return VisualGroundingService(locator=GeminiTargetLocator(), cropper=ScreenshotCropper())
