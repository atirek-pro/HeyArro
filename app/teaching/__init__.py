"""Visual teaching abstraction and service selection."""

from app import config
from app.teaching.coordinates import (
    CoordinateMapper,
    CoordinateMappingError,
    ScreenGeometry,
)
from app.teaching.converter import (
    TeachingPlanConversionError,
    answer_only_teaching_plan,
    response_to_teaching_plan,
)
from app.teaching.difficulty import (
    DEFAULT_DIFFICULTY,
    TeachingDifficulty,
    adjust_difficulty,
    detect_difficulty,
    difficulty_guidance,
)
from app.teaching.followup import (
    TeachingFollowUp,
    TeachingFollowUpType,
    build_follow_up_context,
    detect_follow_up,
    follow_up_difficulty,
    follow_up_guidance,
)
from app.teaching.primitives import VisualPrimitive, resolve_action, resolve_actions
from app.teaching.plan import TeachingPlan, TeachingStep
from app.teaching.service import (
    NullVisualTeachingService,
    OverlayVisualTeachingService,
    TeachingError,
    VisualTeachingService,
)
from app.teaching.sequence import TeachingSequenceService, build_plan_utterances
from app.teaching.steps import (
    has_bounds,
    iter_actions,
    iter_point_actions,
    normalize_label,
    step_actions,
)

__all__ = [
    "CoordinateMapper",
    "CoordinateMappingError",
    "DEFAULT_DIFFICULTY",
    "NullVisualTeachingService",
    "OverlayVisualTeachingService",
    "ScreenGeometry",
    "TeachingDifficulty",
    "TeachingError",
    "TeachingFollowUp",
    "TeachingFollowUpType",
    "TeachingPlan",
    "TeachingPlanConversionError",
    "TeachingSequenceService",
    "TeachingStep",
    "VisualPrimitive",
    "VisualTeachingService",
    "adjust_difficulty",
    "answer_only_teaching_plan",
    "build_follow_up_context",
    "build_plan_utterances",
    "create_visual_teaching_service",
    "detect_difficulty",
    "detect_follow_up",
    "difficulty_guidance",
    "follow_up_difficulty",
    "follow_up_guidance",
    "has_bounds",
    "iter_actions",
    "iter_point_actions",
    "normalize_label",
    "resolve_action",
    "resolve_actions",
    "response_to_teaching_plan",
    "step_actions",
]


def create_visual_teaching_service() -> VisualTeachingService:
    """Build the configured teaching service.

    The Qt overlay is imported here, on demand, so importing the teaching
    package never requires a display (tests and headless runs stay safe).
    """
    if not config.TEACHING_ENABLED:
        return NullVisualTeachingService()

    from app.teaching.overlay import TeachingOverlay, primary_screen_geometry

    screen = primary_screen_geometry()
    return OverlayVisualTeachingService(TeachingOverlay(screen), screen)
