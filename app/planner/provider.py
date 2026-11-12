"""Provider-neutral response planning contract.

Planning is the step after segment selection: from the question's category, the
screenshot and the selected segments (with their pixel boxes) it decides HOW the
question should be answered - which points to make, in what order and where on
screen each one belongs. It writes no sentences of the answer itself. The
application depends on these types, never on a specific model vendor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from app.refinement.provider import ResponseCategory, SelectionResult


class ResponsePlanningError(RuntimeError):
    """Raised when a response plan cannot be created."""


class VisualAction(str, Enum):
    """What the interactive overlay should do on the screen for one step."""

    POINT = "point"
    CIRCLE = "circle"
    UNDERLINE = "underline"
    ARROW = "arrow"
    HIGHLIGHT = "highlight"
    DRAW = "draw"
    NONE = "none"


@dataclass(frozen=True)
class PlanResult:
    """The plan written for one question.

    ``plan_path`` is the ``response_plan.json`` that was written next to the
    screenshot and its manifests.
    """

    plan_path: str
    response_category: ResponseCategory
    step_count: int = 0
    seconds: float = 0.0


class ResponsePlanner(ABC):
    """Plans how to answer a question from its relevant segments."""

    @abstractmethod
    def plan(self, selection: SelectionResult) -> PlanResult:
        """Read ``selection.manifest_path`` and its screenshot, then write ``response_plan.json``."""
