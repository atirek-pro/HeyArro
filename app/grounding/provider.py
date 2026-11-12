"""Provider-neutral contract for turning a plan's targets into screen coordinates.

Grounding is the step after planning: each plan step names a ``target`` (the
element or area it points at) and this step finds where that is on the
screenshot. It writes no sentences and changes no plan. The application depends
on these types, never on a specific model vendor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.planner.provider import PlanResult


class PlanGroundingError(RuntimeError):
    """Raised when a plan's targets cannot be grounded at all."""


@dataclass(frozen=True)
class GroundingResult:
    """The pointers found for one plan.

    ``pointers_path`` is the ``plan_pointers.json`` that was written next to the
    plan. ``located`` of ``total`` steps that need a mark were found on screen.
    """

    pointers_path: str
    located: int = 0
    total: int = 0
    seconds: float = 0.0


class PlanGrounder(ABC):
    """Finds the screen coordinates of every target in a response plan."""

    @abstractmethod
    def ground(self, plan: PlanResult) -> GroundingResult:
        """Read ``plan.plan_path`` and its screenshot, then write ``plan_pointers.json``."""
