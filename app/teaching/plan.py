"""The teaching plan: how Arro will teach something, as pure data.

A plan says what should be taught, in what order, how deeply, what Arro will say
for each step, and what should be shown while it says it. It is a data model
only: it never calls a model, speaks, draws, grounds a target, captures a screen,
or touches application state, and it holds no orchestration logic.

Visual actions are the application's existing ``app.llm.response.VisualAction``
objects, reused unchanged, so the grounding and the overlay that already
understand them will understand a plan's actions too.

Naming note
-----------
``app.llm.response.TeachingPlan`` is the teaching section of the *vision
provider's* answer: a mode plus a flat list of steps to speak. This module is the
richer plan Arro teaches from: the coordinator converts a response into one of
these, grounds every visual target it contains, and the sequence plays it. They
are deliberately separate types - the provider's answer never reaches the overlay
directly.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.llm.response import VisualAction
from app.teaching.difficulty import DEFAULT_DIFFICULTY, TeachingDifficulty


def _is_blank(value) -> bool:
    return not str(value or "").strip()


class TeachingStep(BaseModel):
    """One step of a teaching plan.

    A step is a self-contained unit of teaching: what it is for, what Arro says,
    what is shown while it is said, and how it hands over to the next one.
    """

    step_id: str = Field(
        description=(
            "Stable identifier for this step, for example 'step_1'. Identity comes from "
            "this, never from the step's position in the list."
        )
    )
    order: int = Field(
        ge=1, description="The step's place in the teaching sequence, starting at 1."
    )
    objective: str = Field(description="What the learner should understand after this step.")
    explanation: str = Field(
        description=(
            "The teaching sentence Arro will speak for this step - teaching language, not "
            "metadata or a reasoning trace."
        )
    )
    visual_actions: List[VisualAction] = Field(
        default_factory=list,
        description=(
            "What should be shown while this step is explained, using the application's "
            "existing visual actions. Empty when the step needs nothing on screen."
        ),
    )
    transition: Optional[str] = Field(
        default=None,
        description=(
            "A short optional hand-over to the next concept, for example 'Now that we've "
            "identified Matrix A, let's look at Matrix B.' Null when no hand-over is needed."
        ),
    )

    @field_validator("step_id", "objective", "explanation")
    @classmethod
    def _must_not_be_empty(cls, value, info):
        if _is_blank(value):
            raise ValueError(f"{info.field_name} cannot be empty")
        return value

    @field_validator("transition")
    @classmethod
    def _optional_text_must_not_be_blank(cls, value, info):
        if value is not None and _is_blank(value):
            raise ValueError(f"{info.field_name} cannot be blank when it is given; use null")
        return value


class TeachingPlan(BaseModel):
    """A complete plan for teaching one thing.

    Validation is strict but never creative: the model rejects data that does not
    make sense (a blank objective, no steps, two steps claiming the same order)
    and otherwise keeps exactly what it was given. It never reorders the steps
    and never rewrites their text.

    Step ordering
    -------------
    ``order`` must be a positive integer and no two steps may share one. The
    steps do not have to be numbered contiguously, and their position in the
    list is not significant - ``step_id`` is the identity and ``order`` is the
    sequence - so the model neither requires nor applies a sort.
    """

    objective: str = Field(
        description=(
            "What the user should understand by the end, for example 'Understand how matrix "
            "multiplication produces each value in the resulting matrix.'"
        )
    )
    difficulty: TeachingDifficulty = Field(
        default=DEFAULT_DIFFICULTY,
        description=(
            "How deeply to explain: the level the learner asked for, never an assessment of "
            "the learner. A plan without one is taught at the intermediate level."
        ),
    )
    introduction: str = Field(
        description="A short opening that establishes the concept before the steps."
    )
    steps: List[TeachingStep] = Field(
        min_length=1, description="The ordered teaching sequence; at least one step."
    )
    conclusion: Optional[str] = Field(
        default=None,
        description="A short optional close summarising what the user learned.",
    )

    @field_validator("objective", "introduction")
    @classmethod
    def _must_not_be_empty(cls, value, info):
        if _is_blank(value):
            raise ValueError(f"{info.field_name} cannot be empty")
        return value

    @field_validator("difficulty", mode="before")
    @classmethod
    def _missing_difficulty_is_the_default(cls, value):
        """Accept plans written before difficulty existed, and explicit nulls."""
        if value is None or _is_blank(value):
            return DEFAULT_DIFFICULTY
        return value

    @field_validator("conclusion")
    @classmethod
    def _optional_text_must_not_be_blank(cls, value, info):
        if value is not None and _is_blank(value):
            raise ValueError(f"{info.field_name} cannot be blank when it is given; use null")
        return value

    @model_validator(mode="after")
    def _step_orders_are_unique(self):
        """Reject two steps that claim the same place in the sequence."""
        orders = [step.order for step in self.steps]
        duplicates = sorted({order for order in orders if orders.count(order) > 1})
        if duplicates:
            raise ValueError(
                f"step order must be unique; duplicated order(s): "
                f"{', '.join(str(order) for order in duplicates)}"
            )
        return self


__all__ = ["TeachingPlan", "TeachingStep"]
