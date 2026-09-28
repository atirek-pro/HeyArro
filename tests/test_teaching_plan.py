"""The TeachingPlan data model: valid plans validate, invalid ones do not."""

import inspect
from pathlib import Path
from typing import List

import pytest
from pydantic import ValidationError

import app.teaching.plan as plan_module
from app.llm.response import VisualAction, VisualActionType, VisualTarget
from app.teaching import TeachingDifficulty, TeachingPlan, TeachingStep


def build_step(order=1, step_id=None, actions=None, transition=None, **overrides):
    payload = {
        "step_id": f"step_{order}" if step_id is None else step_id,
        "order": order,
        "objective": f"Understand part {order}.",
        "explanation": f"This is the explanation for step {order}.",
        "visual_actions": actions if actions is not None else [],
        "transition": transition,
    }
    payload.update(overrides)
    return payload


def build_plan(**overrides):
    payload = {
        "objective": "Understand how matrix multiplication produces each value.",
        "introduction": "Let's break matrix multiplication into one simple calculation.",
        "steps": [build_step(1), build_step(2), build_step(3)],
        "conclusion": "Each result value combines one row with one column.",
    }
    payload.update(overrides)
    return payload


def action(x=10, y=20, label="Matrix A"):
    return VisualAction(
        type=VisualActionType.BOX,
        target=VisualTarget(x=x, y=y, width=120, height=80, label=label),
    )


# --- valid plans ------------------------------------------------------------


def test_a_complete_plan_validates():
    plan = TeachingPlan.model_validate(build_plan())

    assert plan.objective.startswith("Understand how matrix multiplication")
    assert plan.introduction.startswith("Let's break")
    assert [step.order for step in plan.steps] == [1, 2, 3]
    assert [step.step_id for step in plan.steps] == ["step_1", "step_2", "step_3"]
    assert plan.conclusion.startswith("Each result value")


def test_a_plan_defaults_to_the_intermediate_level():
    assert TeachingPlan.model_validate(build_plan()).difficulty is TeachingDifficulty.INTERMEDIATE


def test_a_plan_can_carry_the_level_it_was_asked_for():
    plan = TeachingPlan.model_validate(build_plan(difficulty="beginner"))

    assert plan.difficulty is TeachingDifficulty.BEGINNER


def test_a_plan_saved_before_difficulty_existed_still_validates():
    """An older plan simply has no level, and gets the neutral one."""
    payload = build_plan()
    payload.pop("difficulty", None)

    assert TeachingPlan.model_validate(payload).difficulty is TeachingDifficulty.INTERMEDIATE
    assert (
        TeachingPlan.model_validate(build_plan(difficulty=None)).difficulty
        is TeachingDifficulty.INTERMEDIATE
    )


def test_an_unknown_level_is_rejected():
    with pytest.raises(ValidationError):
        TeachingPlan.model_validate(build_plan(difficulty="expert"))


def test_a_plan_with_two_steps_validates():
    plan = TeachingPlan.model_validate(
        build_plan(steps=[build_step(1), build_step(2)], conclusion=None)
    )

    assert len(plan.steps) == 2
    assert plan.conclusion is None


def test_a_conclusion_is_optional():
    plan = TeachingPlan.model_validate(
        {
            "objective": "Understand X.",
            "introduction": "Here is X.",
            "steps": [build_step(1)],
        }
    )

    assert plan.conclusion is None


def test_a_transition_is_optional():
    without = TeachingStep.model_validate(build_step(1, transition=None))
    omitted = TeachingStep.model_validate(
        {"step_id": "step_1", "order": 1, "objective": "X.", "explanation": "X is X."}
    )

    assert without.transition is None
    assert omitted.transition is None


def test_a_transition_may_be_given():
    step = TeachingStep.model_validate(
        build_step(1, transition="Now that we've identified Matrix A, let's look at Matrix B.")
    )

    assert step.transition.startswith("Now that we've identified")


def test_steps_keep_the_order_they_were_given_in():
    """The model validates; it never reorders or rewrites what it was handed."""
    plan = TeachingPlan.model_validate(
        build_plan(steps=[build_step(3), build_step(1), build_step(2)])
    )

    assert [step.order for step in plan.steps] == [3, 1, 2]


def test_step_orders_do_not_have_to_be_contiguous():
    plan = TeachingPlan.model_validate(
        build_plan(steps=[build_step(10, step_id="intro"), build_step(20, step_id="body")])
    )

    assert [step.order for step in plan.steps] == [10, 20]


# --- invalid plans ----------------------------------------------------------


def test_an_empty_objective_is_rejected():
    with pytest.raises(ValidationError):
        TeachingPlan.model_validate(build_plan(objective=""))

    with pytest.raises(ValidationError):
        TeachingPlan.model_validate(build_plan(objective="   "))


def test_an_empty_introduction_is_rejected():
    with pytest.raises(ValidationError):
        TeachingPlan.model_validate(build_plan(introduction=""))


def test_a_plan_without_steps_is_rejected():
    with pytest.raises(ValidationError):
        TeachingPlan.model_validate(build_plan(steps=[]))


def test_a_blank_conclusion_is_rejected_but_null_is_fine():
    with pytest.raises(ValidationError):
        TeachingPlan.model_validate(build_plan(conclusion="  "))

    assert TeachingPlan.model_validate(build_plan(conclusion=None)).conclusion is None


def test_an_order_below_one_is_rejected():
    with pytest.raises(ValidationError):
        TeachingStep.model_validate(build_step(0))


def test_duplicate_orders_are_rejected():
    with pytest.raises(ValidationError) as info:
        TeachingPlan.model_validate(
            build_plan(steps=[build_step(1), build_step(1, step_id="other_step")])
        )

    assert "unique" in str(info.value)


def test_an_empty_step_explanation_is_rejected():
    with pytest.raises(ValidationError):
        TeachingStep.model_validate(build_step(1, explanation=""))


def test_an_empty_step_objective_is_rejected():
    with pytest.raises(ValidationError):
        TeachingStep.model_validate(build_step(1, objective="   "))


def test_an_empty_step_id_is_rejected():
    with pytest.raises(ValidationError):
        TeachingStep.model_validate(build_step(1, step_id=""))


def test_a_blank_transition_is_rejected_but_null_is_fine():
    with pytest.raises(ValidationError):
        TeachingStep.model_validate(build_step(1, transition="   "))

    assert TeachingStep.model_validate(build_step(1, transition=None)).transition is None


def test_a_step_order_must_be_an_integer():
    with pytest.raises(ValidationError):
        TeachingStep.model_validate(build_step("first"))


# --- visual actions are the existing model ----------------------------------


def test_a_step_carries_the_existing_visual_actions():
    plan = TeachingPlan.model_validate(
        build_plan(steps=[build_step(1, actions=[action(1, 2, "Matrix A"), action(3, 4, "Matrix B")])])
    )

    step = plan.steps[0]
    assert len(step.visual_actions) == 2
    assert all(isinstance(item, VisualAction) for item in step.visual_actions)
    assert step.visual_actions[0].target.label == "Matrix A"
    assert step.visual_actions[1].target.type.value == "region"


def test_no_duplicate_visual_action_model_exists():
    """The plan reuses VisualAction; it does not define its own."""
    annotation = TeachingStep.model_fields["visual_actions"].annotation

    assert annotation == List[VisualAction]
    for name in ("TeachingVisualAction", "PlanVisualAction", "TeachingTarget", "PlanTarget"):
        assert not hasattr(plan_module, name)


def test_the_plan_step_is_a_different_type_from_the_response_step():
    """The provider's step and the plan's step are deliberately separate."""
    from app.llm.response import TeachingStep as ResponseTeachingStep

    assert TeachingStep is not ResponseTeachingStep
    assert set(TeachingStep.model_fields) == {
        "step_id",
        "order",
        "objective",
        "explanation",
        "visual_actions",
        "transition",
    }


# --- it is only data --------------------------------------------------------


def test_the_plan_model_is_independent_of_execution():
    source = Path(plan_module.__file__).read_text(encoding="utf-8").lower()

    for forbidden in (
        "from app.tts",
        "from app.ui",
        "from app.visual_grounding",
        "from app.coordinator",
        "import asyncio",
        "pyside6",
    ):
        assert forbidden not in source


def test_the_plan_model_has_no_execution_methods():
    for model in (TeachingPlan, TeachingStep):
        assert not any(
            inspect.iscoroutinefunction(member) for member in vars(model).values()
        )
        for name in ("speak", "show", "ground", "execute", "run", "capture"):
            assert not hasattr(model, name)


def test_visual_actions_default_to_empty():
    step = TeachingStep.model_validate(
        {"step_id": "step_1", "order": 1, "objective": "X.", "explanation": "X is X."}
    )

    assert step.visual_actions == []
