"""HeyArroResponse -> TeachingPlan: the conversion bridge."""

from pathlib import Path

import pytest

import app.teaching.converter as converter_module
from app.llm.mock_provider import MockVisionLLMProvider
from app.llm.provider import VisionLLMRequest
from app.llm.response import (
    HeyArroResponse,
    ResponseContent,
    ResponseTone,
    TeachingMode,
    TeachingPlan as ResponseTeachingPlan,
    TeachingStep as ResponseTeachingStep,
    VisualAction,
    VisualActionType,
    VisualTarget,
)
from app.teaching import (
    TeachingDifficulty,
    TeachingPlan,
    TeachingPlanConversionError,
    answer_only_teaching_plan,
    response_to_teaching_plan,
)

# --- builders ---------------------------------------------------------------


def action(kind="box", x=100, y=200, width=120, height=80, label="Matrix A"):
    return VisualAction(
        type=VisualActionType(kind),
        target=VisualTarget(x=x, y=y, width=width, height=height, label=label),
    )


def response(
    text="Multiplying a row by a column.",
    instructions=("Start with the first row of the left matrix.", "Now take the first column of the right matrix."),
    actions=None,
    mode=TeachingMode.GUIDED,
):
    actions = actions if actions is not None else {}
    steps = [
        ResponseTeachingStep(instruction=instruction, visual_actions=actions.get(index, []))
        for index, instruction in enumerate(instructions)
    ]
    return HeyArroResponse(
        response=ResponseContent(text=text, tone=ResponseTone.INSTRUCTIONAL),
        teaching=ResponseTeachingPlan(mode=mode, steps=steps),
    )


# --- the basics -------------------------------------------------------------


def test_a_response_converts_into_a_plan_with_the_same_steps():
    source = response()

    plan = response_to_teaching_plan(source)

    assert isinstance(plan, TeachingPlan)
    assert len(plan.steps) == 2
    assert [step.order for step in plan.steps] == [1, 2]
    assert [step.explanation for step in plan.steps] == [
        "Start with the first row of the left matrix.",
        "Now take the first column of the right matrix.",
    ]


def test_step_ids_are_deterministic():
    plan = response_to_teaching_plan(
        response(instructions=("One.", "Two.", "Three."))
    )

    assert [step.step_id for step in plan.steps] == ["step_1", "step_2", "step_3"]


def test_the_teaching_order_is_preserved():
    plan = response_to_teaching_plan(response(instructions=("One.", "Two.", "Three.", "Four.", "Five.")))

    assert [step.order for step in plan.steps] == [1, 2, 3, 4, 5]
    assert [step.explanation for step in plan.steps] == ["One.", "Two.", "Three.", "Four.", "Five."]


def test_a_single_step_response_converts():
    plan = response_to_teaching_plan(response(instructions=("Only one step.",)))

    assert len(plan.steps) == 1
    assert plan.steps[0].order == 1


def test_the_objective_and_introduction_come_from_the_response():
    plan = response_to_teaching_plan(response(text="Multiplying a row by a column."))

    assert plan.objective == "Understand: Multiplying a row by a column."
    assert plan.introduction == "Multiplying a row by a column."


def test_the_converted_plan_carries_the_level_the_learner_asked_for():
    plan = response_to_teaching_plan(response(), TeachingDifficulty.BEGINNER)

    assert plan.difficulty is TeachingDifficulty.BEGINNER


def test_a_conversion_without_a_level_is_neutral():
    assert response_to_teaching_plan(response()).difficulty is TeachingDifficulty.INTERMEDIATE


def test_an_answer_only_plan_carries_the_level_too():
    plan = answer_only_teaching_plan("Yes, that is right.", TeachingDifficulty.ADVANCED)

    assert plan.difficulty is TeachingDifficulty.ADVANCED


def test_a_step_objective_comes_from_its_instruction():
    plan = response_to_teaching_plan(response(instructions=("Look at the first row of Matrix A.",)))

    assert plan.steps[0].objective == "Look at the first row of Matrix A."
    assert plan.steps[0].explanation == "Look at the first row of Matrix A."


# --- visual actions ---------------------------------------------------------


def test_visual_actions_are_carried_over_untouched():
    source = response(
        actions={
            0: [action("box", 100, 200, 120, 80, "first row")],
            1: [action("point", 300, 400, None, None, "")],
        }
    )

    plan = response_to_teaching_plan(source)

    # The very same objects, not copies.
    assert plan.steps[0].visual_actions[0] is source.teaching.steps[0].visual_actions[0]
    assert plan.steps[1].visual_actions[0] is source.teaching.steps[1].visual_actions[0]

    first = plan.steps[0].visual_actions[0]
    assert isinstance(first, VisualAction)
    assert first.type is VisualActionType.BOX
    assert first.target.label == "first row"


def test_coordinates_are_not_touched():
    """Nothing here grounds or refines a target."""
    source = response(actions={0: [action("box", 137, 246, 311, 92, "panel")]})

    target = response_to_teaching_plan(source).steps[0].visual_actions[0].target

    assert (target.x, target.y) == (137, 246)
    assert (target.width, target.height) == (311, 92)
    assert target.type.value == "region"


def test_a_point_target_stays_a_point():
    source = response(actions={0: [action("circle", 42, 24, None, None, "dot")]})

    target = response_to_teaching_plan(source).steps[0].visual_actions[0].target

    assert (target.x, target.y) == (42, 24)
    assert (target.width, target.height) == (None, None)
    assert target.type.value == "point"


def test_a_step_may_have_no_visual_actions():
    plan = response_to_teaching_plan(response())

    assert [step.visual_actions for step in plan.steps] == [[], []]


# --- what is deliberately not invented --------------------------------------


def test_transitions_are_not_invented():
    plan = response_to_teaching_plan(response(instructions=("One.", "Two.", "Three.")))

    assert [step.transition for step in plan.steps] == [None, None, None]


def test_the_conclusion_is_not_invented():
    assert response_to_teaching_plan(response()).conclusion is None


# --- failures ---------------------------------------------------------------


def test_a_response_without_steps_cannot_be_converted():
    with pytest.raises(TeachingPlanConversionError) as info:
        response_to_teaching_plan(response(text="Just an answer.", instructions=()))

    assert "no teaching steps" in str(info.value)


def test_a_response_with_no_text_at_all_cannot_be_converted():
    with pytest.raises(TeachingPlanConversionError) as info:
        response_to_teaching_plan(response(text="", instructions=("   ",)))

    assert "no text to teach from" in str(info.value)


def test_a_blank_instruction_falls_back_to_the_response_text():
    plan = response_to_teaching_plan(
        response(text="Multiplying a row by a column.", instructions=("", "Real instruction."))
    )

    assert plan.steps[0].explanation == "Multiplying a row by a column."
    assert plan.steps[1].explanation == "Real instruction."


def test_a_direct_answer_with_no_steps_from_the_real_provider_is_rejected():
    """Today's providers can legitimately answer without teaching steps."""
    generated = MockVisionLLMProvider(delay_seconds=0).process(VisionLLMRequest("hello"))

    with pytest.raises(TeachingPlanConversionError):
        response_to_teaching_plan(generated)


# --- determinism and purity -------------------------------------------------


def test_conversion_is_deterministic():
    source = response(
        actions={0: [action("box", 100, 200, 120, 80, "first row")]},
    )

    first = response_to_teaching_plan(source)
    second = response_to_teaching_plan(source)

    assert first == second
    assert [step.step_id for step in first.steps] == [step.step_id for step in second.steps]


def test_the_source_response_is_not_mutated():
    source = response(actions={0: [action("box", 100, 200, 120, 80, "first row")]})
    before = source.model_dump()

    response_to_teaching_plan(source)

    assert source.model_dump() == before


def test_the_converter_is_offline_and_deterministic():
    """No model, no network, no clock and no randomness in this layer."""
    source = Path(converter_module.__file__).read_text(encoding="utf-8").lower()

    for forbidden in ("gemini", "genai", "requests", "urllib", "socket", "uuid", "random"):
        assert forbidden not in source


# --- integration ------------------------------------------------------------


def test_the_converted_plan_passes_pydantic_validation():
    plan = response_to_teaching_plan(
        response(actions={0: [action("box", 100, 200, 120, 80, "first row")]})
    )

    revalidated = TeachingPlan.model_validate(plan.model_dump())

    assert revalidated == plan
    assert len(revalidated.steps) == 2


def test_a_full_lesson_converts_end_to_end():
    source = response(
        text="Multiplying a row by a column.",
        instructions=(
            "Start with the first row of the left matrix.",
            "Now take the first column of the right matrix.",
            "Multiply the matching values.",
        ),
        actions={
            0: [action("highlight", 440, 265, 320, 70, "first row of A")],
            1: [action("box", 1060, 500, 90, 260, "first column of B")],
            2: [
                action("circle", 560, 300, 70, 60, "1"),
                action("circle", 1060, 430, 70, 60, "5"),
            ],
        },
    )

    plan = response_to_teaching_plan(source)

    assert plan.objective == "Understand: Multiplying a row by a column."
    assert plan.introduction == "Multiplying a row by a column."
    assert len(plan.steps) == 3
    assert [len(step.visual_actions) for step in plan.steps] == [1, 1, 2]
    assert plan.steps[2].visual_actions[0].target.label == "1"
    assert plan.conclusion is None
