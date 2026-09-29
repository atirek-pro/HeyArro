"""The structured response schema: valid payloads validate, invalid ones do not."""

import pytest
from pydantic import ValidationError

from app.llm.response import (
    HeyArroResponse,
    ResponseContent,
    ResponseTone,
    TargetType,
    TeachingMode,
    TeachingPlan,
    TeachingStep,
    VisualAction,
    VisualActionType,
    VisualTarget,
)


def build(**overrides):
    payload = {
        "response": {"text": "An answer.", "tone": "neutral"},
        "teaching": {"mode": "direct", "steps": []},
    }
    payload.update(overrides)
    return payload


def action(kind="highlight", **target_fields):
    target = {"x": 10, "y": 20, "label": "thing"}
    target.update(target_fields)
    return {"type": kind, "target": target}


# --- valid responses --------------------------------------------------------


def test_valid_direct_response_validates():
    result = HeyArroResponse.model_validate(build())

    assert result.response.text == "An answer."
    assert result.response.tone is ResponseTone.NEUTRAL
    assert result.teaching.mode is TeachingMode.DIRECT
    assert result.teaching.steps == []


def test_valid_explanatory_response_validates():
    result = HeyArroResponse.model_validate(
        build(
            response={"text": "The error means the API key is missing.", "tone": "instructional"},
            teaching={"mode": "explanatory", "steps": []},
        )
    )

    assert result.response.tone is ResponseTone.INSTRUCTIONAL
    assert result.teaching.mode is TeachingMode.EXPLANATORY
    assert result.teaching.steps == []


def test_valid_guided_response_with_visual_actions():
    result = HeyArroResponse.model_validate(
        build(
            response={"text": "You can configure it in Settings.", "tone": "friendly"},
            teaching={
                "mode": "guided",
                "steps": [
                    {
                        "instruction": "Open Settings.",
                        "visual_actions": [
                            action("box", x=842, y=316, width=120, height=40, label="Settings"),
                            action("point", x=842, y=316, label=""),
                        ],
                    }
                ],
            },
        )
    )

    step = result.teaching.steps[0]
    assert result.teaching.mode is TeachingMode.GUIDED
    assert step.has_visuals is True
    assert [item.type for item in step.visual_actions] == [
        VisualActionType.BOX,
        VisualActionType.POINT,
    ]
    assert step.visual_actions[0].target.label == "Settings"
    assert step.visual_actions[1].target.label == ""


def test_a_step_can_carry_several_visual_actions():
    result = HeyArroResponse.model_validate(
        build(
            teaching={
                "mode": "guided",
                "steps": [
                    {
                        "instruction": "Multiply the matching values.",
                        "visual_actions": [
                            action("circle", x=1, y=2, width=40, height=30, label="1"),
                            action("circle", x=3, y=4, width=40, height=30, label="5"),
                        ],
                    }
                ],
            }
        )
    )

    assert len(result.teaching.steps[0].visual_actions) == 2


def test_a_step_may_have_no_visual_actions():
    step = TeachingStep.model_validate({"instruction": "Your API key is missing."})

    assert step.visual_actions == []
    assert step.has_visuals is False


def test_guided_response_can_contain_multiple_steps():
    result = HeyArroResponse.model_validate(
        build(
            teaching={
                "mode": "guided",
                "steps": [
                    {"instruction": "One.", "visual_actions": [action("highlight")]},
                    {"instruction": "Two.", "visual_actions": [action("box", label="B")]},
                    {"instruction": "Three."},
                ],
            }
        )
    )

    assert [step.instruction for step in result.teaching.steps] == ["One.", "Two.", "Three."]
    assert result.teaching.steps[2].visual_actions == []


# --- visual target geometry -------------------------------------------------


def test_a_target_with_bounds_is_a_region():
    target = VisualTarget.model_validate({"x": 100, "y": 50, "width": 200, "height": 80})

    assert target.type is TargetType.REGION
    assert (target.x, target.y) == (100, 50)
    assert (target.width, target.height) == (200, 80)


def test_a_target_without_bounds_is_a_point():
    target = VisualTarget.model_validate({"x": 842, "y": 316, "label": "Settings button"})

    assert target.type is TargetType.POINT
    assert target.width is None
    assert target.height is None
    assert target.screen_id is None


def test_partial_bounds_are_dropped_rather_than_guessed():
    target = VisualTarget.model_validate({"x": 1, "y": 2, "width": 100})

    assert target.type is TargetType.POINT
    assert target.width is None
    assert target.height is None


def test_non_positive_bounds_are_dropped():
    target = VisualTarget.model_validate({"x": 1, "y": 2, "width": 0, "height": -5})

    assert target.type is TargetType.POINT
    assert (target.width, target.height) == (None, None)


def test_label_and_confidence_are_optional():
    target = VisualTarget.model_validate({"x": 1, "y": 2})

    assert target.label == ""
    assert target.confidence is None

    confident = VisualTarget.model_validate({"x": 1, "y": 2, "confidence": 0.9})
    assert confident.confidence == 0.9


def test_visual_target_accepts_an_optional_screen_id():
    target = VisualTarget.model_validate({"x": 1, "y": 2, "screen_id": 2})

    assert target.screen_id == 2


# --- invalid responses ------------------------------------------------------


def test_invalid_tone_is_rejected():
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate(build(response={"text": "hi", "tone": "sarcastic"}))


def test_invalid_teaching_mode_is_rejected():
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate(build(teaching={"mode": "lecture", "steps": []}))


def test_invalid_action_type_is_rejected():
    with pytest.raises(ValidationError):
        VisualAction.model_validate({"type": "click", "target": {"x": 1, "y": 2}})


def test_invalid_target_type_is_rejected():
    with pytest.raises(ValidationError):
        VisualTarget.model_validate({"type": "polygon", "x": 1, "y": 2})


def test_an_action_without_a_target_is_rejected():
    with pytest.raises(ValidationError):
        VisualAction.model_validate({"type": "highlight"})


def test_invalid_pydantic_response_is_rejected():
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate({"response": {"text": "missing tone"}})


def test_missing_teaching_section_is_rejected():
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate({"response": {"text": "hi", "tone": "neutral"}})


def test_non_integer_coordinates_are_rejected():
    with pytest.raises(ValidationError):
        VisualTarget.model_validate({"x": "left", "y": 2})


# --- documented values ------------------------------------------------------


def test_tones_and_modes_match_the_documented_values():
    assert [tone.value for tone in ResponseTone] == [
        "neutral",
        "friendly",
        "encouraging",
        "instructional",
    ]
    assert [mode.value for mode in TeachingMode] == ["direct", "guided", "explanatory"]
    assert [value.value for value in TargetType] == ["point", "region"]


def test_supported_visual_action_types():
    assert [value.value for value in VisualActionType] == [
        "point",
        "highlight",
        "box",
        "circle",
        "underline",
    ]


def test_action_is_never_an_interaction():
    """Visual actions only show things; there is no click/type action."""
    assert "click" not in {value.value for value in VisualActionType}
    assert "type" not in {value.value for value in VisualActionType}


def test_response_content_is_not_gemini_specific():
    assert ResponseContent.model_fields.keys() == {"text", "tone"}
    assert TeachingPlan.model_fields.keys() == {"mode", "steps"}
