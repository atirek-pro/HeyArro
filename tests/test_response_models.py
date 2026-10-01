"""The structured answer schema: valid payloads validate, invalid ones do not.

The schema is deliberately only the answer: whatever a provider returns here is
shown to the user, so there is nothing in it that no one displays.
"""

import pytest
from pydantic import ValidationError

from app.llm.response import HeyArroResponse, ResponseContent, ResponseTone


def build(**overrides):
    payload = {"response": {"text": "An answer.", "tone": "neutral"}}
    payload.update(overrides)
    return payload


# --- valid answers ----------------------------------------------------------


def test_a_complete_answer_validates():
    result = HeyArroResponse.model_validate(build())

    assert result.response.text == "An answer."
    assert result.response.tone is ResponseTone.NEUTRAL


@pytest.mark.parametrize("tone", ["neutral", "friendly", "encouraging", "instructional"])
def test_every_tone_is_allowed(tone):
    result = HeyArroResponse.model_validate(build(response={"text": "Hi.", "tone": tone}))

    assert result.response.tone.value == tone


def test_the_answer_text_is_kept_exactly():
    text = "Line one.\n\nLine two: 2 + 2 = 4."

    result = HeyArroResponse.model_validate(build(response={"text": text, "tone": "friendly"}))

    assert result.response.text == text


def test_a_response_round_trips_through_json():
    """This is the schema the provider is asked to fill, so it must serialise."""
    payload = HeyArroResponse.model_validate(build()).model_dump_json()

    assert HeyArroResponse.model_validate_json(payload).response.text == "An answer."


# --- invalid answers --------------------------------------------------------


def test_an_empty_answer_is_rejected():
    """An empty answer is a failure, never a blank bubble on screen."""
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate(build(response={"text": "", "tone": "neutral"}))
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate(build(response={"text": "   ", "tone": "neutral"}))


def test_an_unknown_tone_is_rejected():
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate(build(response={"text": "Hi.", "tone": "shouty"}))


def test_the_answer_is_required():
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate({})
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate({"response": {"tone": "neutral"}})


def test_the_tone_is_required():
    with pytest.raises(ValidationError):
        HeyArroResponse.model_validate({"response": {"text": "Hi."}})


# --- the shape of the contract ----------------------------------------------


def test_the_schema_is_only_the_answer():
    assert ResponseContent.model_fields.keys() == {"text", "tone"}
    assert HeyArroResponse.model_fields.keys() == {"response"}
