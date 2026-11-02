"""Provider-neutral structured answer.

Every VisionLLMProvider returns a HeyArroResponse, so the rest of the
application depends on this type rather than on a model vendor's output format.
No vendor-specific concept belongs in this module, and an answer is validated
before it can leave a provider.

An answer is the text the user reads plus how it should sound. There is
deliberately nothing else: anything a provider returns here is shown.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ResponseTone(str, Enum):
    """How the answer should sound."""

    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    ENCOURAGING = "encouraging"
    INSTRUCTIONAL = "instructional"


class ResponseContent(BaseModel):
    """The answer itself and the tone it is given in."""

    text: str = Field(
        description=(
            "The complete answer to the user in natural language - the whole answer, "
            "not an introduction to one."
        )
    )
    tone: ResponseTone = Field(
        description="'neutral', 'friendly', 'encouraging' or 'instructional'."
    )

    @field_validator("text")
    @classmethod
    def _text_must_not_be_empty(cls, value):
        """An empty answer is a failure, never a blank bubble on screen."""
        if not str(value or "").strip():
            raise ValueError("text cannot be empty")
        return value


class HeyArroResponse(BaseModel):
    """The single structured answer the application consumes.

    A provider must never return an instance that did not validate, so an
    invalid model output becomes an error state instead of a plausible answer.
    """

    response: ResponseContent


__all__ = ["HeyArroResponse", "ResponseContent", "ResponseTone"]
