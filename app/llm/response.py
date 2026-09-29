"""Provider-neutral structured teaching response.

Every VisionLLMProvider returns a HeyArroResponse, so the rest of the
application - the coordinator, the teaching sequence and the overlay - depends
on these types rather than on a model vendor's output format. No vendor-specific
concept belongs in this module, and every field is validated before a response
can leave a provider.

A teaching step says two things: what Arro says, and what the user should see
while it says it. The "what to see" part is a list of visual actions, because a
single sentence often needs more than one thing shown at once.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ResponseTone(str, Enum):
    """How the explanation should sound when it is spoken back."""

    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    ENCOURAGING = "encouraging"
    INSTRUCTIONAL = "instructional"


class TeachingMode(str, Enum):
    """The kind of teaching the response is doing."""

    # The user mainly needs a short, straightforward answer.
    DIRECT = "direct"
    # Walk the user through something using one or more visual steps.
    GUIDED = "guided"
    # Explain a concept or situation without requiring screen interaction.
    EXPLANATORY = "explanatory"


class TargetType(str, Enum):
    """What a target's geometry describes.

    Derived from the target itself: a target with a width and height is a
    ``region``, one with only a centre is a ``point``.
    """

    POINT = "point"
    REGION = "region"


class VisualActionType(str, Enum):
    """How a visual action shows its target.

    These are highlighting primitives, not interactions: showing the user
    something is not the same as acting on it.
    """

    # Move Arro's pointer to the spot.
    POINT = "point"
    # Soft filled region (or a radial glow around a point).
    HIGHLIGHT = "highlight"
    # Outlined rectangle around the region.
    BOX = "box"
    # Outlined ellipse/circle around the region.
    CIRCLE = "circle"
    # Emphasising line under a region (text, an expression, a row).
    UNDERLINE = "underline"


class VisualTarget(BaseModel):
    """A region of the screenshot a visual action refers to.

    Geometry is expressed in the pixel space of the screenshot the provider was
    given, not of the desktop as a whole, so it stays meaningful until
    multi-monitor pointer mapping is added.

    ``(x, y)`` is the CENTRE of the region. ``width``/``height`` are its size and
    are optional: give them when the extent is known, omit them when only a
    point can be located. ``type`` is derived from that - never set by hand.
    """

    type: TargetType = Field(
        default=TargetType.POINT,
        description="Derived: 'region' when width and height are given, otherwise 'point'.",
    )
    x: int = Field(
        description=(
            "Horizontal CENTRE of the region in the supplied screenshot, measured "
            "from the left edge."
        )
    )
    y: int = Field(
        description=(
            "Vertical CENTRE of the region in the supplied screenshot, measured "
            "from the top edge."
        )
    )
    width: Optional[int] = Field(
        default=None,
        description=(
            "Width of the region in screenshot pixels, when its extent is visible. "
            "Omit for a bare point."
        ),
    )
    height: Optional[int] = Field(
        default=None,
        description=(
            "Height of the region in screenshot pixels, when its extent is visible. "
            "Omit for a bare point."
        ),
    )
    label: str = Field(
        default="",
        description=(
            "Short name shown beside the visual, for example 'Settings' or 'first row'. "
            "May be empty."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        description="How sure you are that this region is correct, from 0 to 1.",
    )
    screen_id: Optional[int] = Field(
        default=None,
        description=(
            "Identifier of the screen the coordinates refer to, when the app "
            "captures more than one. Null means the screen of the supplied "
            "screenshot."
        ),
    )

    @model_validator(mode="after")
    def _normalise_geometry(self):
        """Derive ``type`` and drop half-given or nonsensical sizes."""
        width = self.width if (self.width or 0) > 0 else None
        height = self.height if (self.height or 0) > 0 else None

        if width is None or height is None:
            self.width = None
            self.height = None
            self.type = TargetType.POINT
        else:
            self.width = width
            self.height = height
            self.type = TargetType.REGION
        return self


class VisualAction(BaseModel):
    """One thing to show the user while a step's sentence is spoken."""

    type: VisualActionType = Field(
        description=(
            "'point', 'highlight', 'box', 'circle' or 'underline' - how to show the target."
        )
    )
    target: VisualTarget = Field(description="What to show, in screenshot pixels.")


class TeachingStep(BaseModel):
    """One unit of explanation: a sentence, and what to show while it is said."""

    instruction: str = Field(description="One concise, speakable teaching sentence.")
    visual_actions: List[VisualAction] = Field(
        default_factory=list,
        description=(
            "What the user should see while this sentence is spoken. Empty when nothing "
            "on the screenshot helps - a step with no visual is perfectly valid."
        ),
    )

    @property
    def has_visuals(self):
        """True when this step asks for anything to be drawn."""
        return bool(self.visual_actions)


class TeachingPlan(BaseModel):
    """How the answer should be taught, as zero or more ordered steps."""

    mode: TeachingMode = Field(
        description="'direct', 'guided' or 'explanatory'; see the system instructions."
    )
    steps: List[TeachingStep] = Field(
        default_factory=list,
        description="Ordered teaching steps; empty when no step-by-step guidance is needed.",
    )


class ResponseContent(BaseModel):
    """The spoken explanation and the tone it should be delivered in."""

    text: str = Field(
        description="The natural-language explanation to give the user, kept short enough to speak."
    )
    tone: ResponseTone = Field(
        description="'neutral', 'friendly', 'encouraging' or 'instructional'."
    )


class HeyArroResponse(BaseModel):
    """The single structured answer the application consumes.

    A provider must never return an instance that did not validate, so an
    invalid model output becomes an error state instead of a plausible answer.
    """

    response: ResponseContent
    teaching: TeachingPlan


__all__ = [
    "HeyArroResponse",
    "ResponseContent",
    "ResponseTone",
    "TargetType",
    "TeachingMode",
    "TeachingPlan",
    "TeachingStep",
    "VisualAction",
    "VisualActionType",
    "VisualTarget",
]
