"""Models for the visual grounding pipeline.

Grounding turns an approximate target from the first vision pass into a precise
one by cropping a high-resolution region and asking the vision model to locate
the target inside that crop. These models describe that process, and are the
only place where grounding geometry is defined.

Geometry convention: internally a rectangle is stored as **edges**
(``left/top/right/bottom``), because the whole pipeline is about translating
edges between frames. The response model (``VisualTarget``) uses centre+size, so
the two small converters at the bottom of this module bridge the two.

Coordinate frames are documented in ``coordinate_mapper.py``.
"""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from app.llm.response import VisualTarget


class GroundingStatus(str, Enum):
    """What happened to one target."""

    # The refinement was accepted and replaced the approximate target.
    REFINED = "refined"
    # A refinement came back but was not trustworthy, so it was discarded.
    REFINEMENT_REJECTED = "refinement_rejected"
    # No refinement was attempted or it failed; the approximate target stands.
    APPROXIMATE = "approximate"


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in pixels, stored as edges."""

    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_center_size(cls, x, y, width=0, height=0):
        """Build a rectangle from the centre-based geometry used by VisualTarget.

        A target with no width/height becomes a zero-sized rectangle at its
        point, which the cropper then expands by the padding.
        """
        left = int(x) - int(width) // 2
        top = int(y) - int(height) // 2
        return cls(left, top, left + int(width), top + int(height))

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def centre(self):
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def is_empty(self):
        return self.width <= 0 or self.height <= 0

    def expand(self, padding):
        """Grow by ``padding`` pixels on every side."""
        return Rect(
            self.left - padding, self.top - padding, self.right + padding, self.bottom + padding
        )

    def shifted(self, dx, dy):
        return Rect(self.left + dx, self.top + dy, self.right + dx, self.bottom + dy)

    def clamp(self, width, height):
        """Clip the rectangle to a ``width`` x ``height`` frame."""
        left = max(0, min(self.left, width))
        top = max(0, min(self.top, height))
        return Rect(left, top, max(left, min(self.right, width)), max(top, min(self.bottom, height)))

    def as_tuple(self):
        return (self.left, self.top, self.width, self.height)

    def describe(self):
        return f"x={self.left}, y={self.top}, w={self.width}, h={self.height}"


@dataclass(frozen=True)
class CropResult:
    """A crop of a screenshot, plus where that crop came from."""

    image: bytes
    origin_x: int
    origin_y: int
    width: int
    height: int

    @property
    def rect(self):
        """The crop's rectangle in screenshot coordinates."""
        return Rect(self.origin_x, self.origin_y, self.origin_x + self.width, self.origin_y + self.height)

    @property
    def origin(self):
        return (self.origin_x, self.origin_y)

    @property
    def size(self):
        return (self.width, self.height)


@dataclass(frozen=True)
class GroundingRequest:
    """One approximate target to refine, with everything the pipeline needs."""

    screenshot: bytes
    screenshot_size: tuple[int, int]
    # What the refiner is asked to find. The teaching sentence is folded in here
    # for context, which is why it is not used as the target's identity.
    description: str
    # The first pass's best guess, in screenshot pixels. May be a zero-sized
    # rectangle when the first pass only produced a point.
    approximate: Rect
    # The target's own short name, used to recognise the same target across
    # steps. May be empty.
    label: str = ""


class GroundedTarget(BaseModel):
    """The refinement model's answer, in CROP pixels.

    This is the structured schema handed to the vision model, so the field
    descriptions double as the specification it is asked to follow.
    """

    x: int = Field(
        description=(
            "Left edge of the target's bounding box in crop pixels: 0 is the crop's "
            "left edge."
        )
    )
    y: int = Field(
        description=(
            "Top edge of the target's bounding box in crop pixels: 0 is the crop's "
            "top edge."
        )
    )
    width: int = Field(description="Width of the target's bounding box in crop pixels.")
    height: int = Field(description="Height of the target's bounding box in crop pixels.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are that this box covers the requested target, from 0 to 1. "
            "Use a low value when you are unsure."
        ),
    )

    @property
    def rect(self):
        """The bounding box as edges, in crop pixels."""
        return Rect(self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class Refinement:
    """The pipeline's answer for one target, in screenshot pixels.

    Only a ``REFINED`` result may replace the approximate target; for the other
    statuses the caller keeps what the first pass produced. ``rect`` is always
    populated so it can still be logged.
    """

    rect: Rect
    status: GroundingStatus
    confidence: float | None = None

    @property
    def accepted(self):
        return self.status is GroundingStatus.REFINED


# --- bridges to the response model ------------------------------------------


def rect_from_target(target) -> Rect | None:
    """Return the approximate rectangle for a ``VisualTarget``, if it is usable."""
    x = getattr(target, "x", None)
    y = getattr(target, "y", None)
    if not isinstance(x, int) or not isinstance(y, int):
        return None
    return Rect.from_center_size(
        x, y, getattr(target, "width", None) or 0, getattr(target, "height", None) or 0
    )


def target_from_rect(rect: Rect, original=None, confidence=None) -> VisualTarget:
    """Rebuild a ``VisualTarget`` from a refined rectangle.

    The label and screen id of the original target are kept; only the geometry
    (and the new confidence) come from the refinement. The response model
    derives ``type`` from the presence of a size, as always.
    """
    centre_x, centre_y = rect.centre
    return VisualTarget(
        x=centre_x,
        y=centre_y,
        width=rect.width,
        height=rect.height,
        label=getattr(original, "label", "") or "",
        confidence=confidence,
        screen_id=getattr(original, "screen_id", None),
    )


def target_description(target, sentence="") -> str:
    """Describe a target well enough for the refinement model to find it.

    The label is the crispest name, but the sentence being taught usually adds
    the context that tells this target apart from similar things nearby, so both
    are used when they say different things.
    """
    label = (getattr(target, "label", "") or "").strip()
    sentence = (sentence or "").strip()

    if label and sentence and sentence.lower() != label.lower():
        return f"{label} ({sentence})"
    return label or sentence or "the highlighted element"
