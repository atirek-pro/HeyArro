"""Screen-space visual primitives: what the overlay is asked to draw.

The model speaks in screenshot pixels and teaching actions; the overlay speaks
in screen coordinates and drawing primitives. Turning one into the other lives
here, so the overlay never has to know the response model and the teaching
service never has to know about drawing.

A primitive with a width and height is a region; one without is a bare point
the overlay renders with a sensible default marker.
"""

from dataclasses import dataclass

from app.llm.response import VisualActionType
from app.teaching.steps import normalize_label


@dataclass(frozen=True)
class VisualPrimitive:
    """One screen-space thing to draw, independent of any toolkit."""

    kind: str  # a VisualActionType value
    x: int  # screen-space centre
    y: int
    width: int | None = None
    height: int | None = None
    label: str | None = None

    @property
    def is_point(self):
        """True when this is the pointer action."""
        return self.kind == VisualActionType.POINT.value

    @property
    def has_bounds(self):
        """True when the region's size is known."""
        return bool(self.width and self.height)


def resolve_action(mapper, action):
    """Return the screen primitive for one action, or None when unusable.

    Nothing is invented: a target without integer coordinates, or one that is
    not there at all, produces no primitive (and the sentence is still spoken).
    """
    target = getattr(action, "target", None)
    if target is None:
        return None

    x = getattr(target, "x", None)
    y = getattr(target, "y", None)
    if not isinstance(x, int) or not isinstance(y, int):
        return None

    width = getattr(target, "width", None)
    height = getattr(target, "height", None)
    if width and height:
        width, height = mapper.scale_size(width, height)
    else:
        width = height = None

    centre_x, centre_y = mapper.to_screen(x, y)
    action_type = getattr(action, "type", None)

    return VisualPrimitive(
        kind=getattr(action_type, "value", None) or str(action_type),
        x=centre_x,
        y=centre_y,
        width=width,
        height=height,
        label=normalize_label(getattr(target, "label", None)),
    )


def resolve_actions(mapper, actions):
    """Resolve a sequence of actions to screen primitives, skipping unusable ones."""
    primitives = []
    for action in actions or ():
        primitive = resolve_action(mapper, action)
        if primitive is not None:
            primitives.append(primitive)
    return primitives
