"""Helpers for reading the visual actions a teaching step asks for.

A step carries a list of actions rather than a single target: one sentence can
need several things shown at once (both operands of a multiplication, a row and
a column, two related labels). Nothing here invents geometry - a step with no
actions simply has nothing to show.
"""

from app.llm.response import VisualActionType


def normalize_label(label):
    """Return a trimmed label, or None when there is nothing worth drawing."""
    text = (label or "").strip()
    return text or None


def step_actions(step):
    """Return the visual actions of ``step`` as a list (never None)."""
    return list(getattr(step, "visual_actions", None) or ())


def iter_actions(steps):
    """Yield every visual action across ``steps``."""
    for step in steps or ():
        yield from step_actions(step)


def has_bounds(target):
    """True when a target carries a usable width and height."""
    return bool(
        target is not None
        and getattr(target, "width", None)
        and getattr(target, "height", None)
    )


def iter_point_actions(steps):
    """Yield ``(step, action)`` for every ``point`` action across ``steps``."""
    for step in steps or ():
        for action in step_actions(step):
            if getattr(action, "type", None) == VisualActionType.POINT:
                yield step, action
