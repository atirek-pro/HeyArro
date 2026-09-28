"""Provider-neutral visual teaching contract.

The coordinator decides *when* a teaching step is shown (and speaks it); the
service decides *how* to put it on screen. The concrete implementation drives a
``TeachingOverlay``, mapping the step's screenshot coordinates onto the screen
first, so neither the coordinator nor this module contains any drawing code.

A step is shown by applying **all** of its visual actions together - a sentence
can need several things highlighted at once - and a step with no actions simply
clears the overlay. Nothing is ever invented: unusable geometry draws nothing.
"""

from abc import ABC, abstractmethod

from app.teaching.coordinates import CoordinateMapper
from app.teaching.primitives import resolve_actions
from app.teaching.steps import step_actions


class TeachingError(RuntimeError):
    """Raised when a teaching step cannot be shown."""


class VisualTeachingService(ABC):
    """Puts teaching steps on screen."""

    @abstractmethod
    def prepare(self, capture):
        """Set the screenshot the next steps' coordinates refer to.

        ``capture`` is a ScreenCaptureResult, or None when no screenshot was
        taken; passing None means "nothing can be shown".
        """

    @abstractmethod
    def show_step(self, step):
        """Show everything the step asks for, in one visual state."""

    @abstractmethod
    def show_actions(self, actions):
        """Show a list of visual actions directly, without a step."""

    @abstractmethod
    def hide(self):
        """Remove everything the service is showing."""


class NullVisualTeachingService(VisualTeachingService):
    """A service that renders nothing, for headless runs or when teaching is off."""

    def prepare(self, capture):
        return None

    def show_step(self, step):
        return None

    def show_actions(self, actions):
        return None

    def hide(self):
        return None


class OverlayVisualTeachingService(VisualTeachingService):
    """Drives a ``TeachingOverlay`` view, mapping screenshot coordinates first."""

    def __init__(self, overlay, screen):
        self._overlay = overlay
        self._screen = screen
        self._mapper = None

    @property
    def mapper(self):
        """The mapper for the current interaction, or None when there is none."""
        return self._mapper

    def prepare(self, capture):
        if capture is None:
            self._mapper = None
            return
        self._mapper = CoordinateMapper.from_capture(capture, self._screen)

    def show_step(self, step):
        self.show_actions(step_actions(step))

    def show_actions(self, actions):
        if self._mapper is None:
            # Without the screenshot there is no coordinate space to draw in.
            self._overlay.clear()
            return

        # An empty list is meaningful: it fades any previous visual out rather
        # than inventing something to show.
        self._overlay.show_actions(resolve_actions(self._mapper, actions))

    def hide(self):
        self._overlay.clear()
