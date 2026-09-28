"""Screenshot-pixel to screen-coordinate mapping for the teaching overlay.

Gemini returns a target as a pixel in the screenshot it was given. The overlay
draws in Qt's logical (device-independent) screen coordinates, which differ from
screenshot pixels whenever the display is scaled (for example 150%) or when the
capture resolution is not the monitor resolution. Converting between the two
lives here, in one testable place, so no drawing code has to guess.

The mapper is intentionally small: it maps one captured screenshot onto one
screen. Multi-monitor support later means choosing the right ``ScreenGeometry``
and screenshot pair, not changing this maths.
"""

from dataclasses import dataclass


class CoordinateMappingError(RuntimeError):
    """Raised when a screenshot cannot be mapped onto a screen."""


@dataclass(frozen=True)
class ScreenGeometry:
    """A screen's rectangle in Qt logical (device-independent) coordinates."""

    left: int
    top: int
    width: int
    height: int
    name: str = "primary"

    @property
    def size(self):
        return (self.width, self.height)


class CoordinateMapper:
    """Maps screenshot pixels onto the logical rectangle of one screen.

    The screenshot is assumed to cover ``screen`` completely, which holds for
    the single-screen configuration this phase supports.
    """

    def __init__(self, screen, screenshot_width, screenshot_height):
        if screenshot_width <= 0 or screenshot_height <= 0:
            raise CoordinateMappingError(
                f"Screenshot has no usable size ({screenshot_width}x{screenshot_height})"
            )

        self._screen = screen
        self._screenshot_width = screenshot_width
        self._screenshot_height = screenshot_height
        # Screenshot pixel -> screen logical pixel. Equal to 1.0 for an
        # unscaled display captured at its native resolution.
        self._scale_x = screen.width / screenshot_width
        self._scale_y = screen.height / screenshot_height

    @classmethod
    def from_capture(cls, capture, screen):
        """Build a mapper for a ``ScreenCaptureResult`` and a screen."""
        return cls(screen, capture.width, capture.height)

    @property
    def screen(self):
        return self._screen

    @property
    def scale(self):
        """The ``(x, y)`` screenshot-pixel to screen-pixel scale factors."""
        return (self._scale_x, self._scale_y)

    def to_screen(self, x, y):
        """Return the global logical screen point for a screenshot pixel."""
        return (
            int(round(self._screen.left + x * self._scale_x)),
            int(round(self._screen.top + y * self._scale_y)),
        )

    def scale_size(self, width, height):
        """Return the screen-space ``(width, height)`` for a screenshot size."""
        return (
            max(1, int(round(width * self._scale_x))),
            max(1, int(round(height * self._scale_y))),
        )

    def to_screen_rect(self, x, y, width, height):
        """Return the screen-space ``(left, top, width, height)`` for a region.

        ``(x, y)`` is the region's centre in screenshot pixels, matching
        VisualTarget; the returned rectangle is in screen coordinates, which is
        what the overlay draws with.
        """
        centre_x, centre_y = self.to_screen(x, y)
        screen_width, screen_height = self.scale_size(width, height)
        return (
            centre_x - screen_width // 2,
            centre_y - screen_height // 2,
            screen_width,
            screen_height,
        )
