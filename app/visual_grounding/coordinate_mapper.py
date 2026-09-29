"""Coordinate frames used by visual grounding.

Four frames are involved, and confusing them is the classic source of
"the box is slightly off" bugs, so each is named explicitly here:

1. **Screenshot coordinates** - pixels of the captured PNG. This is the frame
   the vision model is told about and the frame every ``VisualTarget`` uses.
2. **Crop coordinates** - pixels of the high-resolution crop cut out of the
   screenshot. The refinement model answers in this frame.
3. **Display coordinates** - Qt logical (device-independent) screen pixels. The
   overlay window lives here, and this is where DPI scaling is accounted for.
4. **Overlay coordinates** - the overlay's own local pixels, which are display
   coordinates minus the overlay window's origin. The overlay does that
   subtraction itself.

This module owns the crop frame: ``crop_to_screenshot`` and the inverse. The
screenshot -> display step is *not* duplicated here - it already lives in
``app.teaching.coordinates.CoordinateMapper`` and the overlay consumes its
output unchanged, so display and overlay frames stay exactly as they are today.
"""

from app.visual_grounding.models import CropResult, Rect


class GroundingCoordinateMapper:
    """Converts between crop pixels and screenshot pixels for one crop."""

    def __init__(self, origin_x, origin_y, width, height):
        self._origin_x = int(origin_x)
        self._origin_y = int(origin_y)
        self._width = int(width)
        self._height = int(height)

    @classmethod
    def for_crop(cls, crop: CropResult):
        """Build the mapper for a ``CropResult`` produced by the cropper."""
        return cls(crop.origin_x, crop.origin_y, crop.width, crop.height)

    @classmethod
    def from_origin(cls, origin_x, origin_y, width, height):
        return cls(origin_x, origin_y, width, height)

    @property
    def origin(self):
        return (self._origin_x, self._origin_y)

    @property
    def size(self):
        return (self._width, self._height)

    def crop_to_screenshot_point(self, x, y):
        """Crop pixel -> screenshot pixel."""
        return (self._origin_x + int(x), self._origin_y + int(y))

    def crop_to_screenshot_rect(self, rect: Rect) -> Rect:
        """Crop rectangle (edges) -> screenshot rectangle (edges)."""
        return rect.shifted(self._origin_x, self._origin_y)

    def screenshot_to_crop_rect(self, rect: Rect) -> Rect:
        """Screenshot rectangle (edges) -> crop rectangle (edges)."""
        return rect.shifted(-self._origin_x, -self._origin_y)

    def crop_bounds(self) -> Rect:
        """The crop's own rectangle, expressed in crop coordinates."""
        return Rect(0, 0, self._width, self._height)
