"""Finding calm, empty space on the screenshot for Hey Arro's handwriting board.

The screenshot is turned once per answer into a coarse "busyness" grid: how much
edge detail (text, icons, lines) each small cell holds. A board is then placed
where it covers the least detail - beside its element when possible, otherwise
in the calmest area of the screen - and never on top of the element itself or
the status pill. Everything is in screenshot pixels.
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Side of one busyness cell, in screenshot pixels.
CELL = 8
# Gap between the element and the board beside it.
GAP = 28
# Space kept free around the element and from the screen edges.
CLEARANCE = 10
EDGE = 12
# Step of the search for a calm spot anywhere on the screen.
SCAN_STEP = 32


class CalmMap:
    """How busy each part of a screenshot is; 0 is an empty area."""

    def __init__(self, image):
        gray = image.convert("L")
        self.width, self.height = gray.size
        pixels = np.asarray(gray, dtype=np.float32)
        edges = np.zeros_like(pixels)
        edges[:, 1:] += np.abs(np.diff(pixels, axis=1))
        edges[1:, :] += np.abs(np.diff(pixels, axis=0))
        rows, cols = self.height // CELL, self.width // CELL
        cells = edges[: rows * CELL, : cols * CELL].reshape(rows, CELL, cols, CELL).mean(axis=(1, 3))
        # Summed-area table: the busyness of any rectangle in constant time.
        self._sums = np.zeros((rows + 1, cols + 1), dtype=np.float64)
        self._sums[1:, 1:] = cells.cumsum(axis=0).cumsum(axis=1)
        self._rows, self._cols = rows, cols

    @classmethod
    def from_file(cls, path):
        """The map of the screenshot at ``path``, or None when it cannot be read."""
        try:
            with Image.open(path) as image:
                return cls(image)
        except (OSError, ValueError) as exc:
            logger.warning("[Writing] could not read the screenshot to place the writing: %s", exc)
            return None

    def busyness(self, rect):
        """Average edge detail inside ``rect`` (left, top, width, height)."""
        left, top, width, height = rect
        c0 = max(0, min(self._cols, int(left // CELL)))
        r0 = max(0, min(self._rows, int(top // CELL)))
        c1 = max(c0 + 1, min(self._cols, int((left + width) // CELL) + 1))
        r1 = max(r0 + 1, min(self._rows, int((top + height) // CELL) + 1))
        if c0 >= self._cols or r0 >= self._rows:
            return float("inf")
        total = self._sums[r1, c1] - self._sums[r0, c1] - self._sums[r1, c0] + self._sums[r0, c0]
        return float(total / ((r1 - r0) * (c1 - c0)))


def _overlaps(a, b, pad=0.0):
    return not (
        a[0] + a[2] + pad <= b[0]
        or b[0] + b[2] + pad <= a[0]
        or a[1] + a[3] + pad <= b[1]
        or b[1] + b[3] + pad <= a[1]
    )


def _fits(rect, bounds, avoid):
    left, top, width, height = rect
    if left < EDGE or top < EDGE or left + width > bounds[0] - EDGE or top + height > bounds[1] - EDGE:
        return False
    return not any(_overlaps(rect, other, CLEARANCE) for other in avoid)


def _beside(target, size):
    """Board positions next to ``target``: right, left, below, above, each in three alignments."""
    left, top, width, height = target
    w, h = size
    middle_y = top + height / 2 - h / 2
    middle_x = left + width / 2 - w / 2
    return [
        (left + width + GAP, middle_y), (left + width + GAP, top), (left + width + GAP, top + height - h),
        (left - GAP - w, middle_y), (left - GAP - w, top), (left - GAP - w, top + height - h),
        (middle_x, top + height + GAP), (left, top + height + GAP), (left + width - w, top + height + GAP),
        (middle_x, top - GAP - h), (left, top - GAP - h), (left + width - w, top - GAP - h),
    ]


def find_spot(size, bounds, target=None, avoid=(), calm=None):
    """Where to put a board of ``size`` (w, h) on a ``bounds`` (w, h) screen.

    Prefers a calm position beside ``target`` (a box dict or None); falls back to the
    calmest place anywhere, nearest the target. Returns (left, top, width, height).
    """
    w, h = size
    blocked = list(avoid)
    target_rect = None
    if target:
        target_rect = (target["left"], target["top"], target["width"], target["height"])
        blocked.append(target_rect)

    def score(rect):
        return calm.busyness(rect) if calm is not None else 0.0

    if target_rect is not None:
        beside = [(x, y, w, h) for x, y in _beside(target_rect, size)]
        fitting = [rect for rect in beside if _fits(rect, bounds, blocked)]
        if fitting:
            best = min(fitting, key=score)
            # Beside the element is best unless that spot is much busier than the calmest one.
            if calm is None or score(best) <= 18.0:
                return best

    centre = (
        (target_rect[0] + target_rect[2] / 2, target_rect[1] + target_rect[3] / 2)
        if target_rect
        else (bounds[0] * 0.6, bounds[1] * 0.4)
    )
    candidates = []
    for top in range(EDGE, int(bounds[1] - h - EDGE) + 1, SCAN_STEP):
        for left in range(EDGE, int(bounds[0] - w - EDGE) + 1, SCAN_STEP):
            rect = (left, top, w, h)
            if _fits(rect, bounds, blocked):
                distance = abs(left + w / 2 - centre[0]) + abs(top + h / 2 - centre[1])
                candidates.append((score(rect) + distance / 400.0, rect))
    if candidates:
        return min(candidates)[1]
    # Nothing fits cleanly (a tiny screen or a huge board): keep it on screen at least.
    left = max(EDGE, min(bounds[0] - w - EDGE, centre[0] - w / 2))
    top = max(EDGE, min(bounds[1] - h - EDGE, centre[1] - h / 2))
    return (left, top, w, h)
