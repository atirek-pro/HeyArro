"""Cutting a high-resolution context region out of a screenshot.

The first vision pass is only approximate, so the crop must include context
around its guess rather than trusting it: the approximate region is grown by a
configurable padding and clipped to the screenshot. The crop remembers where it
came from, which is what makes it possible to put the refined target back into
screenshot coordinates.
"""

import uuid
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from app.visual_grounding.base import GroundingError
from app.visual_grounding.models import CropResult, Rect

# A crop smaller than this cannot say anything useful about a target.
MIN_CROP_SIZE = 16

# Debug colours: the first pass's guess vs what the refinement found.
APPROXIMATE_COLOUR = (255, 170, 0)
REFINED_COLOUR = (0, 220, 120)


def crop_region(approximate: Rect, padding: int, screenshot_size) -> Rect | None:
    """Return the region to crop around ``approximate``, or None when unusable.

    The region is the approximate target grown by ``padding`` on every side and
    clipped to the screenshot, so a target near an edge still produces a valid
    crop and a target outside the screen produces nothing at all.
    """
    width, height = screenshot_size
    if width <= 0 or height <= 0:
        return None

    region = approximate.expand(max(0, int(padding))).clamp(width, height)
    if region.width < MIN_CROP_SIZE or region.height < MIN_CROP_SIZE:
        return None
    return region


def decode_png(data) -> QImage:
    """Read image bytes into a QImage."""
    image = QImage()
    if not data or not image.loadFromData(data):
        raise GroundingError("Could not read the screenshot as an image")
    return image


def encode_png(image) -> bytes:
    """Encode a QImage as PNG bytes (keeping the buffer alive while copying)."""
    data = QByteArray()
    buffer = QBuffer(data)
    try:
        buffer.open(QIODevice.WriteOnly)
        if not image.save(buffer, "PNG"):
            raise GroundingError("Could not encode the image as PNG")
    finally:
        buffer.close()
    return bytes(data)


class ScreenshotCropper:
    """Extracts a region of a screenshot as a standalone PNG.

    Qt's image support does the decoding, so no extra dependency is needed.
    """

    def crop(self, screenshot: bytes, region: Rect) -> CropResult:
        if region.is_empty:
            raise GroundingError(f"Refusing to crop an empty region ({region.describe()})")

        image = decode_png(screenshot)
        if (
            region.left < 0
            or region.top < 0
            or region.right > image.width()
            or region.bottom > image.height()
        ):
            raise GroundingError(
                f"Crop {region.describe()} is outside the "
                f"{image.width()}x{image.height()} screenshot"
            )

        cropped = image.copy(QRect(region.left, region.top, region.width, region.height))
        if cropped.isNull():
            raise GroundingError("Cropping the screenshot produced an empty image")

        return CropResult(
            image=encode_png(cropped),
            origin_x=region.left,
            origin_y=region.top,
            width=region.width,
            height=region.height,
        )


def annotate(image_bytes, boxes) -> bytes:
    """Draw labelled rectangles on an image and return it as PNG.

    ``boxes`` is a sequence of ``(Rect, (r, g, b), label)``. Used by the debug
    mode to compare the first pass's guess with the refinement's answer.
    """
    image = decode_png(image_bytes).convertToFormat(QImage.Format_RGB32)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    metrics = painter.fontMetrics()

    for rect, colour, label in boxes:
        painter.setPen(QPen(QColor(*colour), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRect(rect.left, rect.top, rect.width, rect.height))

        if not label:
            continue

        text_width = metrics.horizontalAdvance(label) + 10
        text_height = metrics.height() + 6
        x = min(max(rect.left, 0), max(0, image.width() - text_width))
        y = rect.top - text_height
        if y < 0:
            y = rect.bottom
        y = min(max(y, 0), max(0, image.height() - text_height))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(*colour))
        painter.drawRect(QRect(x, y, text_width, text_height))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            QRect(x + 5, y, text_width - 5, text_height),
            int(Qt.AlignVCenter | Qt.AlignLeft),
            label,
        )

    painter.end()
    return encode_png(image)


def save_debug_image(directory, name, image_bytes) -> Path:
    """Write a debug image to ``directory`` and return its path."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stamp}-{uuid.uuid4().hex[:6]}-{name}.png"
    path.write_bytes(image_bytes)
    return path
