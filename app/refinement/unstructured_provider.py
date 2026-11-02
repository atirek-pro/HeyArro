"""Screenshot refinement implemented with the ``unstructured`` library.

``unstructured`` (and Pillow) are imported lazily so a missing or broken install
becomes a handled refinement error instead of crashing the application at
startup.

Every step is logged with a ``[Refinement]`` prefix so a bad run can be traced
from the received screenshot down to each cropped segment.
"""

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from app import config
from app.refinement.manifest import write_segment_manifest
from app.refinement.provider import (
    RefinementResult,
    ScreenRefinementError,
    ScreenRefinementProvider,
    ScreenSegment,
)

logger = logging.getLogger(__name__)

SCREENSHOT_NAME = "screenshot.png"
SEGMENTS_DIR = "segments"
ANNOTATED_NAME = "screenshot_bounding_boxes.png"


def _import_partition_image():
    try:
        from unstructured.partition.image import partition_image
    except Exception as exc:
        raise ScreenRefinementError(f"unstructured is not available: {exc}") from exc
    return partition_image


def _import_image():
    try:
        from PIL import Image
    except Exception as exc:
        raise ScreenRefinementError(f"Pillow is not available: {exc}") from exc
    return Image


def _bounds(element, image_size):
    """Return (left, top, right, bottom) in image pixels, or None without coordinates.

    unstructured reports points in its own coordinate system, which is not always
    the pixel size of the image, so the points are scaled to the image.
    """
    coordinates = getattr(element.metadata, "coordinates", None)
    points = getattr(coordinates, "points", None) if coordinates else None
    if not points:
        return None

    scale_x = scale_y = 1.0
    system_width = getattr(coordinates, "system", None) and coordinates.system.width
    system_height = getattr(coordinates, "system", None) and coordinates.system.height
    if system_width and system_height:
        scale_x = image_size[0] / system_width
        scale_y = image_size[1] / system_height

    xs = [point[0] * scale_x for point in points]
    ys = [point[1] * scale_y for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _kind(element):
    return getattr(element, "category", None) or type(element).__name__


class UnstructuredRefinementProvider(ScreenRefinementProvider):
    """Segments a screenshot with ``unstructured.partition.image``."""

    def __init__(self, strategy=None, languages=None, min_segment_pixels=None):
        self._strategy = strategy or config.REFINEMENT_STRATEGY
        self._languages = tuple(languages) if languages is not None else config.REFINEMENT_LANGUAGES
        self._min_pixels = (
            config.REFINEMENT_MIN_SEGMENT_PIXELS if min_segment_pixels is None else min_segment_pixels
        )

    @property
    def strategy(self):
        return self._strategy

    def refine(self, capture, output_dir) -> RefinementResult:
        started = time.perf_counter()
        logger.info(
            "[Refinement] received screenshot: %sx%s, %s bytes, monitor %s",
            capture.width,
            capture.height,
            capture.size_bytes,
            capture.monitor_index,
        )

        folder = self._create_folder(output_dir)
        screenshot_path = self._save_screenshot(capture, folder)
        elements = self._partition(screenshot_path)
        segments = self._crop_segments(screenshot_path, folder, elements)

        result = RefinementResult(
            folder=str(folder),
            screenshot_path=str(screenshot_path),
            segments=segments,
            strategy=self._strategy,
        )
        self._write_manifest(result)

        logger.info(
            "[Refinement] done in %.2fs: %s segment(s) from %s element(s) in %s",
            time.perf_counter() - started,
            len(segments),
            len(elements),
            folder,
        )
        return result

    # --- steps --------------------------------------------------------------

    def _create_folder(self, output_dir):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = Path(output_dir) / f"screenshot-{stamp}-{uuid.uuid4().hex[:6]}"
        try:
            (folder / SEGMENTS_DIR).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ScreenRefinementError(f"Could not create {folder}: {exc}") from exc
        logger.info("[Refinement] created folder %s", folder)
        return folder

    def _save_screenshot(self, capture, folder):
        path = folder / SCREENSHOT_NAME
        try:
            path.write_bytes(capture.image)
        except OSError as exc:
            raise ScreenRefinementError(f"Could not save the screenshot: {exc}") from exc
        logger.info("[Refinement] saved screenshot to %s", path)
        return path

    def _partition(self, screenshot_path):
        partition_image = _import_partition_image()
        logger.info(
            "[Refinement] partitioning with unstructured (strategy=%s, languages=%s)",
            self._strategy,
            list(self._languages) or "default",
        )
        options = {"filename": str(screenshot_path), "strategy": self._strategy}
        if self._languages:
            options["languages"] = list(self._languages)
        try:
            elements = partition_image(**options)
        except Exception as exc:
            raise ScreenRefinementError(f"unstructured could not partition the screenshot: {exc}") from exc

        logger.info("[Refinement] unstructured returned %s element(s)", len(elements))
        return elements

    def _crop_segments(self, screenshot_path, folder, elements):
        Image = _import_image()
        segments = []

        try:
            image = Image.open(screenshot_path)
            image.load()
        except Exception as exc:
            raise ScreenRefinementError(f"Could not open the saved screenshot: {exc}") from exc

        with image:
            for position, element in enumerate(elements, start=1):
                kind = _kind(element)
                text = (getattr(element, "text", "") or "").strip()
                bounds = _bounds(element, image.size)

                if bounds is None:
                    logger.warning("[Refinement] element %s (%s) has no coordinates; skipped", position, kind)
                    continue

                left = max(0, int(bounds[0]))
                top = max(0, int(bounds[1]))
                right = min(image.width, int(bounds[2]))
                bottom = min(image.height, int(bounds[3]))
                width, height = right - left, bottom - top

                if width * height < self._min_pixels:
                    logger.info(
                        "[Refinement] element %s (%s) skipped: %sx%s is below %s pixels",
                        position,
                        kind,
                        width,
                        height,
                        self._min_pixels,
                    )
                    continue

                index = len(segments) + 1
                slug = "".join(c if c.isalnum() else "-" for c in kind.lower())
                path = folder / SEGMENTS_DIR / f"segment-{index:03d}-{slug}.png"
                try:
                    image.crop((left, top, right, bottom)).save(path)
                except Exception as exc:
                    logger.error("[Refinement] could not write segment %s to %s: %s", index, path, exc)
                    path = None

                segments.append(
                    ScreenSegment(
                        index=index,
                        kind=kind,
                        text=text,
                        left=left,
                        top=top,
                        width=width,
                        height=height,
                        path=str(path) if path else None,
                    )
                )
                logger.info(
                    "[Refinement] segment %s: kind=%s box=(%s,%s,%s,%s) text=%r file=%s",
                    index,
                    kind,
                    left,
                    top,
                    right,
                    bottom,
                    text[:80],
                    path,
                )

            self._annotate(image, folder, segments)

        return segments

    def _write_manifest(self, result):
        """Describe every segment in ss_segment_manifest.json; a failure is not fatal."""
        try:
            path = write_segment_manifest(result)
        except OSError as exc:
            logger.error("[Refinement] could not write the segment manifest: %s", exc)
            return
        logger.info("[Refinement] saved segment manifest to %s", path)

    def _annotate(self, image, folder, segments):
        """Draw every segment's box on a copy of the screenshot, for visual debugging."""
        from PIL import ImageDraw

        try:
            canvas = image.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for segment in segments:
                draw.rectangle(segment.box, outline="red", width=3)
                draw.text(
                    (segment.left, max(0, segment.top - 14)),
                    f"{segment.index}: {segment.kind}",
                    fill="red",
                    stroke_width=1,
                    stroke_fill="white",
                )
            path = folder / ANNOTATED_NAME
            canvas.save(path)
        except Exception as exc:
            logger.error("[Refinement] could not write the annotated screenshot: %s", exc)
            return
        logger.info("[Refinement] saved annotated screenshot to %s", path)
