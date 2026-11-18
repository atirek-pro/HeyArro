"""A transparent, click-through window that draws one mark at a time on the screen.

The final response gives each sentence a mark (point, circle, rectangle, marker,
spotlight, callout, connector, magnifier... see ``marks.py``) at a box in
screenshot pixels. The screenshot was taken in physical pixels of one monitor; Qt
places windows in logical pixels, so every box is scaled by the ratio between the
two before it is drawn. The window covers the captured monitor, never takes focus
and lets every click through. Painting itself lives in ``marks.py``.
"""

import logging
from pathlib import Path

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QWidget

from app import config
from app.final_response.marks import ACCENT, MarkContext, paint_mark  # noqa: F401  (ACCENT re-exported)
from app.final_response.placement import CalmMap, find_spot
from app.final_response.writing import build_writing, paint_writing, warm_up

logger = logging.getLogger(__name__)

REVEAL_MS = 550
PULSE_MS = 33
# Screenshot pixels kept around a target in the magnifier's zoomed view.
ZOOM_PADDING = 12
# The status pill's zone at the top centre (logical pixels), kept free of writing.
INDICATOR_HALF_WIDTH = 170
INDICATOR_HEIGHT = 120


def scale_for(screenshot_size, screen_size):
    """The factors that turn screenshot pixels into the screen's logical pixels."""
    shot_width, shot_height = screenshot_size
    screen_width, screen_height = screen_size
    return screen_width / shot_width, screen_height / shot_height


def map_box(box, scale):
    """A screenshot pixel box as a logical ``QRectF`` on the overlay."""
    sx, sy = scale
    return QRectF(box["left"] * sx, box["top"] * sy, box["width"] * sx, box["height"] * sy)


def map_point(point, scale):
    sx, sy = scale
    return QPointF(point["x"] * sx, point["y"] * sy)


def target_screen(screenshot_size, monitor_index):
    """The Qt screen the screenshot was taken from, or None when it cannot be told.

    Monitor 1 is the primary screen. Other monitors are matched by their size in
    physical pixels. Monitor 0 (every monitor combined) is not supported.
    """
    if monitor_index == 0:
        return None
    primary = QGuiApplication.primaryScreen()
    if monitor_index == 1:
        return primary
    for screen in QGuiApplication.screens():
        ratio = screen.devicePixelRatio()
        size = screen.geometry().size()
        if (round(size.width() * ratio), round(size.height() * ratio)) == tuple(screenshot_size):
            return screen
    return primary


class ScreenOverlay(QWidget):
    """Draws the mark of the sentence being spoken over the captured monitor."""

    def __init__(self, monitor_index=None):
        super().__init__(
            None,
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            | Qt.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self._monitor_index = config.SCREENSHOT_MONITOR if monitor_index is None else monitor_index
        self._scale = None
        self._screenshot_path = None
        self._screenshot = None
        self._mark = None
        self._progress = 0.0
        self._phase = 0.0
        # The handwriting layer: (layout, board rect, element rect or None), drawn above the mark.
        self._writing = None
        self._write_progress = 0.0
        self._shot_size = None
        self._calm = None
        self._calm_ready = False

        self._write = QVariantAnimation(self)
        self._write.setStartValue(0.0)
        self._write.setEndValue(1.0)
        self._write.valueChanged.connect(self._on_write)
        warm_up()

        self._reveal = QVariantAnimation(self)
        self._reveal.setDuration(REVEAL_MS)
        self._reveal.setStartValue(0.0)
        self._reveal.setEndValue(1.0)
        self._reveal.setEasingCurve(QEasingCurve.OutCubic)
        self._reveal.valueChanged.connect(self._on_reveal)

        self._pulse = QTimer(self)
        self._pulse.setInterval(PULSE_MS)
        self._pulse.timeout.connect(self._on_pulse)

    # --- public API ---------------------------------------------------------

    def prepare(self, screenshot_size, screenshot_path=None):
        """Cover the monitor the screenshot came from; False when its marks cannot be placed.

        ``screenshot_path`` lets the magnifier show a zoomed view; it is read only when needed.
        """
        self._screenshot_path = screenshot_path
        self._screenshot = None
        self._shot_size = tuple(screenshot_size)
        self._calm = None
        self._calm_ready = False
        self._clear_writing()
        screen = target_screen(screenshot_size, self._monitor_index)
        if screen is None:
            logger.warning("[Overlay] marks are not drawn when every monitor is captured at once")
            self._scale = None
            return False
        geometry = screen.geometry()
        self.setGeometry(geometry)
        self._scale = scale_for(screenshot_size, (geometry.width(), geometry.height()))
        return True

    def show_mark(self, visual_action, box, point, details=None):
        """Draw one mark; nothing is drawn without a box or for the action 'none'."""
        if self._scale is None or not box or visual_action in (None, "none"):
            self._set_mark(None)
            return
        rect = map_box(box, self._scale)
        centre = map_point(point, self._scale) if point else rect.center()
        self._set_mark((visual_action, rect, centre, self._context(visual_action, box, details or {})))

    def _context(self, visual_action, box, details):
        """The mark's extra data, with its second target scaled like the first."""
        to_box = details.get("to_box")
        to_point = details.get("to_point")
        to_rect = map_box(to_box, self._scale) if to_box else None
        return MarkContext(
            canvas=QRectF(0, 0, self.width(), self.height()),
            number=details.get("number"),
            label=details.get("label") or "",
            keys=list(details.get("keys") or []),
            scroll=details.get("scroll") or "",
            to_rect=to_rect,
            to_centre=map_point(to_point, self._scale) if to_point else (to_rect.center() if to_rect else None),
            zoom=self._zoom(box) if visual_action == "magnifier" else None,
        )

    def _zoom(self, box):
        """The target's area cut from the screenshot, sharp at its own resolution; None if unreadable."""
        if self._screenshot is None and self._screenshot_path:
            image = QImage(str(Path(self._screenshot_path)))
            if image.isNull():
                logger.warning("[Overlay] could not read the screenshot for the magnifier")
                self._screenshot_path = None
                return None
            self._screenshot = image
        if self._screenshot is None:
            return None
        area = QRect(
            int(box["left"]) - ZOOM_PADDING,
            int(box["top"]) - ZOOM_PADDING,
            int(box["width"]) + 2 * ZOOM_PADDING,
            int(box["height"]) + 2 * ZOOM_PADDING,
        ).intersected(self._screenshot.rect())
        return self._screenshot.copy(area) if not area.isEmpty() else None

    def show_writing(self, lines, kind="note", box=None):
        """Write ``lines`` on a glass board in calm space beside ``box`` (or anywhere calm).

        The board replaces any earlier one and stays until the next writing or ``clear``.
        """
        if self._scale is None or not lines:
            return
        layout = build_writing(lines, kind)
        if not layout.lines:
            return
        sx, sy = self._scale
        bounds = self._shot_size or (self.width() / sx, self.height() / sy)
        size = (layout.width / sx, layout.height / sy)
        spot = find_spot(size, bounds, target=box, avoid=[self._indicator_zone()], calm=self._calm_map())
        board = map_box({"left": spot[0], "top": spot[1], "width": spot[2], "height": spot[3]}, self._scale)
        link = map_box(box, self._scale) if box else None
        self._writing = (layout, board, link)
        self._write.stop()
        self._write.setDuration(layout.write_ms())
        self._write_progress = 0.0
        if not self.isVisible():
            self.show()
            self.raise_()
        self._write.start()
        logger.info("[Overlay] writing %s line(s) (%s) at %s", len(layout.lines), layout.kind, [round(v) for v in spot])

    def _indicator_zone(self):
        """The status pill's area at the top centre, in screenshot pixels, kept free of writing."""
        sx, sy = self._scale
        width = self._shot_size[0] if self._shot_size else self.width() / sx
        return (width / 2 - INDICATOR_HALF_WIDTH / sx, 0, 2 * INDICATOR_HALF_WIDTH / sx, INDICATOR_HEIGHT / sy)

    def _calm_map(self):
        if not self._calm_ready:
            self._calm_ready = True
            self._calm = CalmMap.from_file(self._screenshot_path) if self._screenshot_path else None
        return self._calm

    def _clear_writing(self):
        self._writing = None
        self._write.stop()
        self._write_progress = 0.0

    def clear(self):
        self._clear_writing()
        self._set_mark(None)
        self.hide()

    # --- animation ----------------------------------------------------------

    def _set_mark(self, mark):
        self._mark = mark
        self._reveal.stop()
        if mark is None:
            self._pulse.stop()
            self.update()
            return
        self._progress = 0.0
        if not self.isVisible():
            self.show()
            self.raise_()
        self._reveal.start()
        self._pulse.start()

    def _on_write(self, value):
        self._write_progress = float(value)
        self.update()

    def _on_reveal(self, value):
        self._progress = float(value)
        self.update()

    def _on_pulse(self):
        self._phase = (self._phase + PULSE_MS / 1400.0) % 1.0
        self.update()

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event):
        if self._mark is None and self._writing is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        try:
            if self._mark is not None:
                action, rect, centre, context = self._mark
                context.canvas = QRectF(0, 0, self.width(), self.height())
                paint_mark(painter, action, rect, centre, self._progress, self._phase, context)
            if self._writing is not None:
                layout, board, link = self._writing
                paint_writing(painter, layout, board, link, self._write_progress)
        finally:
            painter.end()
