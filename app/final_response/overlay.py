"""A transparent, click-through window that draws one mark at a time on the screen.

The final response gives each sentence a mark - point, circle, underline, arrow,
highlight or draw - at a box in screenshot pixels. The screenshot was taken in
physical pixels of one monitor; Qt places windows in logical pixels, so the box
is scaled by the ratio between the two before it is drawn. The window covers the
captured monitor, never takes focus and lets every click through.
"""

import logging
import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from app import config

logger = logging.getLogger(__name__)

ACCENT = QColor(255, 92, 58)
REVEAL_MS = 550
PULSE_MS = 33
# Space between a box and the mark drawn around it, in logical pixels.
MARGIN = 8.0
ARROW_LENGTH = 130.0


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
        self._mark = None
        self._progress = 0.0
        self._phase = 0.0

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

    def prepare(self, screenshot_size):
        """Cover the monitor the screenshot came from; False when its marks cannot be placed."""
        screen = target_screen(screenshot_size, self._monitor_index)
        if screen is None:
            logger.warning("[Overlay] marks are not drawn when every monitor is captured at once")
            self._scale = None
            return False
        geometry = screen.geometry()
        self.setGeometry(geometry)
        self._scale = scale_for(screenshot_size, (geometry.width(), geometry.height()))
        return True

    def show_mark(self, visual_action, box, point):
        """Draw one mark; nothing is drawn without a box or for the action 'none'."""
        if self._scale is None or not box or visual_action in (None, "none"):
            self._set_mark(None)
            return
        rect = map_box(box, self._scale)
        centre = map_point(point, self._scale) if point else rect.center()
        self._set_mark((visual_action, rect, centre))

    def clear(self):
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

    def _on_reveal(self, value):
        self._progress = float(value)
        self.update()

    def _on_pulse(self):
        self._phase = (self._phase + PULSE_MS / 1400.0) % 1.0
        self.update()

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event):
        if self._mark is None:
            return
        action, rect, centre = self._mark
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        draw = {
            "point": self._draw_point,
            "circle": self._draw_circle,
            "underline": self._draw_underline,
            "arrow": self._draw_arrow,
            "highlight": self._draw_highlight,
            "draw": self._draw_sketch,
        }.get(action, self._draw_highlight)
        draw(painter, rect, centre, self._progress)
        painter.end()

    def _glow_pen(self, width, alpha):
        color = QColor(ACCENT)
        color.setAlpha(alpha)
        pen = QPen(color, width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def _stroke(self, painter, path):
        """A soft glow under a solid accent line."""
        painter.setBrush(Qt.NoBrush)
        painter.setPen(self._glow_pen(12, 60))
        painter.drawPath(path)
        painter.setPen(self._glow_pen(4, 255))
        painter.drawPath(path)

    def _draw_point(self, painter, rect, centre, progress):
        # A ripple that keeps pulsing, then a pointer whose tip sits on the target.
        ripple = 10 + 26 * self._phase
        painter.setPen(self._glow_pen(3, int(200 * (1 - self._phase))))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, ripple, ripple)

        size = 30 * progress
        tip = centre
        pointer = QPolygonF(
            [
                tip,
                QPointF(tip.x() + 0.30 * size, tip.y() + 0.95 * size),
                QPointF(tip.x() + 0.48 * size, tip.y() + 0.58 * size),
                QPointF(tip.x() + 0.92 * size, tip.y() + 0.52 * size),
            ]
        )
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
        painter.setBrush(ACCENT)
        painter.drawPolygon(pointer)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(centre, 4, 4)

    def _draw_circle(self, painter, rect, centre, progress):
        ring = rect.adjusted(-MARGIN * 1.5, -MARGIN * 1.5, MARGIN * 1.5, MARGIN * 1.5)
        path = QPainterPath()
        path.arcMoveTo(ring, 90)
        path.arcTo(ring, 90, -360 * progress)
        self._stroke(painter, path)

    def _draw_underline(self, painter, rect, centre, progress):
        y = rect.bottom() + MARGIN / 2
        path = QPainterPath(QPointF(rect.left(), y))
        path.lineTo(rect.left() + rect.width() * progress, y)
        self._stroke(painter, path)

    def _draw_arrow(self, painter, rect, centre, progress):
        # The arrow comes from the side of the screen with more room, and stops at the box edge.
        width, height = self.width(), self.height()
        direction = QPointF(width / 2 - centre.x(), height / 2 - centre.y())
        length = math.hypot(direction.x(), direction.y()) or 1.0
        unit = QPointF(direction.x() / length, direction.y() / length)
        reach_x = (rect.width() / 2) / abs(unit.x()) if unit.x() else math.inf
        reach_y = (rect.height() / 2) / abs(unit.y()) if unit.y() else math.inf
        end = centre + unit * (min(reach_x, reach_y) + MARGIN)
        start = end + unit * ARROW_LENGTH
        head = start + (end - start) * progress

        path = QPainterPath(start)
        path.lineTo(head)
        self._stroke(painter, path)
        if progress > 0.6:
            back = QPointF(-unit.x(), -unit.y())
            side = QPointF(-back.y(), back.x())
            wing = 14
            tip = QPolygonF(
                [
                    head,
                    head - back * wing + side * wing * 0.6,
                    head - back * wing - side * wing * 0.6,
                ]
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(ACCENT)
            painter.drawPolygon(tip)

    def _draw_highlight(self, painter, rect, centre, progress):
        area = rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
        fill = QColor(ACCENT)
        fill.setAlpha(int(55 * progress))
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(area, 10, 10)
        path = QPainterPath()
        path.addRoundedRect(area, 10, 10)
        painter.setOpacity(progress)
        self._stroke(painter, path)
        painter.setOpacity(1.0)

    def _draw_sketch(self, painter, rect, centre, progress):
        # 'draw' marks where an explanatory sketch belongs: a dashed frame that keeps moving.
        area = rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
        pen = self._glow_pen(3, int(255 * progress))
        pen.setStyle(Qt.DashLine)
        pen.setDashOffset(-self._phase * 16)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(area, 12, 12)
