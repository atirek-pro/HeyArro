"""Floating visual companion that sits just next to the mouse cursor.

The widget is a passive view: it renders whichever application state it is told
to render. It never decides state itself - ApplicationCoordinator (via
app.state.ApplicationStateMachine) is the single source of truth.
"""

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QGuiApplication,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget

from app.state import ApplicationState

# Cursor-arrow silhouette (tip first, at the top-left) with the classic leg.
# Coordinates are unit-less and scaled to the widget when painted.
_ARROW_POINTS = (
    (0.0, 0.0),
    (0.0, 16.0),
    (4.0, 12.5),
    (7.0, 21.0),
    (10.5, 19.5),
    (7.5, 11.0),
    (12.5, 11.0),
)
_ARROW_WIDTH = 12.5
_ARROW_HEIGHT = 21.0

_PADDING = 3
_ARROW_PIXELS = 30
_SHADOW_OFFSET = QPointF(1.5, 2.0)

SIZE_WIDTH = round(_ARROW_PIXELS * _ARROW_WIDTH / _ARROW_HEIGHT) + 2 * _PADDING
SIZE_HEIGHT = _ARROW_PIXELS + 2 * _PADDING

OFFSET = QPoint(12, 12)
TRACK_INTERVAL_MS = 16

# Top-left -> bottom-right gradients, one per application state.
APPEARANCE = {
    ApplicationState.IDLE: (QColor(111, 177, 255), QColor(43, 108, 224)),
    ApplicationState.LISTENING: (QColor(107, 227, 155), QColor(22, 163, 74)),
    ApplicationState.PROCESSING: (QColor(255, 198, 110), QColor(214, 122, 0)),
    ApplicationState.RESPONDING: (QColor(198, 160, 255), QColor(124, 58, 237)),
    ApplicationState.ERROR: (QColor(255, 138, 138), QColor(214, 69, 69)),
}


class Companion(QWidget):
    """Blue cursor-shaped marker drawn just below-right of the pointer.

    The window is frameless, transparent, click-through and never takes focus,
    so it floats over other applications without getting in the way. It is a
    companion beside the real cursor, not a replacement for it.
    """

    def __init__(self, state=ApplicationState.IDLE):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFixedSize(SIZE_WIDTH, SIZE_HEIGHT)

        self._rendered_state = ApplicationState(state)

        self._timer = QTimer(self)
        self._timer.setInterval(TRACK_INTERVAL_MS)
        self._timer.timeout.connect(self._follow_cursor)

    @property
    def state(self):
        """The state currently being rendered (owned by the coordinator)."""
        return self._rendered_state

    def start(self):
        """Show the companion and begin tracking the cursor."""
        self._follow_cursor()
        self.show()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    def render_state(self, state):
        """Render ``state``; the companion keeps no state of its own."""
        state = ApplicationState(state)
        if state is not self._rendered_state:
            self._rendered_state = state
            self.update()

    def _follow_cursor(self):
        self.move(self._clamped_position(QCursor.pos()))

    def _clamped_position(self, cursor):
        """Cursor position plus offset, kept inside the cursor's screen."""
        x = cursor.x() + OFFSET.x()
        y = cursor.y() + OFFSET.y()

        screen = QGuiApplication.screenAt(cursor) or QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            x = min(max(x, area.left()), area.right() - self.width() + 1)
            y = min(max(y, area.top()), area.bottom() - self.height() + 1)
        return QPoint(x, y)

    def _arrow_path(self):
        scale = min(
            (self.width() - 2 * _PADDING) / _ARROW_WIDTH,
            (self.height() - 2 * _PADDING) / _ARROW_HEIGHT,
        )
        points = [
            QPointF(_PADDING + x * scale, _PADDING + y * scale)
            for x, y in _ARROW_POINTS
        ]

        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()
        return path

    def _arrow_brush(self):
        start, end = APPEARANCE.get(self._rendered_state, APPEARANCE[ApplicationState.IDLE])
        gradient = QLinearGradient(0.0, 0.0, float(self.width()), float(self.height()))
        gradient.setColorAt(0.0, start)
        gradient.setColorAt(1.0, end)
        return QBrush(gradient)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = self._arrow_path()

        shadow = QPainterPath(path)
        shadow.translate(_SHADOW_OFFSET)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 70))
        painter.drawPath(shadow)

        outline = QPen(QColor(255, 255, 255, 220), 1.6)
        outline.setJoinStyle(Qt.RoundJoin)
        outline.setCapStyle(Qt.RoundCap)

        painter.setPen(outline)
        painter.setBrush(self._arrow_brush())
        painter.drawPath(path)
        painter.end()
