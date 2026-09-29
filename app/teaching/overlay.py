"""Full-screen transparent overlay that draws Arro's visual teaching.

One click-through window shows everything: the pointer, highlights, boxes,
circles, underlines and labels. It is frameless, always on top, never takes
focus and is transparent for mouse input, so it floats above the application
being taught without blocking it.

The overlay only knows screen-space primitives (see ``primitives.py``); it never
sees the response model, the coordinate mapper or speech. New primitives appear
immediately and then ease to full strength, removed ones fade out, and the
pointer eases towards its target - so a step change reads as a transition rather
than a flicker.
"""

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget

from app.llm.response import VisualActionType
from app.teaching.coordinates import ScreenGeometry

FRAME_MS = 16
EASE = 0.22
FADE_STEP = 0.18
PHASE_STEP = 0.08
NEW_ALPHA = 0.55

HIGHLIGHT_RADIUS = 30.0
HIGHLIGHT_PULSE = 0.10
DEFAULT_MARKER = 64.0
UNDERLINE_LENGTH = 150.0
LABEL_GAP = 16.0
POINTER_OFFSET = 54.0
MARGIN = 12.0

_POINT_KIND = VisualActionType.POINT.value
_HIGHLIGHT_KIND = VisualActionType.HIGHLIGHT.value
_BOX_KIND = VisualActionType.BOX.value
_CIRCLE_KIND = VisualActionType.CIRCLE.value
_UNDERLINE_KIND = VisualActionType.UNDERLINE.value

_ACCENT = QColor(198, 160, 255)
_ACCENT_DEEP = QColor(124, 58, 237)
_FILL = QColor(124, 58, 237, 70)
_LABEL_BACKGROUND = QColor(18, 18, 24, 232)
_LABEL_BORDER = QColor(198, 160, 255, 200)
_POINTER_TOP = QColor(198, 160, 255)
_POINTER_BOTTOM = QColor(124, 58, 237)

# Arrow silhouette: tip at the origin, pointing along +x, body trailing behind.
_ARROW_POINTS = (
    (0.0, 0.0),
    (-30.0, -16.0),
    (-20.0, 0.0),
    (-30.0, 16.0),
)


def with_alpha(color, alpha):
    """Return ``color`` at ``alpha`` (0..1) of its own opacity."""
    alpha = max(0.0, min(1.0, alpha))
    return QColor(color.red(), color.green(), color.blue(), int(round(color.alpha() * alpha)))


def primary_screen_geometry():
    """Return the primary screen's logical geometry as a ScreenGeometry."""
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return ScreenGeometry(0, 0, 1920, 1080, name="primary")
    rect = screen.geometry()
    return ScreenGeometry(
        rect.left(), rect.top(), rect.width(), rect.height(), name=screen.name()
    )


@dataclass
class _Shape:
    """One primitive being drawn, with its own fade state."""

    kind: str
    x: float
    y: float
    width: float | None = None
    height: float | None = None
    label: str | None = None
    alpha: float = NEW_ALPHA
    target_alpha: float = 1.0
    dying: bool = False
    phase: float = 0.0

    @property
    def key(self):
        return (self.kind, round(self.x), round(self.y), self.width, self.height, self.label)

    @property
    def has_bounds(self):
        return bool(self.width and self.height)


class TeachingOverlay(QWidget):
    """Click-through overlay that draws Arro's visual teaching primitives."""

    def __init__(self, screen, parent=None):
        super().__init__(parent)
        self._screen = screen
        self._origin = QPointF(float(screen.left), float(screen.top))

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
        self.setFocusPolicy(Qt.NoFocus)
        self.setGeometry(screen.left, screen.top, screen.width, screen.height)

        self._shapes: list[_Shape] = []
        self._pointer = QPointF()
        self._pointer_anchor = QPointF()
        self._pointer_target = QPointF()
        self._pointer_alpha = 0.0
        self._pointer_alpha_target = 0.0
        self._pointer_on_screen = False
        self._phase = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # --- public API ---------------------------------------------------------

    @property
    def shapes(self):
        """The primitives currently drawn, as ``(kind, x, y, width, height, label)``."""
        return [shape.key for shape in self._shapes if not shape.dying]

    @property
    def pointer_visible(self):
        """True while the pointer is (or is on its way to being) shown."""
        return self._pointer_alpha_target > 0.0

    def show_actions(self, primitives):
        """Show exactly ``primitives``, transitioning to them from what is up.

        An empty list fades everything out; nothing is drawn that was not asked
        for.
        """
        primitives = list(primitives or ())
        incoming = {self._primitive_key(primitive): primitive for primitive in primitives}

        for shape in self._shapes:
            if shape.key in incoming:
                shape.dying = False
                shape.target_alpha = 1.0
            else:
                shape.dying = True
                shape.target_alpha = 0.0

        existing = {shape.key for shape in self._shapes}
        for key, primitive in incoming.items():
            if key in existing:
                continue
            self._shapes.append(
                _Shape(
                    kind=primitive.kind,
                    x=float(primitive.x),
                    y=float(primitive.y),
                    width=float(primitive.width) if primitive.width else None,
                    height=float(primitive.height) if primitive.height else None,
                    label=primitive.label,
                    phase=len(self._shapes) * 0.7,
                )
            )

        point = next((item for item in primitives if item.is_point), None)
        if point is not None:
            self._pointer_target = self._to_local(point.x, point.y)
            self._pointer_anchor = self._anchor_for(self._pointer_target)
            if not self._pointer_on_screen:
                # Appear already in place so the words and the visual agree.
                self._pointer = QPointF(self._pointer_anchor)
                self._pointer_on_screen = True
            self._pointer_alpha_target = 1.0
        else:
            # A point is only shown when the step asks for one.
            self._pointer_alpha_target = 0.0

        if self._shapes or self._pointer_on_screen:
            self._start()
            # Paint now instead of on the next frame: the visual must already be
            # on screen when the sentence for this step starts.
            self.repaint()

    def clear(self):
        """Immediately remove everything and hide the overlay."""
        self._timer.stop()
        self._shapes.clear()
        self._pointer_on_screen = False
        self._pointer_alpha = 0.0
        self._pointer_alpha_target = 0.0
        super().hide()
        self.update()

    # --- animation ----------------------------------------------------------

    def _start(self):
        self.show()
        self.raise_()
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self):
        for shape in list(self._shapes):
            shape.alpha += (shape.target_alpha - shape.alpha) * FADE_STEP
            if shape.dying and shape.alpha < 0.02:
                self._shapes.remove(shape)

        if self._pointer_on_screen:
            self._pointer = QPointF(
                self._pointer.x() + (self._pointer_anchor.x() - self._pointer.x()) * EASE,
                self._pointer.y() + (self._pointer_anchor.y() - self._pointer.y()) * EASE,
            )
            self._pointer_alpha += (self._pointer_alpha_target - self._pointer_alpha) * FADE_STEP

        self._phase = (self._phase + PHASE_STEP) % (2.0 * math.pi)

        if not self._shapes and self._pointer_alpha < 0.02:
            # Nothing left on screen: stop animating and get out of the way.
            self._timer.stop()
            self._pointer_on_screen = False
            super().hide()
        self.update()

    def _primitive_key(self, primitive):
        return (
            primitive.kind,
            round(primitive.x),
            round(primitive.y),
            primitive.width,
            primitive.height,
            primitive.label,
        )

    def _to_local(self, x, y):
        return QPointF(float(x) - self._origin.x(), float(y) - self._origin.y())

    def _anchor_for(self, point):
        """Where the pointer sits: offset from its target, toward the centre."""
        dx = 1.0 if point.x() < self.width() / 2.0 else -1.0
        dy = 1.0 if point.y() < self.height() / 2.0 else -1.0
        x = min(max(point.x() + dx * POINTER_OFFSET, MARGIN), self.width() - MARGIN)
        y = min(max(point.y() + dy * POINTER_OFFSET, MARGIN), self.height() - MARGIN)
        return QPointF(x, y)

    def _shape_bounds(self, shape):
        """The rectangle a shape draws in, using a sensible size for bare points."""
        if shape.has_bounds:
            width, height = shape.width, shape.height
        elif shape.kind == _UNDERLINE_KIND:
            width, height = UNDERLINE_LENGTH, 0.0
        else:
            width = height = DEFAULT_MARKER
        return QRectF(shape.x - width / 2.0, shape.y - height / 2.0, width, height)

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        for shape in self._shapes:
            self._paint_shape(painter, shape)
        self._paint_pointer(painter)
        painter.end()

    def _paint_shape(self, painter, shape):
        if shape.alpha <= 0.01:
            return

        pulse = 1.0 + HIGHLIGHT_PULSE * math.sin(self._phase + shape.phase)
        if shape.kind == _HIGHLIGHT_KIND:
            self._paint_highlight(painter, shape, pulse)
        elif shape.kind == _BOX_KIND:
            self._paint_box(painter, shape, pulse)
        elif shape.kind == _CIRCLE_KIND:
            self._paint_circle(painter, shape, pulse)
        elif shape.kind == _UNDERLINE_KIND:
            self._paint_underline(painter, shape, pulse)
        elif shape.kind == _POINT_KIND:
            self._paint_point(painter, shape, pulse)

        if shape.label:
            self._paint_label(painter, shape)

    def _paint_highlight(self, painter, shape, pulse):
        """Soft filled region: a rounded box when bounded, a glow when not."""
        if shape.has_bounds:
            rect = self._shape_bounds(shape)
            painter.setPen(Qt.NoPen)
            painter.setBrush(with_alpha(_FILL, shape.alpha))
            painter.drawRoundedRect(rect, 12.0, 12.0)
            painter.setPen(QPen(with_alpha(_ACCENT, shape.alpha), 2.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, 12.0, 12.0)
            return

        radius = HIGHLIGHT_RADIUS * pulse
        centre = QPointF(shape.x, shape.y)
        painter.setPen(Qt.NoPen)
        painter.setBrush(with_alpha(_FILL, shape.alpha))
        painter.drawEllipse(centre, radius + 10.0, radius + 10.0)
        painter.setPen(QPen(with_alpha(_ACCENT, shape.alpha), 3.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, radius, radius)

    def _paint_box(self, painter, shape, pulse):
        """Outlined rectangle around the region."""
        rect = self._shape_bounds(shape)
        painter.setPen(Qt.NoPen)
        painter.setBrush(with_alpha(_FILL, shape.alpha * 0.55))
        painter.drawRoundedRect(rect, 10.0, 10.0)
        painter.setPen(QPen(with_alpha(_ACCENT, shape.alpha * pulse), 3.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 10.0, 10.0)

    def _paint_circle(self, painter, shape, pulse):
        """Outlined ellipse around the region (a circle for bare points)."""
        rect = self._shape_bounds(shape)
        painter.setPen(Qt.NoPen)
        painter.setBrush(with_alpha(_FILL, shape.alpha * 0.45))
        painter.drawEllipse(rect)
        painter.setPen(QPen(with_alpha(_ACCENT, shape.alpha * pulse), 3.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect)

    def _paint_underline(self, painter, shape, pulse):
        """Emphasising line under the region's bottom edge."""
        rect = self._shape_bounds(shape)
        y = rect.bottom() + 3.0
        painter.setPen(QPen(with_alpha(_ACCENT_DEEP, shape.alpha * 0.45), 9.0))
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QPen(with_alpha(_ACCENT, shape.alpha * pulse), 3.0))
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    def _paint_point(self, painter, shape, pulse):
        """A small marker where Arro is pointing, under the arrow."""
        centre = QPointF(shape.x, shape.y)
        painter.setPen(Qt.NoPen)
        painter.setBrush(with_alpha(_FILL, shape.alpha))
        painter.drawEllipse(centre, 15.0 * pulse, 15.0 * pulse)
        painter.setPen(QPen(with_alpha(_ACCENT, shape.alpha), 2.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, 11.0, 11.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(with_alpha(QColor(255, 255, 255), shape.alpha))
        painter.drawEllipse(centre, 5.0, 5.0)

    def _paint_label(self, painter, shape):
        font = painter.font()
        font.setPointSizeF(11.0)
        font.setBold(True)
        painter.setFont(font)

        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(shape.label) + 24.0
        height = metrics.height() + 12.0

        bounds = self._shape_bounds(shape)
        x = bounds.center().x() - width / 2.0
        y = bounds.bottom() + LABEL_GAP
        if y + height > self.height() - MARGIN:
            y = bounds.top() - LABEL_GAP - height
        x = min(max(x, MARGIN), self.width() - width - MARGIN)
        y = min(max(y, MARGIN), self.height() - height - MARGIN)

        rect = QRectF(x, y, width, height)
        painter.setPen(QPen(with_alpha(_LABEL_BORDER, shape.alpha), 1.4))
        painter.setBrush(with_alpha(_LABEL_BACKGROUND, shape.alpha))
        painter.drawRoundedRect(rect, 9.0, 9.0)

        painter.setPen(with_alpha(QColor(255, 255, 255), shape.alpha))
        painter.drawText(rect, Qt.AlignCenter, shape.label)

    def _paint_pointer(self, painter):
        if not self._pointer_on_screen or self._pointer_alpha <= 0.02:
            return

        vector = self._pointer_target - self._pointer
        length = math.hypot(vector.x(), vector.y())
        angle = 0.0 if length < 1e-3 else math.degrees(math.atan2(vector.y(), vector.x()))

        painter.save()
        painter.setOpacity(self._pointer_alpha)
        painter.translate(self._pointer)
        painter.rotate(angle)
        scale = 0.6 + 0.4 * self._pointer_alpha
        painter.scale(scale, scale)

        path = self._arrow_path()
        shadow = QPainterPath(path)
        shadow.translate(1.5, 2.5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 80))
        painter.drawPath(shadow)

        gradient = QLinearGradient(-30.0, -16.0, 0.0, 16.0)
        gradient.setColorAt(0.0, _POINTER_BOTTOM)
        gradient.setColorAt(1.0, _POINTER_TOP)
        pen = QPen(QColor(255, 255, 255, 220), 2.0)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)

        painter.setPen(pen)
        painter.setBrush(gradient)
        painter.drawPath(path)
        painter.restore()

    def _arrow_path(self):
        path = QPainterPath(QPointF(*_ARROW_POINTS[0]))
        for x, y in _ARROW_POINTS[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()
        return path
