"""A small 'liquid glass' pill at the top of the screen showing what Hey Arro is doing.

It appears when Ctrl + Alt is pressed (Listening), follows the flow
(Transcribing, Refining, Planning, Responding, Speaking) and fades out when the
interaction ends. It never takes focus and lets every click through.

The glass is painted: a tinted translucent body, a soft sheen across the top,
a thin light rim and a moving 'liquid' orb in the status colour. Qt has no
portable background blur, so the body is tinted enough to stay readable on any
wallpaper.
"""

import math

from PySide6.QtCore import QPointF, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from app.status import AssistantStatus

WIDTH = 236
HEIGHT = 58
SHADOW = 14
TOP_OFFSET = 26
FRAME_MS = 33
FADE_MS = 220

COLORS = {
    AssistantStatus.LISTENING: QColor(255, 77, 109),
    AssistantStatus.TRANSCRIBING: QColor(255, 176, 32),
    AssistantStatus.REFINING: QColor(45, 212, 191),
    AssistantStatus.PLANNING: QColor(139, 92, 246),
    AssistantStatus.RESPONDING: QColor(59, 130, 246),
    AssistantStatus.SPEAKING: QColor(34, 197, 94),
}


class StatusIndicator(QWidget):
    """Shows the current AssistantStatus; IDLE fades it out."""

    def __init__(self):
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
        self.setFixedSize(WIDTH + 2 * SHADOW, HEIGHT + 2 * SHADOW)

        self._status = AssistantStatus.IDLE
        self._phase = 0.0

        self._frames = QTimer(self)
        self._frames.setInterval(FRAME_MS)
        self._frames.timeout.connect(self._on_frame)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(FADE_MS)
        self._fade.finished.connect(self._on_faded)

    @property
    def status(self):
        return self._status

    def set_status(self, status):
        status = AssistantStatus(status)
        previous, self._status = self._status, status
        if status is AssistantStatus.IDLE:
            if previous is not AssistantStatus.IDLE:
                self._fade_to(0.0)
            return
        if previous is AssistantStatus.IDLE or not self.isVisible():
            self._place()
            self.setWindowOpacity(0.0)
            self.show()
            self.raise_()
            self._frames.start()
            self._fade_to(1.0)
        self.update()

    # --- animation ----------------------------------------------------------

    def _place(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(area.center().x() - self.width() // 2, area.top() + TOP_OFFSET - SHADOW)

    def _fade_to(self, opacity):
        self._fade.stop()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(opacity)
        self._fade.start()

    def _on_faded(self):
        if self._status is AssistantStatus.IDLE:
            self._frames.stop()
            self.hide()

    def _on_frame(self):
        self._phase = (self._phase + FRAME_MS / 1000.0) % 1000.0
        self.update()

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event):
        if self._status is AssistantStatus.IDLE and not self.isVisible():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        body = QRectF(SHADOW, SHADOW, WIDTH, HEIGHT)
        radius = HEIGHT / 2

        self._paint_shadow(painter, body, radius)
        self._paint_glass(painter, body, radius)
        self._paint_orb(painter, body)
        self._paint_label(painter, body)
        painter.end()

    def _paint_shadow(self, painter, body, radius):
        painter.setPen(Qt.NoPen)
        for step in range(SHADOW, 0, -2):
            shade = QColor(0, 0, 0, int(26 * (1 - step / SHADOW)))
            painter.setBrush(shade)
            painter.drawRoundedRect(body.adjusted(-step, -step + 4, step, step + 4), radius + step, radius + step)

    def _paint_glass(self, painter, body, radius):
        shape = QPainterPath()
        shape.addRoundedRect(body, radius, radius)

        tint = QLinearGradient(body.topLeft(), body.bottomLeft())
        tint.setColorAt(0.0, QColor(58, 62, 78, 200))
        tint.setColorAt(1.0, QColor(22, 24, 34, 215))
        painter.fillPath(shape, tint)

        # A faint wash of the status colour, as if light passed through coloured glass.
        wash = QRadialGradient(QPointF(body.left() + 34, body.center().y()), WIDTH * 0.7)
        color = QColor(COLORS.get(self._status, QColor(255, 255, 255)))
        color.setAlpha(48)
        wash.setColorAt(0.0, color)
        wash.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillPath(shape, wash)

        # Sheen across the top half.
        sheen_rect = QRectF(body.left() + 3, body.top() + 2, body.width() - 6, body.height() * 0.5)
        sheen = QPainterPath()
        sheen.addRoundedRect(sheen_rect, radius - 3, radius - 3)
        gloss = QLinearGradient(sheen_rect.topLeft(), sheen_rect.bottomLeft())
        gloss.setColorAt(0.0, QColor(255, 255, 255, 70))
        gloss.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillPath(sheen, gloss)

        rim = QLinearGradient(body.topLeft(), body.bottomLeft())
        rim.setColorAt(0.0, QColor(255, 255, 255, 150))
        rim.setColorAt(1.0, QColor(255, 255, 255, 35))
        painter.setPen(QPen(rim, 1.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(shape)

    def _paint_orb(self, painter, body):
        color = COLORS.get(self._status, QColor(255, 255, 255))
        centre = QPointF(body.left() + 30, body.center().y())
        size = 15.0
        # Listening and Speaking breathe more, as if following a voice.
        lively = self._status in (AssistantStatus.LISTENING, AssistantStatus.SPEAKING)
        breath = 1.0 + (0.16 if lively else 0.07) * math.sin(self._phase * 2 * math.pi * (1.6 if lively else 0.9))

        halo = QRadialGradient(centre, size * 1.9 * breath)
        glow = QColor(color)
        glow.setAlpha(110)
        halo.setColorAt(0.0, glow)
        halo.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(centre, size * 1.9 * breath, size * 1.9 * breath)

        orb = QPainterPath()
        orb.addEllipse(centre, size * breath, size * breath)
        painter.save()
        painter.setClipPath(orb)
        base = QRadialGradient(centre - QPointF(4, 5), size * 1.4)
        base.setColorAt(0.0, color.lighter(150))
        base.setColorAt(1.0, color.darker(140))
        painter.fillPath(orb, base)
        # Three drifting blobs make the orb look liquid.
        for index in range(3):
            angle = self._phase * 2 * math.pi * (0.6 + 0.25 * index) + index * 2.1
            offset = QPointF(math.cos(angle) * size * 0.45, math.sin(angle * 1.3) * size * 0.45)
            blob = QColor(color.lighter(130 + 20 * index))
            blob.setAlpha(150)
            painter.setBrush(blob)
            painter.drawEllipse(centre + offset, size * 0.55, size * 0.55)
        highlight = QRadialGradient(centre - QPointF(size * 0.35, size * 0.45), size * 0.6)
        highlight.setColorAt(0.0, QColor(255, 255, 255, 190))
        highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillPath(orb, highlight)
        painter.restore()

        painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(orb)

    def _paint_label(self, painter, body):
        label = self._status.label
        dots = "." * (1 + int(self._phase * 2.5) % 3)
        font = QFont("Segoe UI", 11)
        font.setWeight(QFont.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 235))
        text_rect = QRectF(body.left() + 58, body.top(), body.width() - 70, body.height())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, f"{label}{dots}")
