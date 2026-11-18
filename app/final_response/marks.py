"""Painting the on-screen marks, one function per mark.

``ScreenOverlay`` owns the window and the animation; this module only paints.
Every function draws one mark for a target ``rect`` (logical pixels on the
overlay) whose centre is ``centre``:

    progress  0 -> 1 while the mark draws itself in (0.55 s)
    phase     0 -> 1 repeating, for marks that keep moving (pulses, ripples)
    context   the overlay canvas plus the mark's extra data (number, label, keys,
              scroll direction, second target, zoomed view)

Everything is drawn in the accent colour with a soft glow, so the marks look
like one family. An unknown mark is drawn as a highlight.
"""

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QLinearGradient,
    QPainterPath,
    QPen,
    QPolygonF,
)

ACCENT = QColor(255, 92, 58)
INK = QColor(255, 255, 255, 240)
GLASS = QColor(22, 24, 34, 235)
# Space between a box and the mark drawn around it, in logical pixels.
MARGIN = 8.0
ARROW_LENGTH = 130.0
FONT_FAMILY = "Segoe UI"


@dataclass
class MarkContext:
    """What a mark may need besides its target box."""

    canvas: QRectF
    number: int | None = None
    label: str = ""
    keys: list = field(default_factory=list)
    scroll: str = ""
    to_rect: QRectF | None = None
    to_centre: QPointF | None = None
    zoom: QImage | None = None


def paint_mark(painter, action, rect, centre, progress, phase, context):
    """Paint ``action`` for the target ``rect``; nothing for ``none``."""
    if action in (None, "none"):
        return
    draw = MARKS.get(action, _draw_highlight)
    painter.save()
    try:
        draw(painter, rect, centre, progress, phase, context)
    finally:
        painter.restore()


# --- shared pens and helpers ---------------------------------------------------


def _glow_pen(width, alpha, color=ACCENT):
    color = QColor(color)
    color.setAlpha(max(0, min(255, int(alpha))))
    pen = QPen(color, width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _stroke(painter, path, opacity=1.0):
    """A soft glow under a solid accent line."""
    painter.setBrush(Qt.NoBrush)
    painter.setPen(_glow_pen(12, 60 * opacity))
    painter.drawPath(path)
    painter.setPen(_glow_pen(4, 255 * opacity))
    painter.drawPath(path)


def _partial(path, progress, samples=64):
    """The first ``progress`` of ``path`` as a new path (for marks that draw themselves in)."""
    progress = max(0.0, min(1.0, progress))
    if progress >= 1.0:
        return path
    result = QPainterPath(path.pointAtPercent(0.0))
    steps = max(2, int(samples * progress))
    for index in range(1, steps + 1):
        result.lineTo(path.pointAtPercent(progress * index / steps))
    return result


def _font(size, weight=QFont.DemiBold):
    font = QFont(FONT_FAMILY)
    font.setPixelSize(int(size))
    font.setWeight(weight)
    return font


def _place_beside(rect, size, canvas, gap=14.0):
    """Where to put a ``size`` box next to ``rect``: right, left, below or above, whichever fits."""
    width, height = size
    candidates = [
        QRectF(rect.right() + gap, rect.center().y() - height / 2, width, height),
        QRectF(rect.left() - gap - width, rect.center().y() - height / 2, width, height),
        QRectF(rect.center().x() - width / 2, rect.bottom() + gap, width, height),
        QRectF(rect.center().x() - width / 2, rect.top() - gap - height, width, height),
    ]
    for candidate in candidates:
        if canvas.contains(candidate):
            return candidate
    first = candidates[0]
    x = min(max(first.left(), canvas.left() + 4), canvas.right() - width - 4)
    y = min(max(first.top(), canvas.top() + 4), canvas.bottom() - height - 4)
    return QRectF(x, y, width, height)


def _edge_point(rect, toward):
    """Where the line from ``rect``'s centre toward ``toward`` leaves ``rect``."""
    centre = rect.center()
    dx, dy = toward.x() - centre.x(), toward.y() - centre.y()
    if dx == 0 and dy == 0:
        return centre
    scale_x = (rect.width() / 2) / abs(dx) if dx else math.inf
    scale_y = (rect.height() / 2) / abs(dy) if dy else math.inf
    scale = min(scale_x, scale_y)
    return QPointF(centre.x() + dx * scale, centre.y() + dy * scale)


def _arrow_head(painter, tip, angle_degrees, size=14.0, opacity=1.0):
    radians = math.radians(angle_degrees)
    back = QPointF(-math.cos(radians), math.sin(radians))
    side = QPointF(-back.y(), back.x())
    head = QPolygonF([tip, tip + back * size + side * size * 0.6, tip + back * size - side * size * 0.6])
    color = QColor(ACCENT)
    color.setAlphaF(opacity)
    painter.setPen(Qt.NoPen)
    painter.setBrush(color)
    painter.drawPolygon(head)


def _cursor_shape(tip, size):
    return QPolygonF(
        [
            tip,
            QPointF(tip.x(), tip.y() + 1.0 * size),
            QPointF(tip.x() + 0.27 * size, tip.y() + 0.76 * size),
            QPointF(tip.x() + 0.45 * size, tip.y() + 1.12 * size),
            QPointF(tip.x() + 0.6 * size, tip.y() + 1.05 * size),
            QPointF(tip.x() + 0.42 * size, tip.y() + 0.7 * size),
            QPointF(tip.x() + 0.75 * size, tip.y() + 0.7 * size),
        ]
    )


def _glass_card(painter, card, opacity, radius=10.0):
    fill = QColor(GLASS)
    fill.setAlphaF(fill.alphaF() * opacity)
    painter.setPen(_glow_pen(1.5, 230 * opacity))
    painter.setBrush(fill)
    painter.drawRoundedRect(card, radius, radius)


def _ease_out_back(t):
    c1, c3 = 1.70158, 2.70158
    t = max(0.0, min(1.0, t))
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


# --- the original six ----------------------------------------------------------


def _draw_point(painter, rect, centre, progress, phase, context):
    # A ripple that keeps pulsing, then a pointer whose tip sits on the target.
    ripple = 10 + 26 * phase
    painter.setPen(_glow_pen(3, 200 * (1 - phase)))
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


def _draw_circle(painter, rect, centre, progress, phase, context):
    ring = rect.adjusted(-MARGIN * 1.5, -MARGIN * 1.5, MARGIN * 1.5, MARGIN * 1.5)
    path = QPainterPath()
    path.arcMoveTo(ring, 90)
    path.arcTo(ring, 90, -360 * progress)
    _stroke(painter, path)


def _draw_underline(painter, rect, centre, progress, phase, context):
    y = rect.bottom() + MARGIN / 2
    path = QPainterPath(QPointF(rect.left(), y))
    path.lineTo(rect.left() + rect.width() * progress, y)
    _stroke(painter, path)


def _draw_arrow(painter, rect, centre, progress, phase, context):
    # The arrow comes from the side of the screen with more room, and stops at the box edge.
    canvas = context.canvas
    direction = QPointF(canvas.center().x() - centre.x(), canvas.center().y() - centre.y())
    length = math.hypot(direction.x(), direction.y()) or 1.0
    unit = QPointF(direction.x() / length, direction.y() / length)
    reach_x = (rect.width() / 2) / abs(unit.x()) if unit.x() else math.inf
    reach_y = (rect.height() / 2) / abs(unit.y()) if unit.y() else math.inf
    end = centre + unit * (min(reach_x, reach_y) + MARGIN)
    start = end + unit * ARROW_LENGTH
    head = start + (end - start) * progress

    path = QPainterPath(start)
    path.lineTo(head)
    _stroke(painter, path)
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


def _draw_highlight(painter, rect, centre, progress, phase, context):
    area = rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
    fill = QColor(ACCENT)
    fill.setAlpha(int(55 * progress))
    painter.setPen(Qt.NoPen)
    painter.setBrush(fill)
    painter.drawRoundedRect(area, 10, 10)
    path = QPainterPath()
    path.addRoundedRect(area, 10, 10)
    _stroke(painter, path, opacity=progress)


def _draw_sketch(painter, rect, centre, progress, phase, context):
    # 'draw' marks where an explanatory sketch belongs: a dashed frame that keeps moving.
    area = rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
    pen = _glow_pen(3, 255 * progress)
    pen.setStyle(Qt.DashLine)
    pen.setDashOffset(-phase * 16)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(area, 12, 12)


# --- shapes and attention --------------------------------------------------------


def _draw_rectangle(painter, rect, centre, progress, phase, context):
    area = rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
    path = QPainterPath()
    path.addRoundedRect(area, 6, 6)
    _stroke(painter, _partial(path, progress, samples=96))


def _draw_fill_circle(painter, rect, centre, progress, phase, context):
    pad = MARGIN * 1.5
    ring = rect.adjusted(-pad, -pad, pad, pad)
    fill = QColor(ACCENT)
    fill.setAlpha(int(60 * progress))
    painter.setPen(Qt.NoPen)
    painter.setBrush(fill)
    painter.drawEllipse(ring)
    path = QPainterPath()
    path.arcMoveTo(ring, 90)
    path.arcTo(ring, 90, -360 * progress)
    _stroke(painter, path)


def _draw_marker(painter, rect, centre, progress, phase, context):
    # A highlighter-pen stroke behind the text, slightly taller than it and a little skewed.
    band = QRectF(rect.left() - 4, rect.top() + rect.height() * 0.08, (rect.width() + 8) * progress, rect.height() * 0.9)
    fill = QColor(255, 196, 61, 105)
    painter.setPen(Qt.NoPen)
    painter.setBrush(fill)
    painter.translate(band.center())
    painter.shear(-0.08, 0)
    painter.translate(-band.center())
    painter.drawRoundedRect(band, 4, 4)


def _draw_focus_frame(painter, rect, centre, progress, phase, context):
    # Four corner brackets that snap in from a little further out.
    spread = 18 * (1 - progress)
    area = rect.adjusted(-MARGIN - spread, -MARGIN - spread, MARGIN + spread, MARGIN + spread)
    arm = max(10.0, min(26.0, min(area.width(), area.height()) / 3))
    path = QPainterPath()
    for corner, dx, dy in (
        (area.topLeft(), 1, 1),
        (area.topRight(), -1, 1),
        (area.bottomLeft(), 1, -1),
        (area.bottomRight(), -1, -1),
    ):
        path.moveTo(corner.x() + dx * arm, corner.y())
        path.lineTo(corner)
        path.lineTo(corner.x(), corner.y() + dy * arm)
    _stroke(painter, path, opacity=min(1.0, progress * 1.5))


def _draw_spotlight(painter, rect, centre, progress, phase, context):
    # Everything except the target is dimmed; the hole has a soft glowing rim.
    hole = rect.adjusted(-MARGIN * 2, -MARGIN * 2, MARGIN * 2, MARGIN * 2)
    shade = QPainterPath()
    shade.addRect(context.canvas)
    cut = QPainterPath()
    cut.addRoundedRect(hole, 14, 14)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(8, 10, 16, int(150 * progress)))
    painter.drawPath(shade.subtracted(cut))
    _stroke(painter, cut, opacity=progress)


def _draw_beacon(painter, rect, centre, progress, phase, context):
    base = max(8.0, min(rect.width(), rect.height()) / 2)
    painter.setBrush(Qt.NoBrush)
    for index in range(3):
        wave = (phase + index / 3) % 1.0
        radius = base + 46 * wave
        painter.setPen(_glow_pen(3, 220 * (1 - wave) * progress))
        painter.drawEllipse(centre, radius, radius)
    painter.setPen(Qt.NoPen)
    painter.setBrush(ACCENT)
    painter.drawEllipse(centre, 5 * progress, 5 * progress)


def _draw_glow(painter, rect, centre, progress, phase, context):
    breath = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
    area = rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
    path = QPainterPath()
    path.addRoundedRect(area, 10, 10)
    painter.setBrush(Qt.NoBrush)
    for width, alpha in ((26, 28), (18, 45), (10, 80)):
        painter.setPen(_glow_pen(width + 6 * breath, alpha * progress))
        painter.drawPath(path)
    painter.setPen(_glow_pen(2.5, 255 * progress))
    painter.drawPath(path)


def _draw_bracket(painter, rect, centre, progress, phase, context):
    # A tall bracket on the left of a paragraph or list, growing from its middle.
    x = rect.left() - MARGIN * 1.6
    half = (rect.height() / 2 + MARGIN) * progress
    top, bottom = centre.y() - half, centre.y() + half
    serif = 12.0
    path = QPainterPath(QPointF(x + serif, top))
    path.lineTo(x, top)
    path.lineTo(x, bottom)
    path.lineTo(x + serif, bottom)
    _stroke(painter, path)


def _draw_wavy_underline(painter, rect, centre, progress, phase, context):
    y = rect.bottom() + MARGIN / 2 + 2
    wavelength, amplitude = 12.0, 3.5
    end = rect.left() + rect.width() * progress
    path = QPainterPath(QPointF(rect.left(), y))
    x = rect.left()
    while x < end:
        x = min(x + 2.0, end)
        path.lineTo(x, y + amplitude * math.sin((x - rect.left()) / wavelength * 2 * math.pi))
    _stroke(painter, path)


def _draw_strike(painter, rect, centre, progress, phase, context):
    y = centre.y()
    path = QPainterPath(QPointF(rect.left() - 4, y))
    path.lineTo(rect.left() - 4 + (rect.width() + 8) * progress, y)
    _stroke(painter, path)


def _draw_cross(painter, rect, centre, progress, phase, context):
    area = rect.adjusted(-4, -4, 4, 4)
    first = min(1.0, progress * 2)
    second = max(0.0, progress * 2 - 1)
    path = QPainterPath(area.topLeft())
    path.lineTo(area.topLeft() + (area.bottomRight() - area.topLeft()) * first)
    if second > 0:
        path.moveTo(area.topRight())
        path.lineTo(area.topRight() + (area.bottomLeft() - area.topRight()) * second)
    _stroke(painter, path)


def _draw_check(painter, rect, centre, progress, phase, context):
    size = max(20.0, min(40.0, rect.height() * 1.1))
    left = rect.right() + MARGIN * 1.5
    if left + size > context.canvas.right():
        left = rect.left() - MARGIN * 1.5 - size
    mid_y = centre.y()
    tick = QPainterPath(QPointF(left, mid_y))
    tick.lineTo(left + size * 0.35, mid_y + size * 0.35)
    tick.lineTo(left + size, mid_y - size * 0.45)
    _stroke(painter, _partial(tick, progress))


def _draw_cursor_click(painter, rect, centre, progress, phase, context):
    # A cursor glides in from the lower right, then clicks: the ripple repeats while it waits.
    eased = 1 - (1 - progress) ** 3
    start = QPointF(centre.x() + 110, centre.y() + 85)
    tip = start + (centre - start) * eased
    if progress >= 1.0:
        ripple = 6 + 30 * phase
        painter.setPen(_glow_pen(3, 220 * (1 - phase)))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, ripple, ripple)
        press = 0.92 if phase < 0.15 else 1.0
    else:
        press = 1.0
    size = 28 * press
    painter.setPen(QPen(QColor(20, 20, 24, 235), 2))
    painter.setBrush(QColor(255, 255, 255, 245))
    painter.drawPolygon(_cursor_shape(tip, size))


# --- marks with a little data --------------------------------------------------------


def _draw_badge(painter, rect, centre, progress, phase, context):
    radius = 15 * _ease_out_back(progress)
    if radius <= 0.5:
        return
    spot = QPointF(rect.left() - 2, rect.top() - 2)
    ring = 15 + 10 * phase
    painter.setPen(_glow_pen(2.5, 160 * (1 - phase) * progress))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(spot, ring, ring)
    painter.setPen(QPen(QColor(255, 255, 255, 235), 2))
    painter.setBrush(ACCENT)
    painter.drawEllipse(spot, radius, radius)
    if context.number is not None and progress > 0.5:
        painter.setFont(_font(16, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(spot.x() - radius, spot.y() - radius, 2 * radius, 2 * radius), Qt.AlignCenter, str(context.number))
    # A faint outline ties the badge to its element.
    outline = QPainterPath()
    outline.addRoundedRect(rect.adjusted(-4, -4, 4, 4), 6, 6)
    painter.setPen(_glow_pen(2, 140 * progress))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(outline)


def _draw_callout(painter, rect, centre, progress, phase, context):
    if not context.label:
        _draw_highlight(painter, rect, centre, progress, phase, context)
        return
    font = _font(15)
    metrics = QFontMetricsF(font)
    width = metrics.horizontalAdvance(context.label) + 28
    height = metrics.height() + 16
    card = _place_beside(rect, (width, height), context.canvas, gap=26)

    outline = QPainterPath()
    outline.addRoundedRect(rect.adjusted(-4, -4, 4, 4), 6, 6)
    painter.setPen(_glow_pen(2.5, 255 * progress))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(outline)

    start = _edge_point(rect.adjusted(-4, -4, 4, 4), card.center())
    end = _edge_point(card, rect.center())
    leader = QPainterPath(start)
    leader.lineTo(start + (end - start) * min(1.0, progress * 1.4))
    painter.setPen(_glow_pen(2, 230 * progress))
    painter.drawPath(leader)

    opacity = max(0.0, min(1.0, progress * 1.6 - 0.4))
    if opacity <= 0:
        return
    _glass_card(painter, card, opacity)
    painter.setFont(font)
    color = QColor(INK)
    color.setAlphaF(opacity)
    painter.setPen(color)
    painter.drawText(card, Qt.AlignCenter, context.label)


def _draw_key_badge(painter, rect, centre, progress, phase, context):
    keys = [str(key) for key in context.keys if str(key).strip()]
    if not keys:
        _draw_highlight(painter, rect, centre, progress, phase, context)
        return
    font = _font(15, QFont.Bold)
    metrics = QFontMetricsF(font)
    caps = [max(38.0, metrics.horizontalAdvance(key) + 22) for key in keys]
    plus = 18.0
    height = 36.0
    width = sum(caps) + plus * (len(caps) - 1)
    row = _place_beside(rect, (width, height + 4), context.canvas, gap=16)

    outline = QPainterPath()
    outline.addRoundedRect(rect.adjusted(-4, -4, 4, 4), 6, 6)
    painter.setPen(_glow_pen(2.5, 255 * progress))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(outline)

    painter.setFont(font)
    x = row.left()
    for index, (key, cap_width) in enumerate(zip(keys, caps)):
        appear = max(0.0, min(1.0, progress * 2 - index * 0.35))
        if appear <= 0:
            break
        cap = QRectF(x, row.top() + 4 * (1 - appear), cap_width, height)
        shadow = QColor(0, 0, 0, int(110 * appear))
        painter.setPen(Qt.NoPen)
        painter.setBrush(shadow)
        painter.drawRoundedRect(cap.translated(0, 4), 7, 7)
        face = QLinearGradient(cap.topLeft(), cap.bottomLeft())
        face.setColorAt(0, QColor(250, 250, 252, int(255 * appear)))
        face.setColorAt(1, QColor(222, 225, 232, int(255 * appear)))
        painter.setPen(QPen(QColor(150, 155, 165, int(255 * appear)), 1.2))
        painter.setBrush(QBrush(face))
        painter.drawRoundedRect(cap, 7, 7)
        painter.setPen(QColor(30, 32, 40, int(255 * appear)))
        painter.drawText(cap, Qt.AlignCenter, key)
        x += cap_width
        if index < len(keys) - 1:
            painter.setPen(QColor(255, 255, 255, int(230 * appear)))
            painter.drawText(QRectF(x, cap.top(), plus, height), Qt.AlignCenter, "+")
            x += plus


def _draw_scroll_hint(painter, rect, centre, progress, phase, context):
    if context.scroll not in ("up", "down"):
        _draw_beacon(painter, rect, centre, progress, phase, context)
        return
    # A faint frame on the area to scroll, a mouse beside it and chevrons moving the scroll way.
    frame = QPainterPath()
    frame.addRoundedRect(rect.adjusted(-4, -4, 4, 4), 8, 8)
    pen = _glow_pen(2, 150 * progress)
    pen.setStyle(Qt.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(frame)

    slot = _place_beside(rect, (40, 112), context.canvas, gap=18)
    mouse = QRectF(slot.center().x() - 15, slot.center().y() - 22, 30, 44)
    painter.setPen(QPen(QColor(255, 255, 255, int(235 * progress)), 2))
    painter.setBrush(QColor(22, 24, 34, int(220 * progress)))
    painter.drawRoundedRect(mouse, 15, 15)
    wheel_y = mouse.top() + 10 + 6 * (phase if context.scroll == "down" else 1 - phase)
    painter.setPen(_glow_pen(3, 255 * progress))
    painter.drawLine(QPointF(mouse.center().x(), wheel_y), QPointF(mouse.center().x(), wheel_y + 6))

    sign = 1 if context.scroll == "down" else -1
    base_y = mouse.bottom() + 8 if sign > 0 else mouse.top() - 8
    for index in range(3):
        wave = (phase + index / 3) % 1.0
        y = base_y + sign * (index * 9 + 4 * wave)
        alpha = 255 * progress * (1 - wave * 0.6) * (1 - index * 0.25)
        chevron = QPainterPath(QPointF(mouse.center().x() - 8, y - sign * 4))
        chevron.lineTo(mouse.center().x(), y + sign * 4)
        chevron.lineTo(mouse.center().x() + 8, y - sign * 4)
        painter.setPen(_glow_pen(3, alpha))
        painter.drawPath(chevron)


# --- a second target, or a zoomed view ------------------------------------------------


def _draw_connector(painter, rect, centre, progress, phase, context):
    if context.to_rect is None:
        _draw_arrow(painter, rect, centre, progress, phase, context)
        return
    source = rect.adjusted(-4, -4, 4, 4)
    target = context.to_rect.adjusted(-4, -4, 4, 4)
    for area in (source, target):
        outline = QPainterPath()
        outline.addRoundedRect(area, 6, 6)
        painter.setPen(_glow_pen(2, 170 * progress))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(outline)

    start = _edge_point(source, target.center())
    end = _edge_point(target, source.center())
    dx, dy = end.x() - start.x(), end.y() - start.y()
    distance = math.hypot(dx, dy) or 1.0
    # Bend to one side by a fifth of the distance, so the arrow reads as a gesture.
    normal = QPointF(-dy / distance, dx / distance) * min(120.0, distance * 0.22)
    path = QPainterPath(start)
    path.cubicTo(start + QPointF(dx, dy) * 0.25 + normal, start + QPointF(dx, dy) * 0.75 + normal, end)
    _stroke(painter, _partial(path, progress, samples=80))
    if progress > 0.9:
        _arrow_head(painter, end, path.angleAtPercent(1.0), opacity=min(1.0, (progress - 0.9) * 10))
    painter.setPen(Qt.NoPen)
    painter.setBrush(ACCENT)
    painter.drawEllipse(start, 4, 4)


def _draw_magnifier(painter, rect, centre, progress, phase, context):
    if context.zoom is None or context.zoom.isNull():
        _draw_circle(painter, rect, centre, progress, phase, context)
        return
    zoom = context.zoom
    radius = max(64.0, min(130.0, max(rect.width(), rect.height()) * 1.15))
    scale = _ease_out_back(progress)
    slot = _place_beside(rect, (2 * radius, 2 * radius), context.canvas, gap=30)
    lens_centre = slot.center()
    current = radius * max(0.0, scale)
    if current < 2:
        return

    start = _edge_point(rect.adjusted(-4, -4, 4, 4), lens_centre)
    end = QPointF(lens_centre)
    painter.setPen(_glow_pen(2, 200 * progress))
    painter.drawLine(start, start + (end - start) * min(1.0, progress))
    outline = QPainterPath()
    outline.addRoundedRect(rect.adjusted(-4, -4, 4, 4), 6, 6)
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(outline)

    lens = QRectF(lens_centre.x() - current, lens_centre.y() - current, 2 * current, 2 * current)
    clip = QPainterPath()
    clip.addEllipse(lens)
    painter.save()
    painter.setClipPath(clip)
    painter.fillRect(lens, QColor(14, 16, 22))
    # The zoomed image fills the lens, keeping its proportions.
    image_ratio = zoom.width() / max(zoom.height(), 1)
    if image_ratio >= 1:
        target = QRectF(0, 0, lens.width() * 1.0, lens.width() / image_ratio)
    else:
        target = QRectF(0, 0, lens.height() * image_ratio, lens.height())
    target.moveCenter(lens.center())
    painter.drawImage(target, zoom)
    painter.restore()
    _stroke(painter, clip)


MARKS = {
    "point": _draw_point,
    "circle": _draw_circle,
    "underline": _draw_underline,
    "arrow": _draw_arrow,
    "highlight": _draw_highlight,
    "draw": _draw_sketch,
    "rectangle": _draw_rectangle,
    "fill_circle": _draw_fill_circle,
    "marker": _draw_marker,
    "focus_frame": _draw_focus_frame,
    "spotlight": _draw_spotlight,
    "beacon": _draw_beacon,
    "glow": _draw_glow,
    "bracket": _draw_bracket,
    "wavy_underline": _draw_wavy_underline,
    "strike": _draw_strike,
    "cross": _draw_cross,
    "check": _draw_check,
    "cursor_click": _draw_cursor_click,
    "badge": _draw_badge,
    "callout": _draw_callout,
    "key_badge": _draw_key_badge,
    "scroll_hint": _draw_scroll_hint,
    "connector": _draw_connector,
    "magnifier": _draw_magnifier,
}
