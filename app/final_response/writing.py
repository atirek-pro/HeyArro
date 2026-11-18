"""Hey Arro's handwriting: short lines written onto a small glass board on the screen.

A step of the plan may add a few lines (a formula, a worked calculation, a short
note or quote). They are drawn in Segoe Print, the handwriting font that ships with
Windows and covers maths symbols. A line with ``$...$`` is laid out as a formula by
matplotlib's mathtext (fractions, powers, roots) in the same font - no LaTeX is
needed - and a line that cannot be parsed is written as plain text instead.

The writing appears as if being written: each line is revealed left to right
behind a moving pen nib, at roughly handwriting speed, while its sentence is spoken.
"""

import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen

from app.final_response.marks import ACCENT, GLASS, INK, _edge_point, _font, _glow_pen

logger = logging.getLogger(__name__)

HAND_FONT = "Segoe Print"
FORMULA_SIZE = 32.0
NOTE_SIZE = 25.0
PADDING_X = 22.0
PADDING_TOP = 30.0
PADDING_BOTTOM = 18.0
LINE_GAP = 10.0
# Handwriting speed for the reveal, and its limits.
CHARS_PER_SECOND = 14.0
MIN_WRITE_MS = 900
MAX_WRITE_MS = 3600
KIND_LABELS = {"formula": "formula", "note": "note", "quote": "quote", "steps": "steps"}


@dataclass
class WritingLayout:
    """The written lines as shapes, ready to be placed on a board."""

    kind: str
    lines: list = field(default_factory=list)  # (QPainterPath, width, height) per line, top-left at (0, 0)
    text_length: int = 0

    @property
    def width(self):
        return max((w for _, w, _ in self.lines), default=0.0) + 2 * PADDING_X

    @property
    def height(self):
        body = sum(h for _, _, h in self.lines) + LINE_GAP * max(0, len(self.lines) - 1)
        return body + PADDING_TOP + PADDING_BOTTOM

    def write_ms(self):
        """How long writing these lines takes."""
        return int(max(MIN_WRITE_MS, min(MAX_WRITE_MS, self.text_length / CHARS_PER_SECOND * 1000)))


@lru_cache(maxsize=1)
def _configure_mathtext():
    """Point matplotlib's formula renderer at the handwriting font (once)."""
    import matplotlib

    # Missing glyphs (such as the tall root sign) fall back quietly; keep the log clean.
    logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    matplotlib.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": HAND_FONT,
            "mathtext.it": HAND_FONT,
            "mathtext.bf": f"{HAND_FONT}:bold",
            "mathtext.sf": HAND_FONT,
            "mathtext.fallback": "stixsans",
        }
    )
    return True


def _formula_path(text, size):
    """``text`` with ``$...$`` formulas as a Qt path, or None when it cannot be laid out."""
    try:
        _configure_mathtext()
        from matplotlib.font_manager import FontProperties
        from matplotlib.path import Path as MplPath
        from matplotlib.textpath import TextPath

        source = TextPath((0, 0), text, prop=FontProperties(family=HAND_FONT, size=size))
    except Exception as exc:
        logger.info("[Writing] could not lay out %r as a formula (%s); writing it as text", text, exc)
        return None

    path = QPainterPath()
    path.setFillRule(Qt.WindingFill)
    vertices = source.vertices
    codes = source.codes
    index = 0
    # matplotlib's y axis points up; Qt's points down.
    point = lambda i: QPointF(float(vertices[i][0]), float(-vertices[i][1]))  # noqa: E731
    while index < len(vertices):
        code = codes[index] if codes is not None else (MplPath.MOVETO if index == 0 else MplPath.LINETO)
        if code == MplPath.MOVETO:
            path.moveTo(point(index))
            index += 1
        elif code == MplPath.LINETO:
            path.lineTo(point(index))
            index += 1
        elif code == MplPath.CURVE3:
            path.quadTo(point(index), point(index + 1))
            index += 2
        elif code == MplPath.CURVE4:
            path.cubicTo(point(index), point(index + 1), point(index + 2))
            index += 3
        else:  # CLOSEPOLY or STOP
            path.closeSubpath()
            index += 1
    return path if not path.isEmpty() else None


def warm_up():
    """Load matplotlib and the handwriting font once, off the UI thread, so the first formula is instant."""

    def run():
        try:
            _formula_path("$x^2$", FORMULA_SIZE)
        except Exception:
            logger.debug("Formula warm-up failed", exc_info=True)

    threading.Thread(target=run, name="writing-warm-up", daemon=True).start()


def _text_path(text, size):
    font = QFont(HAND_FONT)
    font.setPixelSize(int(size))
    path = QPainterPath()
    path.setFillRule(Qt.WindingFill)
    path.addText(0, 0, font, text)
    return path


def build_writing(lines, kind="note"):
    """Lay out ``lines`` for the board; formula lines are larger than notes."""
    kind = kind if kind in KIND_LABELS else "note"
    layout = WritingLayout(kind=kind)
    for raw in lines:
        text = " ".join(str(raw).split())
        if not text:
            continue
        size = FORMULA_SIZE if kind == "formula" or "$" in text else NOTE_SIZE
        path = _formula_path(text, size) if text.count("$") >= 2 else None
        plain = text.replace("$", "")
        if path is None:
            if kind == "quote" and not plain.startswith(("“", '"')):
                plain = f"“{plain}”"
            path = _text_path(plain, size)
        bounds = path.boundingRect()
        path.translate(-bounds.left(), -bounds.top())
        layout.lines.append((path, bounds.width(), max(bounds.height(), size * 0.9)))
        layout.text_length += len(plain)
    return layout


def paint_writing(painter, layout, board, link_to, progress):
    """Draw the board at ``board`` with its lines written up to ``progress`` (0 -> 1).

    ``link_to`` is the element's rect the board belongs to, or None.
    """
    if not layout.lines:
        return
    painter.save()
    try:
        appear = max(0.0, min(1.0, progress * 6))
        if link_to is not None:
            start = _edge_point(board, link_to.center())
            end = _edge_point(link_to.adjusted(-6, -6, 6, 6), board.center())
            pen = _glow_pen(1.6, 200 * appear)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(start, end)

        shadow = QColor(0, 0, 0, int(90 * appear))
        painter.setPen(Qt.NoPen)
        painter.setBrush(shadow)
        painter.drawRoundedRect(board.translated(0, 5), 14, 14)
        fill = QColor(GLASS)
        fill.setAlphaF(fill.alphaF() * appear)
        painter.setBrush(fill)
        painter.setPen(_glow_pen(1.4, 210 * appear))
        painter.drawRoundedRect(board, 14, 14)

        painter.setFont(_font(11, QFont.Bold))
        tag = QColor(ACCENT)
        tag.setAlphaF(0.9 * appear)
        painter.setPen(tag)
        painter.drawText(
            QRectF(board.left() + PADDING_X, board.top() + 8, board.width(), 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            "✎ " + KIND_LABELS.get(layout.kind, "note").upper(),
        )

        # Reveal the lines one after another, in proportion to their width.
        total = sum(width for _, width, _ in layout.lines) or 1.0
        written = max(0.0, min(1.0, progress)) * total
        y = board.top() + PADDING_TOP
        nib = None
        for path, width, height in layout.lines:
            shown = max(0.0, min(width, written))
            written -= width
            if shown > 0:
                origin = QPointF(board.left() + PADDING_X, y)
                painter.save()
                painter.setClipRect(QRectF(origin.x() - 2, y - 4, shown + 2, height + 10))
                painter.translate(origin)
                painter.setPen(Qt.NoPen)
                painter.setBrush(INK)
                painter.drawPath(path)
                painter.restore()
                if shown < width:
                    nib = QPointF(origin.x() + shown, y + height * 0.62)
            y += height + LINE_GAP

        if nib is not None and progress < 1.0:
            painter.setPen(_glow_pen(10, 70))
            painter.drawPoint(nib)
            painter.setPen(Qt.NoPen)
            painter.setBrush(ACCENT)
            painter.drawEllipse(nib, 3.6, 3.6)
            trail = QPen(QColor(255, 255, 255, 120), 1.4)
            painter.setPen(trail)
            painter.drawLine(nib, nib + QPointF(-14, 3))
    finally:
        painter.restore()
