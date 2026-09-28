"""Temporary transcript bubble shown next to the cursor."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

GAP = 26
MAX_WIDTH = 420


class Caption(QWidget):
    """Frameless overlay that shows the transcript for a few seconds."""

    _STYLE = (
        "color: #ffffff;"
        "background-color: rgba(20, 20, 20, 235);"
        "border: 1px solid rgba(255, 255, 255, 60);"
        "border-radius: 12px;"
        "padding: 12px 18px;"
        "font-size: 15px;"
    )

    def __init__(self, display_ms):
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

        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(MAX_WIDTH)
        self._label.setStyleSheet(self._STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(display_ms)
        self._timer.timeout.connect(self.hide)

    def show_text(self, text):
        """Show ``text`` next to the cursor, hiding again automatically."""
        self._label.setText(text)
        self.adjustSize()
        self._place_near_cursor()
        self.show()
        self.raise_()
        self._timer.start()

    def _place_near_cursor(self):
        cursor = QCursor.pos()
        screen = QGuiApplication.screenAt(cursor) or QGuiApplication.primaryScreen()
        if screen is None:
            return

        area = screen.availableGeometry()
        x = cursor.x() + GAP
        y = cursor.y() + GAP

        if x + self.width() > area.right():
            x = cursor.x() - self.width() - GAP
        if y + self.height() > area.bottom():
            y = cursor.y() - self.height() - GAP

        x = min(max(x, area.left()), area.right() - self.width() + 1)
        y = min(max(y, area.top()), area.bottom() - self.height() + 1)
        self.move(x, y)
