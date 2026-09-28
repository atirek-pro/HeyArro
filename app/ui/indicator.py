"""Frameless, always-on-top indicator used to show the hotkey state."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class Indicator(QWidget):
    """Small non-activating overlay that displays the current hotkey state."""

    _BASE_STYLE = (
        "color: #ffffff;"
        "background-color: rgba(20, 20, 20, 225);"
        "border: 1px solid rgba(255, 255, 255, 60);"
        "border-radius: 12px;"
        "padding: 14px 30px;"
        "font-size: 20px;"
        "font-weight: bold;"
    )

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def show_listening(self):
        self._display("LISTENING...")

    def show_released(self):
        self._display("RELEASED")

    def _display(self, text):
        self._label.setStyleSheet(self._BASE_STYLE)
        self._label.setText(text)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()

    def _reposition(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        x = area.center().x() - self.width() // 2
        y = area.bottom() - self.height() - 80
        self.move(x, y)
