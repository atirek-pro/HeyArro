"""System tray icon and its menu."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon


def _build_icon():
    """Draw a simple dot icon so no external asset file is needed."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(46, 204, 113))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(6, 6, 52, 52)
    painter.end()

    return QIcon(pixmap)


class Tray(QSystemTrayIcon):
    """Tray icon exposing the Show Status and Quit actions."""

    def __init__(self, on_status, on_quit):
        super().__init__(_build_icon())
        self.setToolTip("Clicky Assistant")

        menu = QMenu()

        status_action = QAction("Show Status", menu)
        status_action.triggered.connect(on_status)
        menu.addAction(status_action)

        menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(on_quit)
        menu.addAction(quit_action)

        self.setContextMenu(menu)
