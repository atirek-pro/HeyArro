"""System tray icon and its menu."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

ICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "sunset-tile.ico"


def _draw_fallback_icon():
    """A plain green dot, used only when the icon file is missing."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(46, 204, 113))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(6, 6, 52, 52)
    painter.end()

    return QIcon(pixmap)


def app_icon():
    """The Hey Arro icon from app/assets, or the drawn dot if the file is missing."""
    if ICON_PATH.is_file():
        icon = QIcon(str(ICON_PATH))
        if not icon.isNull():
            return icon
    return _draw_fallback_icon()


class Tray(QSystemTrayIcon):
    """Tray icon exposing the Show Status and Quit actions."""

    def __init__(self, on_status, on_quit):
        super().__init__(app_icon())
        self.setToolTip("Hey Arro")

        menu = QMenu()

        status_action = QAction("Show Status", menu)
        status_action.triggered.connect(on_status)
        menu.addAction(status_action)

        menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(on_quit)
        menu.addAction(quit_action)

        self.setContextMenu(menu)
