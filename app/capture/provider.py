"""Provider-neutral screen capture contract.

The application depends on these types, never on a specific capture library.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class ScreenCaptureError(RuntimeError):
    """Raised when the screen cannot be captured."""


@dataclass(frozen=True)
class MonitorInfo:
    """Geometry of one capturable monitor.

    Index 0 is the virtual screen covering every monitor; index 1 is the primary
    monitor and 2 and above are the additional monitors.
    """

    index: int
    left: int
    top: int
    width: int
    height: int
    is_primary: bool = False

    @property
    def resolution(self):
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class ScreenCaptureResult:
    """A captured screen image plus its metadata."""

    image: bytes
    monitor_index: int
    width: int
    height: int
    timestamp: float
    monitor: MonitorInfo | None = None
    path: str | None = None

    @property
    def size_bytes(self):
        return len(self.image)

    @property
    def resolution(self):
        return f"{self.width}x{self.height}"


def save_png(result, directory):
    """Write a capture to ``directory`` as PNG and return the new path."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"screenshot-{stamp}-{uuid.uuid4().hex[:6]}.png"
    path.write_bytes(result.image)
    return path


class ScreenCaptureProvider(ABC):
    """Captures the screen."""

    @abstractmethod
    def list_monitors(self) -> list[MonitorInfo]:
        """Return every capturable monitor, including the virtual screen at 0."""

    @abstractmethod
    def capture_screen(self, monitor_index=None) -> ScreenCaptureResult:
        """Capture a monitor and return the image and its metadata."""
