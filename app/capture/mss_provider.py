"""Screen capture implemented with mss.

``mss`` is imported lazily so a missing or broken install becomes a handled
capture error instead of crashing the application at startup.
"""

import logging
import time

from app import config
from app.capture.provider import (
    MonitorInfo,
    ScreenCaptureError,
    ScreenCaptureProvider,
    ScreenCaptureResult,
)

logger = logging.getLogger(__name__)


def _import_mss():
    try:
        import mss
        import mss.tools
    except Exception as exc:
        raise ScreenCaptureError(f"mss is not available: {exc}") from exc
    return mss


def _monitor_info(index, monitor):
    return MonitorInfo(
        index=index,
        left=monitor["left"],
        top=monitor["top"],
        width=monitor["width"],
        height=monitor["height"],
        is_primary=(index == 1),
    )


class MssScreenCaptureProvider(ScreenCaptureProvider):
    """Captures monitors with mss.

    A fresh mss session is created per call, which keeps the provider safe to
    use from worker threads.
    """

    def __init__(self, monitor_index=None):
        self._monitor_index = config.SCREENSHOT_MONITOR if monitor_index is None else monitor_index

    @property
    def monitor_index(self):
        return self._monitor_index

    def list_monitors(self) -> list[MonitorInfo]:
        mss = _import_mss()
        try:
            with mss.MSS() as session:
                return [_monitor_info(index, monitor) for index, monitor in enumerate(session.monitors)]
        except ScreenCaptureError:
            raise
        except Exception as exc:
            raise ScreenCaptureError(f"Could not list monitors: {exc}") from exc

    def capture_screen(self, monitor_index=None) -> ScreenCaptureResult:
        index = self._monitor_index if monitor_index is None else monitor_index
        mss = _import_mss()

        try:
            with mss.MSS() as session:
                monitors = session.monitors
                if not 0 <= index < len(monitors):
                    raise ScreenCaptureError(
                        f"Monitor index {index} is out of range (0-{len(monitors) - 1})"
                    )

                shot = session.grab(monitors[index])
                image = mss.tools.to_png(shot.rgb, shot.size)
                info = _monitor_info(index, monitors[index])
                width, height = shot.width, shot.height
        except ScreenCaptureError:
            raise
        except Exception as exc:
            raise ScreenCaptureError(f"Screen capture failed: {exc}") from exc

        logger.info(
            "Captured monitor %s at (%s,%s) with resolution %sx%s",
            info.index,
            info.left,
            info.top,
            width,
            height,
        )

        return ScreenCaptureResult(
            image=image,
            monitor_index=index,
            width=width,
            height=height,
            timestamp=time.time(),
            monitor=info,
        )
