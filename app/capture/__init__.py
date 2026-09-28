"""Screen capture abstraction and provider selection."""

from app.capture.provider import (
    MonitorInfo,
    ScreenCaptureError,
    ScreenCaptureProvider,
    ScreenCaptureResult,
    save_png,
)

__all__ = [
    "MonitorInfo",
    "ScreenCaptureError",
    "ScreenCaptureProvider",
    "ScreenCaptureResult",
    "create_screen_capture_provider",
    "save_png",
]


def create_screen_capture_provider() -> ScreenCaptureProvider:
    """Build the configured provider.

    The application depends on the ScreenCaptureProvider interface, so the
    concrete library is chosen here and can be swapped without touching the app.
    """
    from app.capture.mss_provider import MssScreenCaptureProvider

    return MssScreenCaptureProvider()
