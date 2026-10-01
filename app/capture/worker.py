"""Runs a screen capture on a thread pool so the Qt UI thread stays responsive.

Capturing can outlive the window it was started for, so emitting is guarded: once
Qt has torn the signals object down there is nobody left to tell.
"""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class CaptureSignals(QObject):
    """Signals emitted by a CaptureWorker."""

    finished = Signal(object)
    failed = Signal(str)


class CaptureWorker(QRunnable):
    """Executes ``provider.capture_screen()`` off the UI thread."""

    def __init__(self, provider, monitor_index=None):
        super().__init__()
        self.signals = CaptureSignals()
        self._provider = provider
        self._monitor_index = monitor_index

    def run(self):
        try:
            result = self._provider.capture_screen(self._monitor_index)
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(result))

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping a screenshot: the application is shutting down")
