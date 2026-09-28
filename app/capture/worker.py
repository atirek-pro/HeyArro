"""Runs a screen capture on a thread pool so the Qt UI thread stays responsive."""

from PySide6.QtCore import QObject, QRunnable, Signal


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
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(result)
