"""Runs speech on a thread pool so the Qt UI thread stays responsive."""

from PySide6.QtCore import QObject, QRunnable, Signal


class TTSSignals(QObject):
    """Signals emitted by a TTSWorker."""

    finished = Signal()
    failed = Signal(str)


class TTSWorker(QRunnable):
    """Executes ``provider.speak()`` off the UI thread."""

    def __init__(self, provider, text):
        super().__init__()
        self.signals = TTSSignals()
        self._provider = provider
        self._text = text

    def run(self):
        try:
            self._provider.speak(self._text)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit()
