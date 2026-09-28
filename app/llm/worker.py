"""Runs an LLM request on a thread pool so the Qt UI thread stays responsive."""

from PySide6.QtCore import QObject, QRunnable, Signal


class LLMSignals(QObject):
    """Signals emitted by an LLMWorker."""

    finished = Signal(object)
    failed = Signal(str)


class LLMWorker(QRunnable):
    """Executes ``provider.process()`` off the UI thread."""

    def __init__(self, provider, request):
        super().__init__()
        self.signals = LLMSignals()
        self._provider = provider
        self._request = request

    def run(self):
        try:
            response = self._provider.process(self._request)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(response)
