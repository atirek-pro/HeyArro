"""Runs an LLM request on a thread pool so the Qt UI thread stays responsive.

A model call can outlive the window it was made for (a slow answer while the user
quits), so emitting is guarded: once Qt has torn the signals object down there is
nobody left to tell.
"""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


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
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(response))

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping an answer: the application is shutting down")
