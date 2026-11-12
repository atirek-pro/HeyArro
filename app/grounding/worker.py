"""Runs plan grounding on a thread pool so the Qt UI thread stays responsive."""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class GroundingSignals(QObject):
    """Signals emitted by a GroundingWorker."""

    finished = Signal(object)
    failed = Signal(str)


class GroundingWorker(QRunnable):
    """Executes ``grounder.ground()`` off the UI thread."""

    def __init__(self, grounder, plan):
        super().__init__()
        self.signals = GroundingSignals()
        self._grounder = grounder
        self._plan = plan

    def run(self):
        try:
            result = self._grounder.ground(self._plan)
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(result))

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping the pointers: the application is shutting down")
