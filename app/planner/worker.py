"""Runs response planning on a thread pool so the Qt UI thread stays responsive."""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class PlanningSignals(QObject):
    """Signals emitted by a PlanningWorker."""

    finished = Signal(object)
    failed = Signal(str)


class PlanningWorker(QRunnable):
    """Executes ``planner.plan()`` off the UI thread."""

    def __init__(self, planner, selection):
        super().__init__()
        self.signals = PlanningSignals()
        self._planner = planner
        self._selection = selection

    def run(self):
        try:
            result = self._planner.plan(self._selection)
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(result))

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping a plan: the application is shutting down")
