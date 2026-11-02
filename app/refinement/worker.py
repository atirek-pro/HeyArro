"""Runs screenshot refinement on a thread pool so the Qt UI thread stays responsive.

Layout analysis can be slow and can outlive the window it was started for, so
emitting is guarded: once Qt has torn the signals object down there is nobody
left to tell.
"""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class RefinementSignals(QObject):
    """Signals emitted by a RefinementWorker."""

    finished = Signal(object)
    failed = Signal(str)


class RefinementWorker(QRunnable):
    """Executes ``provider.refine()`` off the UI thread."""

    def __init__(self, provider, capture, output_dir):
        super().__init__()
        self.signals = RefinementSignals()
        self._provider = provider
        self._capture = capture
        self._output_dir = output_dir

    def run(self):
        try:
            result = self._provider.refine(self._capture, self._output_dir)
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(result))

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping a refinement: the application is shutting down")


class SelectionSignals(QObject):
    """Signals emitted by a SelectionWorker."""

    finished = Signal(object)
    failed = Signal(str)


class SelectionWorker(QRunnable):
    """Executes ``selector.select()`` off the UI thread."""

    def __init__(self, selector, refinement, transcript):
        super().__init__()
        self.signals = SelectionSignals()
        self._selector = selector
        self._refinement = refinement
        self._transcript = transcript

    def run(self):
        try:
            result = self._selector.select(self._refinement, self._transcript)
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(result))

    def _safe_emit(self, emit):
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping a selection: the application is shutting down")
