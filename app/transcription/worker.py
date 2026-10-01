"""Runs a transcription on a thread pool so the Qt UI thread stays responsive.

Transcribing can outlive the window it was started for, so emitting is guarded:
once Qt has torn the signals object down there is nobody left to tell.
"""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals emitted by a TranscriptionWorker."""

    finished = Signal(object)
    failed = Signal(str)


class TranscriptionWorker(QRunnable):
    """Executes ``provider.transcribe()`` off the UI thread."""

    def __init__(self, provider, audio_file):
        super().__init__()
        self.signals = WorkerSignals()
        self._provider = provider
        self._audio_file = audio_file

    def run(self):
        try:
            result = self._provider.transcribe(self._audio_file)
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(result))

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping a transcript: the application is shutting down")
