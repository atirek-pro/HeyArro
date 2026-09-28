"""Runs a transcription on a thread pool so the Qt UI thread stays responsive."""

from PySide6.QtCore import QObject, QRunnable, Signal


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
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(result)
