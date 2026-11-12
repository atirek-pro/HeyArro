"""Runs response composing on a thread pool so the Qt UI thread stays responsive."""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger(__name__)


class ComposerSignals(QObject):
    """Signals emitted by a ComposerWorker."""

    finished = Signal(object)
    failed = Signal(str)


class ComposerWorker(QRunnable):
    """Executes ``composer.compose()`` off the UI thread."""

    def __init__(self, composer, grounding):
        super().__init__()
        self.signals = ComposerSignals()
        self._composer = composer
        self._grounding = grounding

    def run(self):
        try:
            result = self._composer.compose(self._grounding)
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(result))

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping a final response: the application is shutting down")


class SpeechSignals(QObject):
    """Signals emitted by a SpeechWorker."""

    finished = Signal()
    failed = Signal(str)


class SpeechWorker(QRunnable):
    """Speaks one sentence off the UI thread."""

    def __init__(self, speech, text):
        super().__init__()
        self.signals = SpeechSignals()
        self._speech = speech
        self._text = text

    def run(self):
        try:
            self._speech.speak(self._text)
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(self.signals.finished.emit)

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping a spoken sentence: the application is shutting down")
