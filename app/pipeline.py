"""Combines the transcript and the screenshot produced by one interaction.

The audio, transcription and capture services stay independent of each other;
this coordinator is the only place that knows their results belong together.
"""

from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, Signal

from app.capture.provider import ScreenCaptureResult


@dataclass(frozen=True)
class ProcessingResult:
    """Everything one Ctrl + Alt interaction produced."""

    audio_path: str | None = None
    transcript: str | None = None
    screenshot: ScreenCaptureResult | None = None
    transcript_error: str | None = None
    capture_error: str | None = None

    @property
    def has_transcript(self):
        return bool(self.transcript)

    @property
    def has_screenshot(self):
        return self.screenshot is not None


class ProcessingCoordinator(QObject):
    """Waits for both the transcript and the screenshot, then emits once."""

    completed = Signal(object)

    _SIDES = ("transcript", "capture")

    def __init__(self):
        super().__init__()
        self._result = None
        self._pending = set()

    def start(self, audio_path=None):
        """Begin a new interaction."""
        self._result = ProcessingResult(audio_path=audio_path)
        self._pending = set(self._SIDES)

    def set_transcript(self, text):
        self._finish_side("transcript", transcript=text)

    def set_transcript_error(self, message):
        self._finish_side("transcript", transcript_error=message)

    def set_screenshot(self, capture):
        self._finish_side("capture", screenshot=capture)

    def set_capture_error(self, message):
        self._finish_side("capture", capture_error=message)

    def _finish_side(self, side, **changes):
        if self._result is None:
            # No interaction in progress, or it already completed.
            return

        self._result = replace(self._result, **changes)
        self._pending.discard(side)
        if self._pending:
            return

        result = self._result
        self._result = None
        self._pending = set()
        self.completed.emit(result)
