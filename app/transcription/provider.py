"""Provider-neutral transcription contract.

The application only depends on these types, never on a specific engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class TranscriptionError(RuntimeError):
    """Raised when an audio file cannot be turned into text."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Outcome of a transcription request, independent of any engine."""

    text: str
    duration: float | None = None
    language: str | None = None
    confidence: float | None = None


class TranscriptionProvider(ABC):
    """Turns an audio file into text."""

    @abstractmethod
    def transcribe(self, audio_file) -> TranscriptionResult:
        """Transcribe ``audio_file`` and return a TranscriptionResult."""
