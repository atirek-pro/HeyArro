"""Transcription abstraction and provider selection."""

from app.transcription.provider import (
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionResult,
)

__all__ = [
    "TranscriptionError",
    "TranscriptionProvider",
    "TranscriptionResult",
    "create_transcription_provider",
]


def create_transcription_provider() -> TranscriptionProvider:
    """Build the configured provider.

    The application depends on the TranscriptionProvider interface, so the
    concrete engine is chosen here and can be swapped without touching the app.
    """
    from app.transcription.whisper_provider import WhisperTranscriptionProvider

    return WhisperTranscriptionProvider()
