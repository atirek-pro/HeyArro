import wave

import pytest

from app.transcription import create_transcription_provider
from app.transcription.provider import (
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionResult,
)
from app.transcription.whisper_provider import WhisperTranscriptionProvider
from app.transcription.worker import TranscriptionWorker


class _StubProvider(TranscriptionProvider):
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def transcribe(self, audio_file):
        if self._error is not None:
            raise self._error
        return self._result


def test_result_defaults():
    result = TranscriptionResult(text="hello")
    assert result.text == "hello"
    assert result.duration is None
    assert result.language is None
    assert result.confidence is None


def test_factory_returns_a_transcription_provider():
    assert isinstance(create_transcription_provider(), TranscriptionProvider)


def test_missing_file_is_rejected(tmp_path):
    provider = WhisperTranscriptionProvider()
    with pytest.raises(TranscriptionError):
        provider.transcribe(tmp_path / "missing.wav")


def test_empty_recording_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)

    with pytest.raises(TranscriptionError):
        WhisperTranscriptionProvider().transcribe(path)


def test_invalid_audio_is_rejected(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"this is not audio")

    with pytest.raises(TranscriptionError):
        WhisperTranscriptionProvider().transcribe(path)


def test_worker_emits_finished():
    expected = TranscriptionResult(text="hello world")
    worker = TranscriptionWorker(_StubProvider(result=expected), "audio.wav")

    captured = []
    worker.signals.finished.connect(captured.append)
    worker.run()

    assert captured == [expected]


def test_worker_emits_failed_instead_of_raising():
    worker = TranscriptionWorker(_StubProvider(error=TranscriptionError("boom")), "audio.wav")

    captured = []
    worker.signals.failed.connect(captured.append)
    worker.run()

    assert captured == ["boom"]
