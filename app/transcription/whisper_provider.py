"""Local Whisper transcription built on faster-whisper.

``faster_whisper`` is imported lazily so a missing or broken install surfaces as
a handled transcription error instead of crashing the application at startup.
"""

import logging
import math
import threading
import wave
from pathlib import Path

from app import config
from app.transcription.provider import (
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


def _validate_wav(path):
    """Reject missing, corrupt or frame-less audio before loading the model."""
    try:
        with wave.open(str(path), "rb") as wav:
            frame_count = wav.getnframes()
    except (wave.Error, EOFError) as exc:
        raise TranscriptionError(f"Invalid or unreadable audio file: {path} ({exc})") from exc
    except OSError as exc:
        raise TranscriptionError(f"Could not read audio file: {path} ({exc})") from exc

    if frame_count == 0:
        raise TranscriptionError("Recording contains no audio frames")


class WhisperTranscriptionProvider(TranscriptionProvider):
    """Transcribes audio with a locally loaded faster-whisper model."""

    def __init__(self, model_size=None, device=None, compute_type=None, language=None):
        self._model_size = model_size or config.WHISPER_MODEL_SIZE
        self._device = device or config.WHISPER_DEVICE
        self._compute_type = compute_type or config.WHISPER_COMPUTE_TYPE
        self._language = language if language is not None else config.WHISPER_LANGUAGE
        self._model = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def model_size(self):
        return self._model_size

    def transcribe(self, audio_file) -> TranscriptionResult:
        path = Path(audio_file)
        if not path.exists():
            raise TranscriptionError(f"Audio file not found: {path}")

        _validate_wav(path)
        model = self._load_model()

        try:
            # The model is shared, so serialise inference across worker threads.
            with self._inference_lock:
                segments, info = model.transcribe(str(path), language=self._language)
                pieces = []
                logprobs = []
                for segment in segments:
                    pieces.append(segment.text)
                    if segment.avg_logprob is not None:
                        logprobs.append(segment.avg_logprob)
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        text = " ".join(piece.strip() for piece in pieces).strip()
        if not text:
            raise TranscriptionError("No speech was detected in the recording")

        confidence = None
        if logprobs:
            confidence = round(math.exp(sum(logprobs) / len(logprobs)), 3)

        return TranscriptionResult(
            text=text,
            duration=getattr(info, "duration", None),
            language=getattr(info, "language", None),
            confidence=confidence,
        )

    def _load_model(self):
        """Load the Whisper model once, on first use."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise TranscriptionError(f"faster-whisper is not available: {exc}") from exc

            logger.info(
                "Loading Whisper model '%s' (device=%s, compute_type=%s)",
                self._model_size,
                self._device,
                self._compute_type,
            )
            try:
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
            except Exception as exc:
                raise TranscriptionError(
                    f"Could not load Whisper model '{self._model_size}': {exc}"
                ) from exc

            logger.info("Whisper model '%s' ready", self._model_size)

        return self._model
