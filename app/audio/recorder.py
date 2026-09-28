"""Microphone capture to temporary WAV files.

Records mono 16 kHz 16-bit PCM using ``sounddevice`` (PortAudio) and writes a
standard WAV file that any audio player can open.
"""

import logging
import tempfile
import threading
import uuid
import wave
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (int16 PCM)
DTYPE = "int16"
MIN_DURATION_SECONDS = 0.3
RECORDINGS_DIR = Path(tempfile.gettempdir()) / "clicky-recordings"


class RecorderError(RuntimeError):
    """Raised when capture cannot start or produced no usable audio."""


def _safe_close(stream):
    try:
        stream.stop()
        stream.close()
    except Exception:  # pragma: no cover - depends on the host
        logger.debug("Ignoring error while closing the audio stream", exc_info=True)


def select_input_device():
    """Return the index of a usable microphone, or raise RecorderError."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RecorderError(f"Could not query audio devices: {exc}") from exc

    if not devices:
        raise RecorderError("No audio devices found on this system")

    try:
        index = int(sd.default.device[0])
    except (TypeError, ValueError, IndexError):
        index = -1

    if index < 0:
        for candidate, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                index = candidate
                break

    if index < 0:
        raise RecorderError("No microphone input device is available")

    if devices[index]["max_input_channels"] < 1:
        raise RecorderError(f"Default input device has no microphone input: {devices[index]['name']}")

    return index


def write_wav(frames):
    """Write raw PCM frames to a new WAV file and return its path."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RECORDINGS_DIR / f"recording-{stamp}-{uuid.uuid4().hex[:6]}.wav"

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(frames)

    return path


def finalize_recording(chunks):
    """Validate captured PCM and save it, returning ``(path, metadata)``."""
    frames = b"".join(chunks)

    if not frames:
        raise RecorderError("No audio was captured - the microphone returned no data")

    duration = len(frames) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
    if duration < MIN_DURATION_SECONDS:
        raise RecorderError(
            f"Recording too short ({duration:.2f}s) - hold Ctrl + Alt a bit longer"
        )

    path = write_wav(frames)
    metadata = {
        "path": str(path),
        "duration_seconds": round(duration, 3),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_width_bytes": SAMPLE_WIDTH,
        "frames": len(frames) // (CHANNELS * SAMPLE_WIDTH),
        "size_bytes": path.stat().st_size,
    }
    return path, metadata


class Recorder(QObject):
    """Emits ``finished(path, metadata)`` or ``failed(message)`` per capture."""

    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._stream = None
        self._chunks = []
        self._lock = threading.Lock()
        self._recording = False

    @property
    def recording(self):
        return self._recording

    def start(self):
        if self._recording:
            return

        stream = None
        stream_started = False
        try:
            device_index = select_input_device()
            with self._lock:
                self._chunks = []
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=device_index,
                callback=self._on_audio,
            )
            stream.start()
            stream_started = True
        except RecorderError as exc:
            if stream is not None:
                _safe_close(stream)
            self._fail(str(exc))
            return
        except Exception as exc:
            if stream is not None:
                _safe_close(stream)
            self._fail(
                "Could not start recording - check microphone availability and Windows "
                f"privacy permissions: {exc}"
            )
            return

        if stream_started:
            self._stream = stream
            self._recording = True
            logger.info(
                "Recording started (device=%s, %d Hz, %d channel, %s)",
                device_index,
                SAMPLE_RATE,
                CHANNELS,
                DTYPE,
            )

    def stop(self):
        """Stop capture and save the audio, emitting finished or failed."""
        if not self._recording:
            return

        stream, self._stream = self._stream, None
        self._recording = False
        if stream is not None:
            _safe_close(stream)

        with self._lock:
            chunks = self._chunks
            self._chunks = []

        try:
            path, metadata = finalize_recording(chunks)
        except RecorderError as exc:
            self._fail(str(exc))
            return
        except OSError as exc:
            self._fail(f"Could not save the recording to disk: {exc}")
            return

        logger.info("Recording saved to %s", path)
        logger.info("Audio metadata: %s", metadata)
        self.finished.emit(str(path), metadata)

    def abort(self):
        """Discard the current capture without saving (used on shutdown)."""
        stream, self._stream = self._stream, None
        self._recording = False
        if stream is not None:
            _safe_close(stream)
        with self._lock:
            self._chunks = []

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio stream status: %s", status)
        if indata is None:
            return
        with self._lock:
            self._chunks.append(bytes(indata))

    def _fail(self, message):
        logger.error("Recording failed: %s", message)
        self.failed.emit(message)
