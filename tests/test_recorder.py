import wave

import pytest

from app.audio.recorder import (
    CHANNELS,
    MIN_DURATION_SECONDS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    RecorderError,
    finalize_recording,
)


def _pcm(seconds):
    return bytes(SAMPLE_WIDTH * CHANNELS * int(SAMPLE_RATE * seconds))


def test_empty_capture_is_rejected():
    with pytest.raises(RecorderError):
        finalize_recording([])


def test_very_short_capture_is_rejected():
    with pytest.raises(RecorderError):
        finalize_recording([_pcm(MIN_DURATION_SECONDS / 2)])


def test_valid_capture_writes_playable_wav():
    path, metadata = finalize_recording([_pcm(0.1) for _ in range(10)])

    try:
        with wave.open(str(path), "rb") as wav:
            assert wav.getnchannels() == CHANNELS
            assert wav.getsampwidth() == SAMPLE_WIDTH
            assert wav.getframerate() == SAMPLE_RATE
            assert wav.getnframes() == metadata["frames"]
            assert wav.readframes(wav.getnframes())

        assert metadata["duration_seconds"] == pytest.approx(1.0, abs=0.01)
        assert metadata["sample_rate"] == SAMPLE_RATE
        assert metadata["channels"] == CHANNELS
        assert metadata["size_bytes"] > 44
    finally:
        path.unlink(missing_ok=True)
