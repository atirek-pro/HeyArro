"""Local Windows text-to-speech built on pyttsx3 (the SAPI5 voice).

``pyttsx3`` is imported lazily, so a missing or broken install surfaces as a
handled TTSError on the first utterance instead of crashing the application at
startup. A fresh engine is created for each utterance because pyttsx3 engines
are not thread-safe and must be driven from the thread that created them - the
coordinator only ever calls speak() from a worker thread.
"""

import logging
import threading

from app import config
from app.tts.provider import TTSError, TTSProvider, normalize_text

logger = logging.getLogger(__name__)


def create_windows_engine():
    """Create a pyttsx3 engine, translating import/init failures into TTSError."""
    try:
        import pyttsx3
    except Exception as exc:
        raise TTSError(
            "pyttsx3 is not installed - install it to enable voice output "
            "(python -m pip install pyttsx3)"
        ) from exc

    try:
        return pyttsx3.init()
    except Exception as exc:
        raise TTSError(f"Could not initialize the Windows speech engine: {exc}") from exc


def _safe_stop(engine):
    if engine is None:
        return
    try:
        engine.stop()
    except Exception:
        logger.debug("Ignoring error while stopping speech", exc_info=True)


class WindowsTTSProvider(TTSProvider):
    """Speaks with the voice installed on Windows; no API key and no network."""

    name = "windows"

    def __init__(self, rate_wpm=None, volume=None, voice_id=None, engine_factory=None):
        self._rate = config.TTS_RATE_WPM if rate_wpm is None else rate_wpm
        self._volume = config.TTS_VOLUME if volume is None else volume
        self._voice_id = config.TTS_VOICE_ID if voice_id is None else voice_id
        self._engine_factory = engine_factory or create_windows_engine
        self._engine = None
        self._lock = threading.Lock()

    def speak(self, text):
        speakable = normalize_text(text)
        if not speakable:
            logger.debug("Nothing to speak")
            return

        engine = self._engine_factory()
        try:
            self._configure(engine)
            # Published so stop() can interrupt speech running on this thread.
            with self._lock:
                self._engine = engine
            engine.say(speakable)
            engine.runAndWait()
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Speech failed: {exc}") from exc
        finally:
            with self._lock:
                self._engine = None
            _safe_stop(engine)

    def stop(self):
        with self._lock:
            engine = self._engine
        if engine is None:
            return
        logger.info("Stopping active speech")
        _safe_stop(engine)

    def _configure(self, engine):
        """Apply the optional voice settings, ignoring engines that reject them."""
        try:
            if self._rate is not None:
                engine.setProperty("rate", self._rate)
            if self._volume is not None:
                engine.setProperty("volume", self._volume)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
        except Exception:
            logger.debug("Could not apply the configured TTS voice settings", exc_info=True)
