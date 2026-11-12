"""Speaking the final response aloud.

Two voices: Kokoro (``kokoro_speech.py``, natural, the default) and the voice
installed on Windows (below), used when Kokoro is not installed or is switched off.

``pyttsx3`` (the SAPI5 voice) is imported lazily, so a missing or broken install
surfaces as a handled SpeechError instead of crashing the application. A fresh
engine is created for every sentence because pyttsx3 engines are not
thread-safe: ``speak`` is always called from a worker thread.
"""

import logging
import threading
from abc import ABC, abstractmethod

from app import config

logger = logging.getLogger(__name__)


class SpeechError(RuntimeError):
    """Raised when a sentence cannot be spoken."""


class SpeechProvider(ABC):
    """Speaks one sentence at a time."""

    @abstractmethod
    def speak(self, text):
        """Speak ``text`` and return once it has been said."""

    def prepare(self, texts):
        """Called with every sentence of an answer before the first is spoken.

        Providers that need time to make speech start on it here; others ignore it.
        """

    def is_ready(self, text):
        """True when ``text`` can start playing without a wait."""
        return True

    def stop(self):
        """Interrupt the sentence being spoken, when supported."""


def _create_engine():
    try:
        import pyttsx3
    except Exception as exc:
        raise SpeechError("pyttsx3 is not installed (python -m pip install pyttsx3)") from exc
    try:
        return pyttsx3.init()
    except Exception as exc:
        raise SpeechError(f"Could not start the Windows speech engine: {exc}") from exc


class WindowsSpeechProvider(SpeechProvider):
    """Speaks with the Windows voice; no key and no network."""

    name = "windows"

    def __init__(self, rate_wpm=None, volume=None, voice_id=None, engine_factory=None):
        self._rate = config.TTS_RATE_WPM if rate_wpm is None else rate_wpm
        self._volume = config.TTS_VOLUME if volume is None else volume
        self._voice_id = config.TTS_VOICE_ID if voice_id is None else voice_id
        self._engine_factory = engine_factory or _create_engine
        self._engine = None
        self._lock = threading.Lock()

    def speak(self, text):
        text = " ".join(str(text).split())
        if not text:
            return

        engine = self._engine_factory()
        try:
            self._configure(engine)
            # Published so stop() can interrupt speech running on this thread.
            with self._lock:
                self._engine = engine
            engine.say(text)
            engine.runAndWait()
        except SpeechError:
            raise
        except Exception as exc:
            raise SpeechError(f"Speech failed: {exc}") from exc
        finally:
            with self._lock:
                self._engine = None
            _safe_stop(engine)

    def stop(self):
        with self._lock:
            engine = self._engine
        _safe_stop(engine)

    def _configure(self, engine):
        """Apply the optional voice settings; an engine that rejects one keeps its default."""
        try:
            if self._rate:
                engine.setProperty("rate", self._rate)
            if self._volume is not None:
                engine.setProperty("volume", self._volume)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
        except Exception:
            logger.debug("Could not apply the configured voice settings", exc_info=True)


def _safe_stop(engine):
    if engine is None:
        return
    try:
        engine.stop()
    except Exception:
        logger.debug("Ignoring an error while stopping speech", exc_info=True)


def create_speech_provider() -> SpeechProvider | None:
    """Build the speech provider, or None when speech is switched off.

    ``TTS_PROVIDER=kokoro`` (the default) uses the natural Kokoro voice; when
    kokoro-onnx is not installed it falls back to the Windows voice.
    """
    if not config.TTS_ENABLED:
        return None
    if config.TTS_PROVIDER == "kokoro":
        try:
            import kokoro_onnx  # noqa: F401
        except Exception:
            logger.warning(
                "[Speech] kokoro-onnx is not installed (python -m pip install kokoro-onnx); "
                "using the Windows voice"
            )
        else:
            from app.final_response.kokoro_speech import KokoroSpeechProvider

            return KokoroSpeechProvider()
    elif config.TTS_PROVIDER != "windows":
        logger.warning("[Speech] unknown TTS_PROVIDER=%r; using the Windows voice", config.TTS_PROVIDER)
    return WindowsSpeechProvider()
