"""Speaking the final response with Kokoro-82M, a natural-sounding open-source voice.

Kokoro runs locally on the CPU through ONNX Runtime (``kokoro-onnx``): no key and
no network, except a one-time download of the model when it is missing. The
model lives in its own folder (``tts_model/`` in the project by default):

    tts_model/kokoro-v1.0.onnx    the voice model (~310 MB)
    tts_model/voices-v1.0.bin     every Kokoro voice (~27 MB)

If either file is missing it is downloaded into that folder first, from the
official kokoro-onnx release, and then used from there.

On this kind of CPU Kokoro needs about as long to make a sentence as the sentence
lasts, so sentences are made ahead of time: ``prepare(texts)`` queues the whole
answer as soon as it is known, and each sentence is made while the one before it
is playing. The model is loaded in the background when the provider is created,
so the first answer does not wait for it.

Voices (female): American ``af_heart``, ``af_bella``; British ``bf_emma``,
``bf_isabella``; Indian ``hf_alpha``, ``hf_beta`` (Hindi voices reading English).
"""

import logging
import threading
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.final_response.speech import SpeechError, SpeechProvider

logger = logging.getLogger(__name__)

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
DOWNLOAD_CHUNK = 1024 * 1024
# Seconds to wait for the server to answer or send the next chunk.
DOWNLOAD_TIMEOUT_SECONDS = 60

# The first letter of a voice id names its accent; Indian (h) voices read English text.
LANGUAGE_BY_PREFIX = {"a": "en-us", "b": "en-gb", "h": "en-us"}


def language_for(voice, override=None):
    """The phonemizer language for ``voice``: ``override`` when set, else from its prefix."""
    if override:
        return override
    return LANGUAGE_BY_PREFIX.get(str(voice)[:1].lower(), "en-us")


def missing_model_files(model_dir):
    """The model files not yet in ``model_dir`` (empty when nothing needs downloading)."""
    folder = Path(model_dir)
    return [
        name
        for name in (MODEL_FILE, VOICES_FILE)
        if not (folder / name).is_file() or (folder / name).stat().st_size == 0
    ]


def ensure_model_files(model_dir, opener=None):
    """Make sure the model and voices are in ``model_dir``, downloading what is missing.

    Returns (model_path, voices_path). Raises SpeechError when a download fails.
    """
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    opener = opener or _open_url
    for name in missing_model_files(folder):
        _download(RELEASE_URL + name, folder / name, opener)
    return folder / MODEL_FILE, folder / VOICES_FILE


def _open_url(url):
    return urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)


def _download(url, path, opener):
    partial = path.with_name(path.name + ".part")
    logger.info("[Speech] downloading %s into %s (one time only)...", path.name, path.parent)
    try:
        with opener(url) as response, open(partial, "wb") as out:
            total = _content_length(response)
            done, next_report = 0, 10
            while True:
                chunk = response.read(DOWNLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and done * 100 // total >= next_report:
                    logger.info("[Speech] %s: %d%% of %.0f MB", path.name, done * 100 // total, total / 1e6)
                    next_report = done * 100 // total // 10 * 10 + 10
        partial.replace(path)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise SpeechError(f"Could not download {path.name}: {exc}") from exc
    logger.info("[Speech] %s ready (%.0f MB)", path.name, path.stat().st_size / 1e6)


def _content_length(response):
    try:
        return int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _load_kokoro(model_path, voices_path):
    try:
        from kokoro_onnx import Kokoro
    except Exception as exc:
        raise SpeechError("kokoro-onnx is not installed (python -m pip install kokoro-onnx)") from exc
    try:
        return Kokoro(str(model_path), str(voices_path))
    except Exception as exc:
        raise SpeechError(f"Could not load the Kokoro model: {exc}") from exc


class _SoundDevicePlayer:
    """Plays samples on the default output device and waits until they finish."""

    def play(self, samples, sample_rate):
        import sounddevice as sd

        sd.play(samples, sample_rate)
        sd.wait()

    def stop(self):
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            logger.debug("Ignoring an error while stopping playback", exc_info=True)


class KokoroSpeechProvider(SpeechProvider):
    """Speaks with a Kokoro voice, making each sentence ahead of time."""

    name = "kokoro"

    def __init__(
        self,
        voice=None,
        speed=None,
        language=None,
        model_dir=None,
        loader=None,
        player=None,
        prepare_files=None,
        fallback=None,
    ):
        self._voice = voice or config.KOKORO_VOICE
        self._speed = speed if speed is not None else config.KOKORO_SPEED
        self._language = language_for(self._voice, language if language is not None else config.KOKORO_LANG)
        self._model_dir = Path(model_dir or config.KOKORO_MODEL_DIR)
        self._loader = loader or _load_kokoro
        self._player = player or _SoundDevicePlayer()
        self._prepare_files = prepare_files or ensure_model_files
        self._fallback_factory = fallback or _windows_voice
        self._fallback = None
        self.needs_download = bool(missing_model_files(self._model_dir)) if prepare_files is None else False

        # One worker makes every sentence, in order, after the model has loaded. Loading
        # (and the one-time download) starts right now, when Hey Arro starts.
        self._maker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")
        self._model = self._maker.submit(self._load)
        self._pending = {}
        self._lock = threading.Lock()

    @property
    def voice(self):
        return self._voice

    def is_loaded(self):
        """True once loading has finished, successfully or not."""
        return self._model.done()

    def wait_until_loaded(self, timeout=None):
        """Block until loading has finished; True when Kokoro is ready to speak."""
        try:
            self._model.result(timeout=timeout)
            return True
        except Exception:
            return False

    def load_error(self):
        """Why Kokoro could not load, or None (also None while still loading)."""
        if not self._model.done():
            return None
        return self._model.exception()

    def _failed(self):
        return self._model.done() and self._model.exception() is not None

    def _fallback_voice(self):
        if self._fallback is None:
            logger.warning("[Speech] Kokoro is unavailable (%s); using the Windows voice", self.load_error())
            self._fallback = self._fallback_factory()
        return self._fallback

    def _load(self):
        model_path, voices_path = self._prepare_files(self._model_dir)
        kokoro = self._loader(model_path, voices_path)
        # The first sentence a fresh model makes is slow; make a throwaway one now.
        try:
            kokoro.create("Ready.", voice=self._voice, speed=self._speed, lang=self._language)
        except Exception:
            logger.debug("Kokoro warm-up failed; the first sentence will be slower", exc_info=True)
        logger.info("[Speech] Kokoro ready: voice %s (%s), speed %s", self._voice, self._language, self._speed)
        return kokoro

    def _make(self, text):
        kokoro = self._model.result()
        try:
            return kokoro.create(text, voice=self._voice, speed=self._speed, lang=self._language)
        except Exception as exc:
            raise SpeechError(f"Kokoro could not make the sentence: {exc}") from exc

    def _queue(self, text):
        future = self._maker.submit(self._make, text)
        with self._lock:
            self._pending.setdefault(text, deque()).append(future)
        return future

    def prepare(self, texts):
        """Start making ``texts`` now, in order, so each is ready when it is spoken."""
        if self._failed():
            return
        for text in texts:
            text = " ".join(str(text).split())
            if text:
                self._queue(text)

    def is_ready(self, text):
        """True when ``text`` can start playing at once (made already, or never queued)."""
        text = " ".join(str(text).split())
        with self._lock:
            queued = self._pending.get(text)
            return not queued or queued[0].done()

    def speak(self, text):
        text = " ".join(str(text).split())
        if not text:
            return
        with self._lock:
            queued = self._pending.get(text)
            future = queued.popleft() if queued else None
            if queued is not None and not queued:
                del self._pending[text]
        if self._failed():
            self._fallback_voice().speak(text)
            return
        if future is None:
            future = self._maker.submit(self._make, text)
        try:
            samples, sample_rate = future.result()
            self._player.play(samples, sample_rate)
        except SpeechError:
            raise
        except Exception as exc:
            raise SpeechError(f"Speech failed: {exc}") from exc

    def stop(self):
        self._player.stop()
        if self._fallback is not None:
            self._fallback.stop()


def _windows_voice():
    from app.final_response.speech import WindowsSpeechProvider

    return WindowsSpeechProvider()
