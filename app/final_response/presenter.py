"""Presents the final response: speaks each sentence while its mark is drawn.

Order of events, after ``final_response.json`` is written:

    Responding (a short pause, 2 s by default)
    for each sentence: draw its mark -> Speaking -> speak it -> next sentence
    clear the screen -> finished

Speech runs on the thread pool, one sentence at a time, so the mark on screen
always belongs to the sentence being heard. Without a speech provider, or when
speaking fails, each mark stays up for about as long as reading the sentence
takes, so the drawings still follow the response.
"""

import json
import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal

from app import config
from app.final_response.provider import FinalResponseResult
from app.final_response.worker import SpeechWorker
from app.status import AssistantStatus

logger = logging.getLogger(__name__)

WORDS_PER_SECOND = 2.6
MIN_HOLD_SECONDS = 1.5
# How often, and for how long at most, to wait for a sentence's voice to be ready.
READY_POLL_MS = 100
MAX_READY_WAIT_MS = 10_000


def reading_ms(text):
    """Roughly how long saying ``text`` takes, in milliseconds."""
    words = len(str(text).split())
    return int(max(words / WORDS_PER_SECOND, MIN_HOLD_SECONDS) * 1000)


class ResponsePresenter(QObject):
    """Speaks the final response and draws each sentence's mark on the screen.

    Signals
        status_changed(AssistantStatus): Responding, then Speaking.
        finished(): the response is over and the screen is clear again.
    """

    status_changed = Signal(object)
    finished = Signal()

    def __init__(self, overlay=None, speech=None, thread_pool=None, responding_seconds=None, hold_ms=None, parent=None):
        super().__init__(parent)
        self._overlay = overlay
        self._speech = speech
        self._thread_pool = thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        self._responding_ms = int(
            1000 * (config.RESPONDING_PAUSE_SECONDS if responding_seconds is None else responding_seconds)
        )
        self._hold_ms = hold_ms or reading_ms
        self._sentences = []
        self._index = 0
        self._drawing = False
        self._workers = set()

    def present(self, response: FinalResponseResult):
        try:
            document = json.loads(Path(response.response_path).read_text(encoding="utf-8"))
            sentences = list(document["sentences"])
            size = document.get("screenshot_size") or {}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("[Presenting] could not read the final response: %s", exc)
            self.finished.emit()
            return

        self._sentences = sentences
        self._index = 0
        self._drawing = bool(
            self._overlay is not None
            and size.get("width")
            and size.get("height")
            and self._overlay.prepare((size["width"], size["height"]))
        )
        logger.info(
            "[Presenting] %s sentence(s); speech %s, marks %s",
            len(sentences),
            "on" if self._speech is not None else "off",
            "on" if self._drawing else "off",
        )

        self._prepare_speech(sentences)
        self.status_changed.emit(AssistantStatus.RESPONDING)
        self._after(self._responding_ms, self._next)

    def _prepare_speech(self, sentences):
        """Let the voice start making every sentence now, during the Responding pause."""
        prepare = getattr(self._speech, "prepare", None)
        if prepare is None:
            return
        try:
            prepare([sentence.get("text", "") for sentence in sentences])
        except Exception:
            logger.warning("[Presenting] could not prepare the speech ahead of time", exc_info=True)

    # --- one sentence at a time -----------------------------------------------

    def _next(self, waited_ms=0):
        if self._index >= len(self._sentences):
            self._done()
            return

        # Draw the mark only when its sentence can be heard straight away, so the two
        # start together; the pill keeps its current status while the voice gets ready.
        text = self._sentences[self._index].get("text", "")
        if waited_ms < MAX_READY_WAIT_MS and not self._speech_ready(text):
            self._after(READY_POLL_MS, lambda: self._next(waited_ms + READY_POLL_MS))
            return

        sentence = self._sentences[self._index]
        self._index += 1
        pointer = sentence.get("pointer") or {}
        logger.info(
            "[Presenting] %s/%s (%s): %s",
            self._index,
            len(self._sentences),
            sentence.get("visual_action", "none") if pointer else "no mark",
            sentence.get("text", ""),
        )
        if self._drawing:
            self._overlay.show_mark(sentence.get("visual_action"), pointer.get("box"), pointer.get("point"))
        self.status_changed.emit(AssistantStatus.SPEAKING)

        text = sentence.get("text", "")
        if self._speech is None:
            self._after(self._hold_ms(text), self._next)
            return

        worker = SpeechWorker(self._speech, text)
        worker.setAutoDelete(False)
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._on_spoken(worker))
        worker.signals.failed.connect(lambda message: self._on_speech_failed(worker, text, message))
        self._thread_pool.start(worker)

    def _speech_ready(self, text):
        is_ready = getattr(self._speech, "is_ready", None)
        if is_ready is None:
            return True
        try:
            return bool(is_ready(text))
        except Exception:
            return True

    def _on_spoken(self, worker):
        self._workers.discard(worker)
        self._next()

    def _on_speech_failed(self, worker, text, message):
        # The marks still follow the response: the sentence's mark stays up for its reading time.
        self._workers.discard(worker)
        logger.error("[Presenting] could not speak the sentence: %s", message)
        self._after(self._hold_ms(text), self._next)

    def _done(self):
        if self._overlay is not None:
            self._overlay.clear()
        self._sentences = []
        logger.info("[Presenting] done")
        self.finished.emit()

    @staticmethod
    def _after(ms, callback):
        if ms <= 0:
            callback()
        else:
            QTimer.singleShot(ms, callback)
