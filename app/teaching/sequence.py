"""Synchronized voice + visual teaching.

A prepared plan is the unit of execution: speak the introduction, then for each
step render everything it asks the user to see, speak its explanation, and only
move on once that sentence has finished, then speak the conclusion. The visual is
therefore on screen before the first word of the sentence it belongs to, and
stays there until that sentence ends.

This service only executes a plan that has already been prepared (converted and
grounded). It never refines anything, never calls a model and never captures a
screen, so a step never waits on anything but speech.

The sequence service owns that loop, so the coordinator only has to start it and
react to its completion - neither of them knows how speech is synthesized or how
visuals are drawn.

Speech runs on the shared thread pool (the existing TTS worker) and completion
arrives as a Qt signal, so the UI thread never blocks while a sentence plays or
while we wait for it to finish.
"""

import logging

from PySide6.QtCore import QObject, QThreadPool, Signal

from app.teaching.service import VisualTeachingService
from app.tts import TTSProvider, normalize_text
from app.tts.worker import TTSWorker

logger = logging.getLogger(__name__)


def build_plan_utterances(plan):
    """Return the ordered ``(text, step)`` utterances to speak for ``plan``.

    The introduction comes first, then each step's explanation together with its
    own visual state, then the conclusion. A step is always yielded, even with a
    blank explanation, so its visuals are still shown; blank text and text that
    has already been spoken are skipped, so nothing is ever said twice.
    """
    utterances = []

    introduction = normalize_text(getattr(plan, "introduction", ""))
    if introduction:
        utterances.append((introduction, None))

    for step in plan.steps:
        utterances.append((normalize_text(getattr(step, "explanation", "")), step))

    conclusion = normalize_text(getattr(plan, "conclusion", "") or "")
    if conclusion:
        utterances.append((conclusion, None))

    return utterances


class TeachingSequenceService(QObject):
    """Plays a response's teaching steps, keeping the voice and pointer in step.

    Signals
        finished(): the sequence played to the end and the overlay was cleared.
        failed(str): speaking could not continue; the overlay was cleared.
    """

    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        tts: TTSProvider,
        teaching: VisualTeachingService,
        thread_pool=None,
        parent=None,
    ):
        super().__init__(parent)

        self._tts = tts
        self._teaching = teaching
        self._thread_pool = (
            thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        )

        # A sequence is guarded by a generation counter, so a cancelled or
        # superseded sequence ignores any late speech completion.
        self._generation = 0
        self._queue = []
        self._steps = []
        self._shown = 0
        self._spoken = set()
        self._active = False
        self._workers = set()

    # --- public API ---------------------------------------------------------

    @property
    def active(self):
        """True while a sequence is still being taught."""
        return self._active

    def prepare(self, capture):
        """Point the next steps at the screenshot their coordinates come from."""
        try:
            self._teaching.prepare(capture)
        except Exception:
            logger.exception("Could not prepare the visual teaching overlay")

    def play(self, plan):
        """Teach a prepared plan: introduction, then each step, then the conclusion.

        The plan must already be prepared - nothing here refines, requests or
        captures anything.
        """
        self.cancel()

        self._generation += 1
        generation = self._generation
        self._steps = list(plan.steps)
        self._shown = 0
        self._queue = build_plan_utterances(plan)
        self._spoken = set()
        self._active = True
        logger.info("[Teaching] Execution started (%s step(s))", len(self._steps))
        self._advance(generation)

    def cancel(self):
        """Stop speech, clear the overlay and abandon the running sequence."""
        self._generation += 1
        self._queue = []
        self._spoken = set()
        self._active = False
        self._stop_speaking()
        self._hide_teaching()

    # --- sequence -----------------------------------------------------------

    def _advance(self, generation):
        """Show and speak the next utterance, or finish when the queue is empty."""
        if generation != self._generation:
            return

        while self._queue:
            text, step = self._queue.pop(0)
            # Render first: the visual must be on screen before its sentence is
            # heard, and must stay there until that sentence has finished.
            if step is not None:
                self._shown += 1
                logger.info("[Teaching] Step %s/%s", self._shown, len(self._steps))
                self._show_step(step)
            if self._speak(text, generation):
                return  # wait for this sentence before moving to the next step

        self._active = False
        self._hide_teaching()
        logger.info("[Teaching] Execution completed")
        self.finished.emit()

    def _show_step(self, step):
        try:
            self._teaching.show_step(step)
        except Exception:
            logger.exception("Showing a teaching step failed; continuing without the overlay")
            self._hide_teaching()

    def _speak(self, text, generation):
        """Speak ``text`` unless it is blank or already spoken.

        Returns True when speech was started, so the caller knows whether to
        wait for it or move straight on to the next step.
        """
        speakable = normalize_text(text)
        if not speakable or speakable in self._spoken:
            return False

        self._spoken.add(speakable)
        worker = self._track(TTSWorker(self._tts, speakable))
        worker.signals.finished.connect(lambda: self._advance(generation))
        worker.signals.failed.connect(lambda message: self._on_failed(message, generation))
        logger.info("Speaking with %s: %s", type(self._tts).__name__, speakable)
        self._thread_pool.start(worker)
        return True

    def _on_failed(self, message, generation):
        if generation != self._generation:
            return

        self._active = False
        self._stop_speaking()
        self._hide_teaching()
        logger.error("Speaking a teaching step failed: %s", message)
        self.failed.emit(message)

    # --- helpers ------------------------------------------------------------

    def _stop_speaking(self):
        try:
            self._tts.stop()
        except Exception:
            logger.debug("Ignoring error while stopping speech", exc_info=True)

    def _hide_teaching(self):
        try:
            self._teaching.hide()
        except Exception:
            logger.debug("Ignoring error while hiding the teaching overlay", exc_info=True)

    def _track(self, worker):
        worker.setAutoDelete(False)
        self._workers.add(worker)
        worker.signals.finished.connect(lambda *_: self._workers.discard(worker))
        worker.signals.failed.connect(lambda *_: self._workers.discard(worker))
        return worker
