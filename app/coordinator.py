"""Central coordinator: owns the application state machine and drives services.

The coordinator decides WHEN each service runs. The services own HOW they work,
so no microphone, Whisper, mss or model code lives here.

One interaction, end to end:

    hotkey pressed   -> record
    hotkey released  -> transcribe + capture the screen
    both ready       -> log the input, then send it to the model
    answer           -> hand it to the UI, hold RESPONDING, go idle

Nothing happens after the answer is displayed: there is no voice, no overlay and
no second model call. What the model is given is logged in full (transcript,
screen size, depth instruction), so any interaction can be read back from the log.
"""

import logging
from dataclasses import replace

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal

from app import config
from app.capture import save_png
from app.capture.worker import CaptureWorker
from app.llm import VisionLLMProvider, VisionLLMRequest
from app.llm.worker import LLMWorker
from app.pipeline import ProcessingCoordinator
from app.state import ApplicationState, ApplicationStateMachine
from app.teaching import detect_difficulty, difficulty_guidance
from app.transcription.worker import TranscriptionWorker

logger = logging.getLogger(__name__)


class ApplicationCoordinator(QObject):
    """The single source of truth for the application's high-level state.

    Signals
        state_changed(previous, current): forwarded from the state machine.
        input_ready(str, object): the transcript and the screenshot of one
            interaction, once both are available.
        response_ready(HeyArroResponse): the model's answer.
        error_occurred(str): a human-readable description of a failed step.

    The providers are whatever concrete implementations the composition root
    selected; this class never chooses a vendor or knows one. It owns the
    lifecycle only - recording, transcription, capture, the model call and the
    state machine - and no prompt text, speech or drawing.
    """

    state_changed = Signal(object, object)
    input_ready = Signal(str, object)
    response_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        hotkey,
        recorder,
        transcription,
        capture,
        llm: VisionLLMProvider,
        thread_pool=None,
        screenshot_dir=config.SCREENSHOT_DIR,
        responding_hold_ms=config.RESPONDING_HOLD_MS,
        error_hold_ms=config.ERROR_HOLD_MS,
        parent=None,
    ):
        super().__init__(parent)

        self._hotkey = hotkey
        self._recorder = recorder
        self._transcription = transcription
        self._capture = capture
        self._llm = llm

        self._thread_pool = thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        self._screenshot_dir = screenshot_dir
        self._responding_hold_ms = responding_hold_ms
        self._error_hold_ms = error_hold_ms

        self.state_machine = ApplicationStateMachine()
        self.state_machine.state_changed.connect(self.state_changed)

        self._pipeline = ProcessingCoordinator()
        self._pipeline.completed.connect(self._on_processing_complete)

        self._workers = set()
        self._last_response = None
        self._pending_state = None

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._apply_pending_state)

        hotkey.pressed.connect(self.on_hotkey_pressed)
        hotkey.released.connect(self.on_hotkey_released)
        recorder.finished.connect(self._on_recording_finished)
        recorder.failed.connect(self._on_service_error)

    # --- public API ---------------------------------------------------------

    @property
    def state(self):
        return self.state_machine.state

    @property
    def last_response(self):
        """The most recent answer, or None before the first one."""
        return self._last_response

    @property
    def thread_pool(self):
        return self._thread_pool

    def shutdown(self):
        """Stop pending transitions and wait briefly for in-flight work."""
        self._cancel_pending_state()
        self._thread_pool.waitForDone(2000)

    # --- hotkey -------------------------------------------------------------

    def on_hotkey_pressed(self):
        self._cancel_pending_state()

        if self.state is ApplicationState.ERROR:
            # Let a new interaction recover from a previous failure.
            self._transition(ApplicationState.IDLE)

        # Ask before transitioning so an ignored press is not reported as an
        # invalid transition by the state machine.
        if not self.state_machine.can_transition_to(ApplicationState.LISTENING):
            logger.info("Ignoring hotkey press while in %s", self.state.value)
            return

        self._transition(ApplicationState.LISTENING)
        self._recorder.start()

    def on_hotkey_released(self):
        if not self.state_machine.can_transition_to(ApplicationState.PROCESSING):
            logger.info("Ignoring hotkey release while in %s", self.state.value)
            return

        # Transition first: stopping the recorder emits its results, which are
        # only valid once the application is in PROCESSING.
        self._transition(ApplicationState.PROCESSING)

        if self._recorder.recording:
            self._recorder.stop()
        else:
            self._fail("No audio was recorded")

    # --- recording ----------------------------------------------------------

    def _on_recording_finished(self, path, metadata):
        logger.info("Recording available at %s", path)
        self._pipeline.start(audio_path=path)
        self._start_transcription(path)
        self._start_capture()

    def _on_service_error(self, message):
        self._fail(message)

    # --- transcription ------------------------------------------------------

    def _start_transcription(self, path):
        worker = self._track(TranscriptionWorker(self._transcription, path))
        worker.signals.finished.connect(self._on_transcript)
        worker.signals.failed.connect(self._pipeline.set_transcript_error)
        logger.info("Transcribing %s", path)
        self._thread_pool.start(worker)

    def _on_transcript(self, result):
        logger.info(
            "Transcript details: duration=%s language=%s confidence=%s",
            result.duration,
            result.language,
            result.confidence,
        )
        self._pipeline.set_transcript(result.text)

    # --- screen capture -----------------------------------------------------

    def _start_capture(self):
        worker = self._track(CaptureWorker(self._capture))
        worker.signals.finished.connect(self._on_capture)
        worker.signals.failed.connect(self._pipeline.set_capture_error)
        logger.info("Capturing screen with %s", type(self._capture).__name__)
        self._thread_pool.start(worker)

    def _on_capture(self, result):
        try:
            path = save_png(result, self._screenshot_dir)
        except OSError as exc:
            logger.error("Could not save the screenshot: %s", exc)
            self._pipeline.set_capture_error(f"Could not save the screenshot: {exc}")
            return

        result = replace(result, path=str(path))
        logger.info("Screenshot saved to %s", path)
        logger.info(
            "Screenshot details: monitor=%s width=%s height=%s timestamp=%s size_bytes=%s",
            result.monitor_index,
            result.width,
            result.height,
            round(result.timestamp, 3),
            result.size_bytes,
        )
        self._pipeline.set_screenshot(result)

    # --- input, then the model ----------------------------------------------

    def _on_processing_complete(self, result):
        problem = result.transcript_error or result.capture_error
        if problem:
            self._fail(problem)
            return

        if not self.state_machine.can_transition_to(ApplicationState.RESPONDING):
            logger.info("Ignoring processing result while in %s", self.state.value)
            return

        self._transition(ApplicationState.RESPONDING)
        logger.info(
            "Input complete: transcript=%r screenshot=%s",
            result.transcript,
            result.screenshot.path if result.screenshot else None,
        )
        self.input_ready.emit(result.transcript or "", result.screenshot)
        self._ask_model(result)

    def _ask_model(self, result):
        """Log exactly what the model is given, then ask it."""
        screenshot = result.screenshot
        difficulty = detect_difficulty(result.transcript)
        request = VisionLLMRequest(
            transcript=result.transcript,
            screenshot=screenshot.image if screenshot else None,
            guidance=difficulty_guidance(difficulty),
        )

        logger.info("[Input] transcript: %r", request.transcript)
        if screenshot is not None:
            logger.info(
                "[Input] screenshot: %sx%s, %s bytes, monitor %s",
                screenshot.width,
                screenshot.height,
                screenshot.size_bytes,
                screenshot.monitor_index,
            )
        else:
            logger.info("[Input] screenshot: none (no screen was captured)")
        logger.info("[Input] depth: %s", difficulty.value)
        logger.info("[Input] guidance: %s", request.guidance)
        logger.info(
            "[Input] sending to %s (model=%s)",
            type(self._llm).__name__,
            getattr(self._llm, "model", "n/a"),
        )

        worker = self._track(LLMWorker(self._llm, request))
        worker.signals.finished.connect(self._on_response)
        worker.signals.failed.connect(self._fail)
        self._thread_pool.start(worker)

    def _on_response(self, response):
        self._last_response = response
        text = response.response.text or ""
        logger.info("Response [tone=%s, %s characters]", response.response.tone.value, len(text))
        logger.info("[Output] %s", text)

        self.response_ready.emit(response)
        self._schedule(ApplicationState.IDLE, self._responding_hold_ms)

    # --- state plumbing -----------------------------------------------------

    def _transition(self, state):
        return self.state_machine.transition_to(state)

    def _fail(self, message):
        logger.error("Interaction failed: %s", message)
        self.error_occurred.emit(message)

        if self.state is not ApplicationState.ERROR:
            self._transition(ApplicationState.ERROR)

        self._schedule(ApplicationState.IDLE, self._error_hold_ms)

    def _schedule(self, state, delay_ms):
        if delay_ms <= 0:
            self._transition(state)
            return

        self._pending_state = state
        self._hold_timer.start(int(delay_ms))

    def _apply_pending_state(self):
        state, self._pending_state = self._pending_state, None
        if state is not None:
            self._transition(state)

    def _cancel_pending_state(self):
        self._hold_timer.stop()
        self._pending_state = None

    def _track(self, worker):
        worker.setAutoDelete(False)
        self._workers.add(worker)
        worker.signals.finished.connect(lambda *_: self._workers.discard(worker))
        worker.signals.failed.connect(lambda *_: self._workers.discard(worker))
        return worker
