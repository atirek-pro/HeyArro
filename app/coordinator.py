"""Central coordinator: owns the application state machine and drives services.


The coordinator decides WHEN each service runs. The services own HOW they work,
so no microphone, Whisper, mss or model code lives here.
"""

import logging
from dataclasses import replace

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal

from app import config
from app.capture import save_png
from app.capture.worker import CaptureWorker
from app.llm import VisionLLMProvider, VisionLLMRequest
from app.llm.worker import LLMWorker, TeachingPlanWorker
from app.pipeline import ProcessingCoordinator
from app.state import ApplicationState, ApplicationStateMachine
from app.teaching import (
    TeachingPlanConversionError,
    TeachingSequenceService,
    VisualTeachingService,
    answer_only_teaching_plan,
    build_follow_up_context,
    detect_difficulty,
    detect_follow_up,
    difficulty_guidance,
    follow_up_difficulty,
    follow_up_guidance,
    response_to_teaching_plan,
)
from app.transcription.worker import TranscriptionWorker
from app.tts import TTSProvider
from app.visual_grounding.worker import GroundingWorker

logger = logging.getLogger(__name__)


class ApplicationCoordinator(QObject):
    """The single source of truth for the application's high-level state.

    Signals
        state_changed(previous, current): forwarded from the state machine.
        response_ready(HeyArroResponse): emitted when a structured response has been produced.
        error_occurred(str): a human-readable description of a failed step.

    The providers are whatever concrete implementations the composition root
    selected; this class never chooses a vendor or knows one. Every answer it
    receives is turned into a teaching plan, and that whole plan is grounded
    before the TeachingSequenceService starts, so the sequence only ever draws
    and speaks. The coordinator decides when preparation and execution start and
    what to do when they end, so it contains no speech, drawing or sequencing
    logic of its own.
    """

    state_changed = Signal(object, object)
    response_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        hotkey,
        recorder,
        transcription,
        capture,
        llm: VisionLLMProvider,
        tts: TTSProvider,
        teaching: VisualTeachingService,
        grounder=None,
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
        self._grounder = grounder

        self._thread_pool = thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        self._screenshot_dir = screenshot_dir
        self._responding_hold_ms = responding_hold_ms
        self._error_hold_ms = error_hold_ms

        # Voice and visuals are coordinated by the sequence service; the
        # coordinator only starts it and reacts to how it ends. Visual targets
        # are grounded beforehand, so the sequence only ever draws and speaks.
        self._sequence = TeachingSequenceService(tts, teaching, self._thread_pool)
        self._sequence.finished.connect(self._on_teaching_finished)
        self._sequence.failed.connect(self._fail)

        # Grounding runs off the UI thread, so a new interaction invalidates a
        # preparation that is still in flight.
        self._grounding_generation = 0
        self._screenshot = None

        # The lesson just taught, so the next request can continue it. This is
        # the whole of the teaching context: no history, no learner profile.
        self._last_plan = None
        self._last_question = ""

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
        return self._last_response

    @property
    def last_plan(self):
        """The lesson most recently taught, or None before the first one."""
        return self._last_plan

    @property
    def thread_pool(self):
        return self._thread_pool

    def shutdown(self):
        """Stop pending transitions and wait briefly for in-flight work."""
        self._cancel_pending_state()
        self._cancel_grounding()
        self._sequence.cancel()
        self._thread_pool.waitForDone(2000)

    # --- hotkey -------------------------------------------------------------

    def on_hotkey_pressed(self):
        self._cancel_pending_state()
        # A new question supersedes anything still being prepared or spoken.
        self._cancel_grounding()
        self._sequence.cancel()

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

    # --- combined result ----------------------------------------------------

    def _on_processing_complete(self, result):
        problem = result.transcript_error or result.capture_error
        if problem:
            self._fail(problem)
            return

        logger.info(
            "Processing complete: transcript=%r screenshot=%s",
            result.transcript,
            result.screenshot.path if result.screenshot else None,
        )

        if not self.state_machine.can_transition_to(ApplicationState.RESPONDING):
            logger.info("Ignoring processing result while in %s", self.state.value)
            return

        self._transition(ApplicationState.RESPONDING)
        # Everything visual refers to this screenshot, for the rest of the run.
        self._screenshot = result.screenshot
        self._sequence.prepare(result.screenshot)

        # A request that continues the lesson just taught is planned from that
        # lesson instead of answered from scratch.
        follow_up = self._detect_follow_up(result.transcript)
        if follow_up is not None:
            self._start_follow_up(result.transcript, follow_up)
            return

        difficulty = detect_difficulty(result.transcript)
        logger.info("[Teaching] Difficulty: %s", difficulty.value)
        self._last_question = (result.transcript or "").strip()

        screenshot = result.screenshot
        request = VisionLLMRequest(
            transcript=result.transcript,
            screenshot=screenshot.image if screenshot else None,
            # The model needs the screenshot's own pixel grid to place targets.
            screenshot_size=(screenshot.width, screenshot.height) if screenshot else None,
            guidance=difficulty_guidance(difficulty),
        )

        worker = self._track(LLMWorker(self._llm, request))
        worker.signals.finished.connect(
            lambda response: self._on_response(response, difficulty)
        )
        worker.signals.failed.connect(self._fail)
        logger.info("Requesting a response from %s", type(self._llm).__name__)
        self._thread_pool.start(worker)

    def _detect_follow_up(self, transcript):
        """Recognise a request that continues the lesson just taught."""
        follow_up = detect_follow_up(transcript, has_previous_plan=self._last_plan is not None)
        if follow_up is not None:
            logger.info("[Teaching] Follow-up detected: %s", follow_up.type.value)
        return follow_up

    def _start_follow_up(self, request_text, follow_up):
        """Plan a *new* lesson for a follow-up; the finished plan is never touched."""
        self._grounding_generation += 1
        generation = self._grounding_generation

        lesson = self._last_plan
        difficulty = follow_up_difficulty(lesson.difficulty, follow_up, request_text)
        logger.info("[Teaching] Difficulty: %s", difficulty.value)

        # Only the lesson being continued travels, and the screen is the current
        # one - a follow-up is about what is in front of the learner now.
        context = build_follow_up_context(self._last_question, lesson, follow_up)
        guidance = "\n".join(
            part for part in (difficulty_guidance(difficulty), follow_up_guidance(follow_up)) if part
        )
        screenshot = self._screenshot if follow_up.needs_screenshot else None
        if follow_up.needs_screenshot:
            logger.info("[Teaching] Follow-up grounding will use the current screen")

        worker = self._track(
            TeachingPlanWorker(
                self._llm,
                request_text,
                screenshot=screenshot.image if screenshot else None,
                screenshot_size=(
                    (screenshot.width, screenshot.height) if screenshot else None
                ),
                context=context,
                guidance=guidance,
            )
        )
        worker.signals.finished.connect(
            lambda plan: self._prepare(
                self._plan_at(plan, difficulty),
                generation,
                need_screenshot=follow_up.needs_screenshot,
            )
        )
        worker.signals.failed.connect(self._fail)
        self._thread_pool.start(worker)

    def _plan_at(self, plan, difficulty):
        """Return the generated plan, taught at the level that was asked for."""
        logger.info("[Teaching] Follow-up plan generated: %s step(s)", len(plan.steps))
        return plan.model_copy(update={"difficulty": difficulty})

    # --- response and teaching sequence -------------------------------------

    def _on_response(self, response, difficulty=None):
        self._last_response = response
        steps = response.teaching.steps
        logger.info(
            "Response [%s/%s, %s step(s), %s visual action(s)]: %s",
            response.teaching.mode.value,
            response.response.tone.value,
            len(steps),
            sum(len(step.visual_actions) for step in steps),
            response.response.text,
        )
        self.response_ready.emit(response)
        self._start_teaching(response, difficulty)

    # --- teaching plan: preparation, then execution -------------------------

    def _start_teaching(self, response, difficulty=None):
        """Turn the answer into a plan and prepare it for execution."""
        self._grounding_generation += 1
        generation = self._grounding_generation

        plan = self._build_plan(response, difficulty)
        if plan is None:
            self._fail("The answer could not be turned into a teaching plan")
            return

        self._prepare(plan, generation)

    def _prepare(self, plan, generation, need_screenshot=True):
        """Ground the whole plan, then hand it over for execution.

        This is preparation: it happens once, before any teaching step runs, so
        the sequence that follows only ever draws and speaks. A plan that needs
        no screen (a spoken recap, say) is not grounded at all.
        """
        if generation != self._grounding_generation:
            return

        if self._grounder is None or not need_screenshot or not self._has_screenshot():
            # Nothing to refine: the approximate targets are all we have.
            self._play(plan, generation)
            return

        worker = self._track(
            GroundingWorker(
                self._grounder,
                plan,
                self._screenshot.image,
                (self._screenshot.width, self._screenshot.height),
            )
        )
        worker.signals.finished.connect(lambda grounded: self._play(grounded, generation))
        worker.signals.failed.connect(
            lambda message: self._grounding_failed(message, generation)
        )
        logger.info("[Teaching] Grounding the plan before execution")
        self._thread_pool.start(worker)

    def _build_plan(self, response, difficulty=None):
        """Convert the answer into a teaching plan, or an answer-only plan."""
        try:
            plan = response_to_teaching_plan(response, difficulty)
        except TeachingPlanConversionError:
            # Nothing to teach step by step, so say the answer on its own.
            text = (getattr(response.response, "text", "") or "").strip()
            if not text:
                return None
            plan = answer_only_teaching_plan(text, difficulty)

        logger.info("[TeachingPlan] Plan ready: %s step(s)", len(plan.steps))
        return plan

    def _grounding_failed(self, message, generation):
        """A preparation that blew up must never start a partial lesson."""
        if generation != self._grounding_generation:
            return
        logger.error("Teaching-plan preparation failed: %s", message)
        self._fail(message)

    def _play(self, plan, generation):
        """Execute a prepared plan; nothing here refines anything.

        The plan becomes the lesson of record here rather than when it was
        planned: only something actually taught is worth continuing.
        """
        if generation != self._grounding_generation:
            return
        self._last_plan = plan
        self._sequence.play(plan)

    def _has_screenshot(self):
        return self._screenshot is not None and bool(getattr(self._screenshot, "image", None))

    def _cancel_grounding(self):
        """Invalidate a preparation that is still in flight."""
        self._grounding_generation += 1

    def _on_teaching_finished(self):
        """The sequence played to the end; let RESPONDING settle, then go idle."""
        self._schedule(ApplicationState.IDLE, self._responding_hold_ms)

    # --- state plumbing -----------------------------------------------------

    def _transition(self, state):
        return self.state_machine.transition_to(state)

    def _fail(self, message):
        logger.error("Interaction failed: %s", message)
        self.error_occurred.emit(message)
        # Ensure a half-finished teaching sequence leaves nothing on screen.
        self._sequence.cancel()
        self._cancel_grounding()

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
