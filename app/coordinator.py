"""Central coordinator: drives one interaction from the microphone to the refined screen.

The coordinator decides WHEN each service runs. The services own HOW they work,
so no microphone, Whisper, mss or model code lives here.

One interaction, end to end:

    hotkey pressed   -> record
    hotkey released  -> transcribe + capture the screen
    screenshot       -> refine it first: segment it into its own folder
    both ready       -> log the input, then categorise the question and score the
                        relevant segments (relevant_ss_manifest.json)
    segments chosen  -> plan how to respond (response_plan.json), except for a
                        fallback question
    plan written     -> locate each step's target on the screenshot
                        (plan_pointers.json)

There is no state machine, no output stage and no model answer: the interaction
ends once the response is planned, so everything the application did is in the
log.
"""

import logging
from dataclasses import replace

from PySide6.QtCore import QObject, QThreadPool, Signal

from app import config
from app.capture import save_png
from app.capture.worker import CaptureWorker
from app.grounding.worker import GroundingWorker
from app.pipeline import ProcessingCoordinator
from app.planner.worker import PlanningWorker
from app.refinement.provider import ResponseCategory
from app.refinement.worker import RefinementWorker, SelectionWorker
from app.transcription.worker import TranscriptionWorker

logger = logging.getLogger(__name__)


class ApplicationCoordinator(QObject):
    """Drives one interaction: record, transcribe, capture, refine, log.

    Signals
        error_occurred(str): a human-readable description of a failed step.

    The providers are whatever concrete implementations the composition root
    selected; this class never chooses a vendor or knows one. It owns the
    lifecycle only - recording, transcription, capture and refinement - and no
    prompt text, no state machine and no output stage.
    """

    error_occurred = Signal(str)

    def __init__(
        self,
        hotkey,
        recorder,
        transcription,
        capture,
        thread_pool=None,
        screenshot_dir=config.SCREENSHOT_DIR,
        refinement=None,
        refinement_dir=config.REFINEMENT_DIR,
        segment_selector=None,
        response_planner=None,
        plan_grounder=None,
        parent=None,
    ):
        super().__init__(parent)

        self._hotkey = hotkey
        self._recorder = recorder
        self._transcription = transcription
        self._capture = capture

        self._thread_pool = thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        self._screenshot_dir = screenshot_dir
        # Optional: without a refinement provider the screenshot is only saved.
        self._refinement = refinement
        self._refinement_dir = refinement_dir
        # Optional: scores which segments matter once the transcript is known.
        self._segment_selector = segment_selector
        # Optional: plans how to respond once the question's segments are selected.
        self._response_planner = response_planner
        # Optional: finds where each step of the plan points on the screenshot.
        self._plan_grounder = plan_grounder

        # One interaction at a time: a press is ignored while a recording or a
        # question is still in flight, so two runs can never interleave in the log.
        self._working = False

        self._pipeline = ProcessingCoordinator()
        self._pipeline.completed.connect(self._on_processing_complete)

        self._workers = set()

        hotkey.pressed.connect(self.on_hotkey_pressed)
        hotkey.released.connect(self.on_hotkey_released)
        recorder.finished.connect(self._on_recording_finished)
        recorder.failed.connect(self._on_service_error)

    # --- public API ---------------------------------------------------------

    @property
    def working(self):
        """True while one interaction is still being processed."""
        return self._working

    @property
    def thread_pool(self):
        return self._thread_pool

    def shutdown(self):
        """Wait briefly for in-flight work."""
        self._thread_pool.waitForDone(2000)

    # --- hotkey -------------------------------------------------------------

    def on_hotkey_pressed(self):
        if self._working or self._recorder.recording:
            logger.info("Ignoring hotkey press: an interaction is already running")
            return

        self._recorder.start()

    def on_hotkey_released(self):
        if not self._recorder.recording:
            logger.info("Ignoring hotkey release: nothing is being recorded")
            return

        # Stopping the recorder emits its result, which starts the rest.
        self._recorder.stop()

    # --- recording ----------------------------------------------------------

    def _on_recording_finished(self, path, metadata):
        logger.info("Recording available at %s", path)
        self._working = True
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
        """The first step on a received screenshot is refinement, when configured."""
        logger.info(
            "Screenshot received: monitor=%s %sx%s %s bytes",
            result.monitor_index,
            result.width,
            result.height,
            result.size_bytes,
        )
        if self._refinement is None:
            logger.info("Refinement is not configured; saving the screenshot as is")
            self._save_and_continue(result)
            return

        logger.info(
            "Refining screenshot with %s into %s", type(self._refinement).__name__, self._refinement_dir
        )
        worker = self._track(RefinementWorker(self._refinement, result, self._refinement_dir))
        worker.signals.finished.connect(lambda refined: self._on_refined(result, refined))
        worker.signals.failed.connect(lambda message: self._on_refinement_failed(result, message))
        self._thread_pool.start(worker)

    def _on_refined(self, result, refined):
        logger.info(
            "Refinement complete: %s segment(s), strategy=%s, folder=%s",
            refined.segment_count,
            refined.strategy,
            refined.folder,
        )
        self._pipeline.set_refinement(refined)
        self._continue_with(replace(result, path=refined.screenshot_path))

    def _on_refinement_failed(self, result, message):
        # Segments are an enrichment: a failed refinement must not lose the question.
        logger.error("Refinement failed, continuing with the plain screenshot: %s", message)
        self._save_and_continue(result)

    def _save_and_continue(self, result):
        try:
            path = save_png(result, self._screenshot_dir)
        except OSError as exc:
            logger.error("Could not save the screenshot: %s", exc)
            self._pipeline.set_capture_error(f"Could not save the screenshot: {exc}")
            return

        self._continue_with(replace(result, path=str(path)))

    def _continue_with(self, result):
        logger.info("Screenshot saved to %s", result.path)
        logger.info(
            "Screenshot details: monitor=%s width=%s height=%s timestamp=%s size_bytes=%s",
            result.monitor_index,
            result.width,
            result.height,
            round(result.timestamp, 3),
            result.size_bytes,
        )
        self._pipeline.set_screenshot(result)

    # --- input, then the relevant segments ----------------------------------

    def _on_processing_complete(self, result):
        problem = result.transcript_error or result.capture_error
        if problem:
            self._fail(problem)
            return

        self._log_input(result)

        refinement = result.refinement
        if self._segment_selector is not None and refinement is not None and refinement.segments:
            self._select_segments(result)
            return
        self._finish()

    def _log_input(self, result):
        screenshot = result.screenshot
        logger.info("[Input] transcript: %r", result.transcript)
        if screenshot is not None:
            logger.info(
                "[Input] screenshot: %sx%s, %s bytes, monitor %s, saved to %s",
                screenshot.width,
                screenshot.height,
                screenshot.size_bytes,
                screenshot.monitor_index,
                screenshot.path,
            )
        else:
            logger.info("[Input] screenshot: none (no screen was captured)")

    def _finish(self):
        self._working = False
        logger.info("Interaction complete")

    def _select_segments(self, result):
        """Categorise the question and score the segments against the transcript."""
        logger.info(
            "Selecting relevant segments with %s (%s segment(s))",
            type(self._segment_selector).__name__,
            result.refinement.segment_count,
        )
        worker = self._track(SelectionWorker(self._segment_selector, result.refinement, result.transcript))
        worker.signals.finished.connect(lambda selection: self._on_segments_selected(result, selection))
        worker.signals.failed.connect(lambda message: self._on_selection_failed(result, message))
        self._thread_pool.start(worker)

    def _on_segments_selected(self, result, selection):
        logger.info(
            "Segment selection complete: %s relevant, %.2fs, manifest=%s",
            selection.selected_count,
            selection.seconds,
            selection.manifest_path,
        )
        if self._response_planner is None:
            self._finish()
            return
        if selection.response_category is ResponseCategory.FALLBACK:
            logger.info("Question is a fallback; no response plan is needed")
            self._finish()
            return
        self._plan_response(selection)

    def _on_selection_failed(self, result, message):
        # Selection is an enrichment: a failure ends the interaction, it does not crash it.
        logger.error("Segment selection failed: %s", message)
        self._finish()

    # --- the response plan --------------------------------------------------

    def _plan_response(self, selection):
        """Plan how to respond to the question from the selected segments."""
        logger.info(
            "Planning a %s response with %s",
            selection.response_category.value,
            type(self._response_planner).__name__,
        )
        worker = self._track(PlanningWorker(self._response_planner, selection))
        worker.signals.finished.connect(self._on_planned)
        worker.signals.failed.connect(self._on_planning_failed)
        self._thread_pool.start(worker)

    def _on_planned(self, plan):
        logger.info("Response plan complete: %s step(s), %.2fs, plan=%s", plan.step_count, plan.seconds, plan.plan_path)
        if self._plan_grounder is None:
            self._finish()
            return
        self._ground_plan(plan)

    def _on_planning_failed(self, message):
        # Planning is a step of its own: a failure ends the interaction, it does not crash it.
        logger.error("Response planning failed: %s", message)
        self._finish()

    # --- the plan's coordinates ---------------------------------------------

    def _ground_plan(self, plan):
        """Find where each step of the plan points on the screenshot."""
        logger.info("Locating the plan's targets with %s", type(self._plan_grounder).__name__)
        worker = self._track(GroundingWorker(self._plan_grounder, plan))
        worker.signals.finished.connect(self._on_grounded)
        worker.signals.failed.connect(self._on_grounding_failed)
        self._thread_pool.start(worker)

    def _on_grounded(self, grounding):
        logger.info(
            "Plan grounding complete: %s of %s located, %.2fs, pointers=%s",
            grounding.located,
            grounding.total,
            grounding.seconds,
            grounding.pointers_path,
        )
        self._finish()

    def _on_grounding_failed(self, message):
        # Grounding is a step of its own: a failure ends the interaction, it does not crash it.
        logger.error("Plan grounding failed: %s", message)
        self._finish()

    # --- failures -----------------------------------------------------------

    def _fail(self, message):
        self._working = False
        logger.error("Interaction failed: %s", message)
        self.error_occurred.emit(message)

    def _track(self, worker):
        worker.setAutoDelete(False)
        self._workers.add(worker)
        worker.signals.finished.connect(lambda *_: self._workers.discard(worker))
        worker.signals.failed.connect(lambda *_: self._workers.discard(worker))
        return worker
