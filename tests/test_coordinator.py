import logging

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from app.capture.provider import (
    ScreenCaptureError,
    ScreenCaptureProvider,
    ScreenCaptureResult,
)
from app.coordinator import ApplicationCoordinator
from app.llm.mock_provider import MockVisionLLMProvider
from app.llm.provider import (
    LLMError,
    VisionLLMProvider,
    VisionLLMRequest,
)
from app.llm.response import (
    HeyArroResponse,
    ResponseContent,
    ResponseTone,
    TeachingMode,
    TeachingPlan,
    TeachingStep,
    VisualAction,
    VisualActionType,
    VisualTarget,
)
from app.state import ApplicationState
from app.teaching import TeachingDifficulty
from app.teaching.plan import TeachingPlan as LessonPlan
from app.teaching.plan import TeachingStep as LessonStep
from app.teaching.service import VisualTeachingService
from app.transcription.provider import TranscriptionProvider, TranscriptionResult
from app.tts.mock_provider import MockTTSProvider
from app.tts.provider import TTSError, TTSProvider
from app.visual_grounding import GroundingStatus, Refinement, VisualGrounder

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    yield QApplication.instance() or QApplication([])


class SyncPool:
    """Runs workers inline so the whole flow can be asserted synchronously."""

    def __init__(self):
        self.started = 0

    def start(self, runnable):
        self.started += 1
        runnable.run()

    def waitForDone(self, timeout=0):
        return True


class DeferredPool:
    """Queues workers without running them, so intermediate states can be seen."""

    def __init__(self):
        self.queue = []

    def start(self, runnable):
        self.queue.append(runnable)

    def run_next(self):
        assert self.queue, "no worker is waiting to run"
        self.queue.pop(0).run()

    def run_all(self):
        while self.queue:
            self.queue.pop(0).run()

    def waitForDone(self, timeout=0):
        return True


class FakeHotkey(QObject):
    pressed = Signal()
    released = Signal()


class StubRecorder(QObject):
    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(self, path="audio.wav", fail_on_start=False):
        super().__init__()
        self._path = path
        self._fail_on_start = fail_on_start
        self._recording = False
        self.starts = 0

    @property
    def recording(self):
        return self._recording

    def start(self):
        self.starts += 1
        if self._fail_on_start:
            self.failed.emit("microphone unavailable")
            return
        self._recording = True

    def stop(self):
        self._recording = False
        self.finished.emit(self._path, {"path": self._path})

    def abort(self):
        self._recording = False


class StubTranscription(TranscriptionProvider):
    def __init__(self, text="hello world", error=None):
        self._text = text
        self._error = error

    def transcribe(self, audio_file):
        if self._error is not None:
            raise self._error
        return TranscriptionResult(text=self._text, duration=1.5, language="en", confidence=0.9)


class ScriptedTranscription(TranscriptionProvider):
    """Returns each scripted utterance in turn, so a follow-up can be asked."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = 0

    def transcribe(self, audio_file):
        self.calls += 1
        index = min(self.calls - 1, len(self._texts) - 1)
        return TranscriptionResult(
            text=self._texts[index], duration=1.5, language="en", confidence=0.9
        )


class ChangingCapture(ScreenCaptureProvider):
    """Returns a different screenshot each time, so 'which screen' is answerable."""

    def __init__(self):
        self.calls = 0

    def list_monitors(self):
        return []

    def capture_screen(self, monitor_index=None):
        self.calls += 1
        return ScreenCaptureResult(
            image=PNG + bytes([self.calls]),
            monitor_index=1,
            width=4,
            height=4,
            timestamp=float(self.calls),
        )


class StubCapture(ScreenCaptureProvider):
    def __init__(self, error=None):
        self._error = error

    def list_monitors(self):
        return []

    def capture_screen(self, monitor_index=None):
        if self._error is not None:
            raise self._error
        return ScreenCaptureResult(
            image=PNG, monitor_index=1, width=4, height=4, timestamp=1234.5
        )


class StubLLM(VisionLLMProvider):
    def __init__(self, text="mock reply", error=None, steps=None, plan_error=None):
        self._text = text
        self._error = error
        self._steps = steps or []
        self._plan_error = plan_error
        self.requests = []
        self.plan_requests = []
        self.plans = []

    def process(self, request: VisionLLMRequest) -> HeyArroResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return HeyArroResponse(
            response=ResponseContent(text=self._text, tone=ResponseTone.NEUTRAL),
            teaching=TeachingPlan(
                mode=TeachingMode.GUIDED if self._steps else TeachingMode.DIRECT,
                steps=self._steps,
            ),
        )

    def generate_teaching_plan(
        self, user_query, screenshot=None, screenshot_size=None, context=None, guidance=None
    ):
        self.plan_requests.append(
            {
                "user_query": user_query,
                "screenshot": screenshot,
                "screenshot_size": screenshot_size,
                "context": context,
                "guidance": guidance,
            }
        )
        if self._plan_error is not None:
            raise self._plan_error
        plan = follow_up_lesson(user_query)
        self.plans.append(plan)
        return plan


class FailingTTS(TTSProvider):
    """A voice provider whose every utterance fails, for error-path tests."""

    name = "failing"

    def __init__(self, message="speech exploded"):
        self._message = message

    def speak(self, text):
        raise TTSError(self._message)

    def stop(self):
        return None


class RecordingTeachingService(VisualTeachingService):
    """Records what the sequence asked the overlay to show."""

    def __init__(self, error_on_show=False, events=None):
        self.prepared = []
        self.steps = []
        self.actions = []
        self.hide_calls = 0
        self._error_on_show = error_on_show
        self._events = events

    def prepare(self, capture):
        self.prepared.append(capture)

    def show_step(self, step):
        if self._events is not None:
            self._events.append(("show", getattr(step, "explanation", "")))
        if self._error_on_show:
            raise RuntimeError("overlay exploded")
        self.steps.append(step)

    def show_actions(self, actions):
        self.actions.append(list(actions))

    def hide(self):
        self.hide_calls += 1


def visual(kind, x, y, label=None, width=None, height=None):
    """Build one visual action for a step."""
    return VisualAction(
        type=VisualActionType(kind),
        target=VisualTarget(x=x, y=y, width=width, height=height, label=label or ""),
    )


def build(tmp_path, recorder=None, transcription=None, capture=None, llm=None, tts=None, teaching=None, grounder=None, pool=None, **kwargs):
    options = {"responding_hold_ms": 0, "error_hold_ms": 0}
    options.update(kwargs)

    hotkey = FakeHotkey()
    recorder = recorder if recorder is not None else StubRecorder()
    coordinator = ApplicationCoordinator(
        hotkey=hotkey,
        recorder=recorder,
        transcription=transcription or StubTranscription(),
        capture=capture or StubCapture(),
        llm=llm or StubLLM(),
        tts=tts if tts is not None else MockTTSProvider(),
        teaching=teaching if teaching is not None else RecordingTeachingService(),
        grounder=grounder,
        thread_pool=pool if pool is not None else SyncPool(),
        screenshot_dir=tmp_path,
        **options,
    )
    return hotkey, recorder, coordinator


def build_with_tts(tmp_path, tts=None, **kwargs):
    """Build a coordinator and hand back the voice provider for assertions."""
    tts = tts if tts is not None else MockTTSProvider()
    hotkey, recorder, coordinator = build(tmp_path, tts=tts, **kwargs)
    return hotkey, recorder, coordinator, tts


def build_services(tmp_path, tts=None, teaching=None, **kwargs):
    """Build a coordinator and hand back both output services for assertions."""
    tts = tts if tts is not None else MockTTSProvider()
    teaching = teaching if teaching is not None else RecordingTeachingService()
    hotkey, recorder, coordinator = build(tmp_path, tts=tts, teaching=teaching, **kwargs)
    return hotkey, recorder, coordinator, tts, teaching


def watch(coordinator):
    seen = []
    responses = []
    errors = []
    coordinator.state_changed.connect(lambda old, new: seen.append(new))
    coordinator.response_ready.connect(responses.append)
    coordinator.error_occurred.connect(errors.append)
    return seen, responses, errors


def test_starts_idle(tmp_path):
    _hotkey, _recorder, coordinator = build(tmp_path)
    assert coordinator.state is ApplicationState.IDLE


def test_hotkey_press_enters_listening_and_starts_recording(tmp_path):
    hotkey, recorder, coordinator = build(tmp_path)

    hotkey.pressed.emit()

    assert coordinator.state is ApplicationState.LISTENING
    assert recorder.recording is True
    assert recorder.starts == 1


def test_hotkey_release_enters_processing(tmp_path):
    pool = DeferredPool()
    hotkey, _recorder, coordinator = build(tmp_path, pool=pool)

    hotkey.pressed.emit()
    assert coordinator.state is ApplicationState.LISTENING

    hotkey.released.emit()
    assert coordinator.state is ApplicationState.PROCESSING

    pool.run_all()
    assert coordinator.state is ApplicationState.IDLE


def test_full_interaction_flow_ends_idle(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path)
    seen, responses, errors = watch(coordinator)

    hotkey.pressed.emit()
    assert coordinator.state is ApplicationState.LISTENING

    hotkey.released.emit()

    assert seen == [
        ApplicationState.LISTENING,
        ApplicationState.PROCESSING,
        ApplicationState.RESPONDING,
        ApplicationState.IDLE,
    ]
    assert coordinator.state is ApplicationState.IDLE
    assert [response.response.text for response in responses] == ["mock reply"]
    assert all(isinstance(response, HeyArroResponse) for response in responses)
    assert errors == []


def test_response_is_available_on_the_coordinator(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert coordinator.last_response is not None
    assert isinstance(coordinator.last_response, HeyArroResponse)
    assert coordinator.last_response.response.text == "mock reply"


def test_screenshot_is_saved_for_each_interaction(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path)

    hotkey.pressed.emit()
    hotkey.released.emit()

    saved = list(tmp_path.glob("screenshot-*.png"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == PNG


def test_responding_hold_keeps_the_state_visible(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path, responding_hold_ms=5000)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert coordinator.state is ApplicationState.RESPONDING


def test_release_without_press_is_ignored(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path)
    seen, _responses, errors = watch(coordinator)

    hotkey.released.emit()

    assert coordinator.state is ApplicationState.IDLE
    assert seen == []
    assert errors == []


def test_second_press_while_listening_does_not_restart_recording(tmp_path):
    hotkey, recorder, coordinator = build(tmp_path)

    hotkey.pressed.emit()
    hotkey.pressed.emit()

    assert coordinator.state is ApplicationState.LISTENING
    assert recorder.starts == 1


def test_microphone_failure_enters_error_then_idle(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, recorder=StubRecorder(fail_on_start=True)
    )
    seen, _responses, errors = watch(coordinator)

    hotkey.pressed.emit()

    assert seen == [ApplicationState.LISTENING, ApplicationState.ERROR, ApplicationState.IDLE]
    assert coordinator.state is ApplicationState.IDLE
    assert errors == ["microphone unavailable"]


def test_transcription_failure_enters_error_then_idle(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, transcription=StubTranscription(error=RuntimeError("whisper exploded"))
    )
    seen, _responses, errors = watch(coordinator)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert seen[-2:] == [ApplicationState.ERROR, ApplicationState.IDLE]
    assert coordinator.state is ApplicationState.IDLE
    assert errors == ["whisper exploded"]


def test_capture_failure_enters_error_then_idle(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, capture=StubCapture(error=ScreenCaptureError("no display"))
    )
    seen, _responses, errors = watch(coordinator)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert seen[-2:] == [ApplicationState.ERROR, ApplicationState.IDLE]
    assert coordinator.state is ApplicationState.IDLE
    assert errors == ["no display"]


def test_llm_failure_enters_error_then_idle(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, llm=StubLLM(error=LLMError("model exploded"))
    )
    seen, responses, errors = watch(coordinator)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert seen == [
        ApplicationState.LISTENING,
        ApplicationState.PROCESSING,
        ApplicationState.RESPONDING,
        ApplicationState.ERROR,
        ApplicationState.IDLE,
    ]
    assert coordinator.state is ApplicationState.IDLE
    assert responses == []
    assert errors == ["model exploded"]


def test_press_recovers_from_error_state(tmp_path):
    hotkey, recorder, coordinator = build(tmp_path)
    coordinator.state_machine.transition_to(ApplicationState.ERROR)

    hotkey.pressed.emit()

    assert coordinator.state is ApplicationState.LISTENING
    assert recorder.recording is True


def test_coordinator_sends_transcript_and_screenshot_to_the_provider(tmp_path):
    stub = StubLLM()
    hotkey, _recorder, coordinator = build(tmp_path, llm=stub)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert len(stub.requests) == 1
    request = stub.requests[0]
    assert isinstance(request, VisionLLMRequest)
    assert request.transcript == "hello world"
    assert request.screenshot == PNG
    # The screenshot's own pixel grid travels with the request so the model can
    # place visual targets in it.
    assert request.screenshot_size == (4, 4)


def test_coordinator_works_with_any_provider_implementation(tmp_path):
    """The coordinator only needs the abstraction, not a specific provider."""
    provider = StubLLM(text="a different answer")
    hotkey, _recorder, coordinator = build(tmp_path, llm=provider)
    responses = []
    coordinator.response_ready.connect(responses.append)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert [response.response.text for response in responses] == ["a different answer"]


def test_press_during_responding_is_ignored(tmp_path):
    hotkey, recorder, coordinator = build(tmp_path, responding_hold_ms=5000)

    hotkey.pressed.emit()
    hotkey.released.emit()
    assert coordinator.state is ApplicationState.RESPONDING

    hotkey.pressed.emit()

    assert coordinator.state is ApplicationState.RESPONDING
    assert recorder.starts == 1


def test_release_during_responding_is_ignored(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path, responding_hold_ms=5000)

    hotkey.pressed.emit()
    hotkey.released.emit()

    hotkey.released.emit()

    assert coordinator.state is ApplicationState.RESPONDING


def test_ignored_press_is_not_reported_as_an_invalid_transition(tmp_path, caplog):
    """The state machine's error log must stay meaningful, not noisy."""
    hotkey, _recorder, coordinator = build(tmp_path, responding_hold_ms=5000)
    hotkey.pressed.emit()
    hotkey.released.emit()

    with caplog.at_level(logging.ERROR):
        hotkey.pressed.emit()

    assert "Rejected invalid transition" not in caplog.text
    assert coordinator.state is ApplicationState.RESPONDING


# --- voice output -----------------------------------------------------------


def test_response_text_is_spoken(tmp_path):
    tts = MockTTSProvider()
    hotkey, _recorder, _coordinator, _tts = build_with_tts(
        tmp_path, tts=tts, llm=StubLLM(text="Open Settings and select Network.")
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert tts.spoken_texts == ["Open Settings and select Network."]


def test_the_introduction_and_then_each_step_are_spoken(tmp_path):
    steps = [
        TeachingStep(
            instruction="Open Settings.",
            visual_actions=[visual("box", 842, 316, "Settings")],
        )
    ]
    tts = MockTTSProvider()
    hotkey, _recorder, _coordinator, _tts = build_with_tts(
        tmp_path, tts=tts, llm=StubLLM(text="You can configure it in Settings.", steps=steps)
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    # The plan opens with its introduction, then teaches each step once - and
    # the answer is never spoken a second time as well.
    assert tts.spoken_texts == ["You can configure it in Settings.", "Open Settings."]


def test_tts_failure_enters_error_then_idle(tmp_path):
    hotkey, _recorder, coordinator, _tts = build_with_tts(
        tmp_path, tts=FailingTTS("speech exploded"), responding_hold_ms=5000
    )
    seen, _responses, errors = watch(coordinator)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert seen == [
        ApplicationState.LISTENING,
        ApplicationState.PROCESSING,
        ApplicationState.RESPONDING,
        ApplicationState.ERROR,
        ApplicationState.IDLE,
    ]
    assert coordinator.state is ApplicationState.IDLE
    assert errors == ["speech exploded"]


def test_tts_is_not_asked_to_speak_when_the_model_fails(tmp_path):
    tts = MockTTSProvider()
    hotkey, _recorder, _coordinator, _tts = build_with_tts(
        tmp_path, tts=tts, llm=StubLLM(error=LLMError("model exploded"))
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert tts.spoken_texts == []


def test_starting_a_new_interaction_stops_previous_speech(tmp_path):
    tts = MockTTSProvider()
    hotkey, _recorder, _coordinator, _tts = build_with_tts(
        tmp_path, tts=tts, responding_hold_ms=5000
    )

    hotkey.pressed.emit()

    assert tts.stop_calls == 1


def test_mock_llm_response_text_reaches_mock_tts(tmp_path):
    """End-to-end offline: mock vision model -> coordinator -> mock voice."""
    tts = MockTTSProvider()
    llm = MockVisionLLMProvider(delay_seconds=0)
    hotkey, _recorder, coordinator, _tts = build_with_tts(tmp_path, tts=tts, llm=llm)

    hotkey.pressed.emit()
    hotkey.released.emit()

    expected = coordinator.last_response.response.text
    assert "[mock response]" in expected
    assert tts.spoken_texts == [expected]


# --- visual teaching --------------------------------------------------------


def two_step_plan():
    return [
        TeachingStep(
            instruction="Click the Settings icon.",
            visual_actions=[visual("box", 200, 150, "Settings"), visual("point", 200, 150)],
        ),
        TeachingStep(
            instruction="Select Network.",
            visual_actions=[visual("box", 500, 350, "Network"), visual("point", 500, 350)],
        ),
    ]


def follow_up_lesson(query="Explain that again."):
    """The fresh lesson a follow-up produces."""
    return LessonPlan(
        objective="Understand the same idea again, more clearly.",
        difficulty=TeachingDifficulty.INTERMEDIATE,
        introduction="Here it is once more.",
        steps=[
            LessonStep(
                step_id="step_1",
                order=1,
                objective="See the same row again.",
                explanation="Look at the same row, this time from the start.",
                visual_actions=[visual("box", 300, 200, "Row")],
            ),
            LessonStep(
                step_id="step_2",
                order=2,
                objective="See why it matters.",
                explanation="That is why the value comes out where it does.",
            ),
        ],
    )


def test_each_teaching_step_is_shown_in_order(tmp_path):
    steps = two_step_plan()
    teaching = RecordingTeachingService()
    hotkey, _recorder, coordinator, _tts, _teaching = build_services(
        tmp_path, teaching=teaching, llm=StubLLM(text="Intro.", steps=steps)
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert [step.explanation for step in teaching.steps] == [
        "Click the Settings icon.",
        "Select Network.",
    ]
    assert coordinator.state is ApplicationState.IDLE


def test_teaching_steps_are_spoken_in_order(tmp_path):
    tts = MockTTSProvider()
    hotkey, _recorder, _coordinator, _tts, _teaching = build_services(
        tmp_path, tts=tts, llm=StubLLM(text="Intro.", steps=two_step_plan())
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert tts.spoken_texts == ["Intro.", "Click the Settings icon.", "Select Network."]


def test_step_without_visuals_is_still_spoken(tmp_path):
    steps = [TeachingStep(instruction="Your API key is missing.")]
    tts = MockTTSProvider()
    teaching = RecordingTeachingService()
    hotkey, _recorder, _coordinator, _tts, _teaching = build_services(
        tmp_path, tts=tts, teaching=teaching, llm=StubLLM(text="Heads up.", steps=steps)
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert tts.spoken_texts == ["Heads up.", "Your API key is missing."]
    # The step is passed on; the service has nothing to show for it.
    assert [step.visual_actions for step in teaching.steps] == [[]]


def test_duplicate_instruction_is_not_spoken_twice(tmp_path):
    step = TeachingStep(
        instruction="Open Settings.",
        visual_actions=[visual("box", 1, 2, "Settings")],
    )
    tts = MockTTSProvider()
    teaching = RecordingTeachingService()
    hotkey, _recorder, _coordinator, _tts, _teaching = build_services(
        tmp_path, tts=tts, teaching=teaching, llm=StubLLM(text="Open Settings.", steps=[step])
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    # The answer becomes the plan's introduction and the step says the same
    # thing, so it is still only heard once.
    assert tts.spoken_texts == ["Open Settings."]
    assert [step.explanation for step in teaching.steps] == ["Open Settings."]


def test_teaching_receives_the_captured_screenshot(tmp_path):
    teaching = RecordingTeachingService()
    hotkey, _recorder, _coordinator, _tts, _teaching = build_services(
        tmp_path, teaching=teaching, llm=StubLLM(text="Intro.", steps=two_step_plan())
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert len(teaching.prepared) == 1
    capture = teaching.prepared[0]
    assert capture is not None
    assert (capture.width, capture.height) == (4, 4)
    assert capture.monitor_index == 1


def test_overlay_is_hidden_when_teaching_finishes(tmp_path):
    teaching = RecordingTeachingService()
    hotkey, _recorder, coordinator, _tts, _teaching = build_services(
        tmp_path, teaching=teaching, llm=StubLLM(text="Intro.", steps=two_step_plan())
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert teaching.hide_calls >= 1
    assert coordinator.state is ApplicationState.IDLE


def test_no_teaching_steps_speaks_only_the_answer(tmp_path):
    tts = MockTTSProvider()
    teaching = RecordingTeachingService()
    hotkey, _recorder, coordinator, _tts, _teaching = build_services(
        tmp_path, tts=tts, teaching=teaching, llm=StubLLM(text="Just an answer.")
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert tts.spoken_texts == ["Just an answer."]
    # A direct answer is executed as a single step with nothing to show.
    assert [step.visual_actions for step in teaching.steps] == [[]]
    assert coordinator.state is ApplicationState.IDLE


def test_empty_response_text_is_not_spoken(tmp_path):
    tts = MockTTSProvider()
    hotkey, _recorder, coordinator, _tts, _teaching = build_services(
        tmp_path, tts=tts, llm=StubLLM(text="")
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert tts.spoken_texts == []
    assert coordinator.state is ApplicationState.IDLE


def test_overlay_failure_does_not_break_the_interaction(tmp_path):
    tts = MockTTSProvider()
    teaching = RecordingTeachingService(error_on_show=True)
    hotkey, _recorder, coordinator, _tts, _teaching = build_services(
        tmp_path, tts=tts, teaching=teaching, llm=StubLLM(text="Intro.", steps=two_step_plan())
    )
    seen, responses, errors = watch(coordinator)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert coordinator.state is ApplicationState.IDLE
    assert errors == []
    assert len(responses) == 1
    assert tts.spoken_texts == ["Intro.", "Click the Settings icon.", "Select Network."]
    assert teaching.hide_calls >= 1


def test_new_interaction_hides_the_overlay(tmp_path):
    teaching = RecordingTeachingService()
    hotkey, _recorder, _coordinator, _tts, _teaching = build_services(
        tmp_path, teaching=teaching, responding_hold_ms=5000
    )

    hotkey.pressed.emit()
    hotkey.released.emit()
    before = teaching.hide_calls

    hotkey.pressed.emit()

    assert teaching.hide_calls > before


def test_tts_failure_during_teaching_cleans_up(tmp_path):
    teaching = RecordingTeachingService()
    hotkey, _recorder, coordinator, _tts, _teaching = build_services(
        tmp_path,
        tts=FailingTTS("speech exploded"),
        teaching=teaching,
        llm=StubLLM(text="Intro.", steps=two_step_plan()),
        responding_hold_ms=5000,
    )
    seen, _responses, errors = watch(coordinator)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert errors == ["speech exploded"]
    assert seen[-2:] == [ApplicationState.ERROR, ApplicationState.IDLE]
    assert teaching.hide_calls >= 1


# --- visual grounding: preparation, not execution ---------------------------


class ShiftingGrounder(VisualGrounder):
    """Stands in for the refinement pass: moves every target by a fixed offset."""

    def __init__(self, events=None, shift=(40, 30), status=GroundingStatus.REFINED, confidence=0.95):
        self._events = events
        self._shift = shift
        self._status = status
        self._confidence = confidence
        self.calls = []

    def clear_cache(self):
        return None

    def refine_target(self, request):
        self.calls.append(request)
        if self._events is not None:
            self._events.append(("ground", request.description))
        return Refinement(
            request.approximate.shifted(*self._shift), self._status, self._confidence
        )


class ExplodingGrounder(VisualGrounder):
    """A grounder that blows up, to prove a failed preparation aborts the run."""

    def refine_target(self, request):
        raise RuntimeError("grounding exploded")


def grounded_plan():
    """Two steps: the first names one target twice, the second a new target."""
    return [
        TeachingStep(
            instruction="Click the Settings icon.",
            visual_actions=[
                visual("box", 200, 150, "Settings"),
                visual("highlight", 200, 150, "Settings"),
            ],
        ),
        TeachingStep(
            instruction="Select Network.",
            visual_actions=[visual("box", 500, 350, "Network")],
        ),
    ]


def test_every_target_is_grounded_before_the_first_step_runs(tmp_path):
    events = []
    grounder = ShiftingGrounder(events)
    teaching = RecordingTeachingService(events=events)
    hotkey, _recorder, _coordinator, _tts, _teaching = build_services(
        tmp_path,
        teaching=teaching,
        grounder=grounder,
        llm=StubLLM(text="Intro.", steps=grounded_plan()),
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    kinds = [event[0] for event in events]
    first_show = kinds.index("show")

    # All grounding happens up front, and never again once teaching starts.
    assert kinds[:2] == ["ground", "ground"]
    assert "ground" not in kinds[first_show:]
    # Three visual actions, but only two distinct targets.
    assert len(grounder.calls) == 2
    assert [event for event in events if event[0] == "show"] == [
        ("show", "Click the Settings icon."),
        ("show", "Select Network."),
    ]


def test_execution_receives_the_refined_targets(tmp_path):
    teaching = RecordingTeachingService()
    hotkey, _recorder, _coordinator, _tts, _teaching = build_services(
        tmp_path,
        teaching=teaching,
        grounder=ShiftingGrounder(),
        llm=StubLLM(text="Intro.", steps=grounded_plan()),
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    target = teaching.steps[0].visual_actions[0].target
    assert (target.x, target.y) == (240, 180)


def test_a_preparation_that_blows_up_never_starts_a_partial_lesson(tmp_path):
    teaching = RecordingTeachingService()
    tts = MockTTSProvider()
    hotkey, _recorder, coordinator, _tts, _teaching = build_services(
        tmp_path,
        tts=tts,
        teaching=teaching,
        grounder=ExplodingGrounder(),
        llm=StubLLM(text="Intro.", steps=grounded_plan()),
        responding_hold_ms=5000,
    )
    seen, _responses, errors = watch(coordinator)

    hotkey.pressed.emit()
    hotkey.released.emit()

    # Executing a half-prepared plan is worse than not executing it: the
    # interaction fails instead of teaching something unrefined.
    assert errors == ["grounding exploded"]
    assert seen[-2:] == [ApplicationState.ERROR, ApplicationState.IDLE]
    assert teaching.steps == []
    assert tts.spoken_texts == []


def test_a_new_interaction_discards_a_pending_preparation(tmp_path):
    pool = DeferredPool()
    teaching = RecordingTeachingService()
    hotkey, _recorder, _coordinator, _tts, _teaching = build_services(
        tmp_path,
        pool=pool,
        teaching=teaching,
        grounder=ShiftingGrounder(),
        llm=StubLLM(text="Intro.", steps=grounded_plan()),
    )

    hotkey.pressed.emit()
    hotkey.released.emit()

    pool.run_next()  # transcription
    pool.run_next()  # screen capture
    pool.run_next()  # the model

    hotkey.pressed.emit()  # a new question supersedes the pending preparation
    pool.run_next()  # the stale preparation lands

    assert teaching.steps == []


# --- how deeply to teach ----------------------------------------------------


def teaching_session(tmp_path, texts, **kwargs):
    """A coordinator that hears one scripted utterance per interaction."""
    kwargs.setdefault("llm", StubLLM(text="Intro.", steps=two_step_plan()))
    return build_services(tmp_path, transcription=ScriptedTranscription(texts), **kwargs)


def ask(hotkey, times=1):
    for _ in range(times):
        hotkey.pressed.emit()
        hotkey.released.emit()


def test_the_level_the_learner_asked_for_reaches_the_lesson(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    hotkey, _recorder, coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Explain this like I'm a beginner."], llm=llm
    )

    ask(hotkey)

    assert coordinator.last_plan.difficulty is TeachingDifficulty.BEGINNER
    # The depth reaches the prompt that writes the lesson, without an extra call.
    assert "plain language" in llm.requests[0].guidance
    assert len(llm.requests) == 1


def test_an_ordinary_request_is_taught_at_the_middle_level(tmp_path):
    hotkey, _recorder, coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["What is this?"]
    )

    ask(hotkey)

    assert coordinator.last_plan.difficulty is TeachingDifficulty.INTERMEDIATE


def test_a_technical_question_does_not_assume_a_technical_learner(tmp_path):
    hotkey, _recorder, coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Explain how the garbage collector's write barrier works."]
    )

    ask(hotkey)

    assert coordinator.last_plan.difficulty is TeachingDifficulty.INTERMEDIATE


# --- following a lesson up --------------------------------------------------


def test_a_follow_up_is_planned_as_a_new_lesson(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    teaching = RecordingTeachingService()
    tts = MockTTSProvider()
    hotkey, _recorder, coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Teach me this.", "Make it simpler."], llm=llm, teaching=teaching, tts=tts
    )

    ask(hotkey)
    finished_plan = coordinator.last_plan
    words_before = len(tts.spoken_texts)

    ask(hotkey)

    assert len(llm.plan_requests) == 1
    assert llm.plan_requests[0]["user_query"] == "Make it simpler."
    assert tts.spoken_texts[words_before:] == [
        "Here it is once more.",
        "Look at the same row, this time from the start.",
        "That is why the value comes out where it does.",
    ]
    assert [step.explanation for step in teaching.steps][-2:] == [
        "Look at the same row, this time from the start.",
        "That is why the value comes out where it does.",
    ]

    # The lesson that finished is exactly as it was: a follow-up never edits it.
    assert coordinator.last_plan is not finished_plan
    assert finished_plan.difficulty is TeachingDifficulty.INTERMEDIATE
    assert [step.explanation for step in finished_plan.steps] == [
        "Click the Settings icon.",
        "Select Network.",
    ]


def test_a_follow_up_that_names_a_level_is_taken_at_its_word(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    hotkey, _recorder, coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Give me the technical implementation details.", "Make it simpler."], llm=llm
    )

    ask(hotkey)
    assert coordinator.last_plan.difficulty is TeachingDifficulty.ADVANCED

    ask(hotkey)
    # "simpler" is an instruction about depth, so it wins over the old level.
    assert coordinator.last_plan.difficulty is TeachingDifficulty.BEGINNER


def test_a_gentle_request_for_something_easier_moves_one_level(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    hotkey, _recorder, coordinator, _tts, _teaching = teaching_session(
        tmp_path,
        ["Give me the technical implementation details.", "That was too technical, make it easier to follow."],
        llm=llm,
    )

    ask(hotkey)
    assert coordinator.last_plan.difficulty is TeachingDifficulty.ADVANCED

    ask(hotkey)
    # Nothing in the wording names a level, so it moves exactly one step.
    assert coordinator.last_plan.difficulty is TeachingDifficulty.INTERMEDIATE


def test_the_follow_up_carries_the_lesson_that_just_finished(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    hotkey, _recorder, _coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["How do I multiply matrices?", "I don't understand step 2."], llm=llm
    )

    ask(hotkey)
    ask(hotkey)

    context = llm.plan_requests[0]["context"]
    assert "How do I multiply matrices?" in context
    assert "Click the Settings icon." in context
    assert "asking about step 2" in context
    assert "step_2" in llm.plan_requests[0]["guidance"] or "that step" in llm.plan_requests[0]["guidance"]


def test_the_follow_up_plan_is_grounded_before_it_is_taught(tmp_path):
    events = []
    grounder = ShiftingGrounder(events)
    teaching = RecordingTeachingService(events=events)
    hotkey, _recorder, _coordinator, _tts, _teaching = teaching_session(
        tmp_path,
        ["Teach me this.", "Show me another example."],
        teaching=teaching,
        grounder=grounder,
    )

    ask(hotkey)
    ask(hotkey)

    shown = events.index(("show", "Look at the same row, this time from the start."))
    assert [event for event in events[:shown] if event[0] == "ground"]
    # Nothing is refined once the new lesson has started.
    assert [event for event in events[shown:] if event[0] == "ground"] == []


def test_a_follow_up_uses_the_screen_in_front_of_the_learner(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    capture = ChangingCapture()
    hotkey, _recorder, _coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Teach me this.", "Show me another example."], llm=llm, capture=capture
    )

    ask(hotkey)
    ask(hotkey)

    assert capture.calls == 2
    assert llm.requests[0].screenshot == PNG + b"\x01"
    # The follow-up is planned and grounded against the screen as it is now.
    assert llm.plan_requests[0]["screenshot"] == PNG + b"\x02"


def test_a_recap_is_planned_without_the_screen(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    grounder = ShiftingGrounder()
    hotkey, _recorder, _coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Teach me this.", "Summarize that."], llm=llm, grounder=grounder
    )

    ask(hotkey)
    refined_for_the_lesson = len(grounder.calls)

    ask(hotkey)

    assert llm.plan_requests[0]["screenshot"] is None
    assert llm.plan_requests[0]["screenshot_size"] is None
    # Nothing to show means nothing to refine.
    assert len(grounder.calls) == refined_for_the_lesson


def test_a_new_question_is_answered_from_scratch(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    hotkey, _recorder, _coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Teach me this.", "How do I export a file?"], llm=llm
    )

    ask(hotkey, times=2)

    assert llm.plan_requests == []
    assert len(llm.requests) == 2


def test_a_follow_up_that_cannot_be_planned_fails_cleanly(tmp_path):
    llm = StubLLM(
        text="Intro.", steps=two_step_plan(), plan_error=LLMError("planning exploded")
    )
    teaching = RecordingTeachingService()
    tts = MockTTSProvider()
    hotkey, _recorder, coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Teach me this.", "Go deeper."], llm=llm, teaching=teaching, tts=tts
    )
    seen, _responses, errors = watch(coordinator)

    ask(hotkey)
    words_before = len(tts.spoken_texts)
    steps_before = len(teaching.steps)

    ask(hotkey)

    assert errors == ["planning exploded"]
    assert seen[-2:] == [ApplicationState.ERROR, ApplicationState.IDLE]
    assert len(tts.spoken_texts) == words_before
    assert len(teaching.steps) == steps_before


# --- nothing happens on its own ---------------------------------------------


def test_finishing_a_lesson_does_not_start_another(tmp_path):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    teaching = RecordingTeachingService()
    tts = MockTTSProvider()
    hotkey, _recorder, coordinator, _tts, _teaching = teaching_session(
        tmp_path, ["Teach me this."], llm=llm, teaching=teaching, tts=tts
    )

    ask(hotkey)
    spoken = list(tts.spoken_texts)
    plan = coordinator.last_plan

    assert coordinator.state is ApplicationState.IDLE
    assert plan.difficulty is TeachingDifficulty.INTERMEDIATE
    # No new plan, no question, no changed difficulty, no second lesson.
    assert llm.plan_requests == []
    assert len(llm.requests) == 1
    assert len(teaching.steps) == 2
    assert tts.spoken_texts == spoken


def test_the_teaching_logs_say_what_happened(tmp_path, caplog):
    llm = StubLLM(text="Intro.", steps=two_step_plan())
    hotkey, _recorder, _coordinator, _tts, _teaching = teaching_session(
        tmp_path,
        ["Explain this simply.", "Make it simpler."],
        llm=llm,
        grounder=ShiftingGrounder(),
    )

    with caplog.at_level(logging.INFO):
        ask(hotkey, times=2)

    text = caplog.text
    assert "[Teaching] Difficulty: beginner" in text
    assert "[Teaching] Follow-up detected: simplify" in text
    assert "[Teaching] Follow-up plan generated: 2 step(s)" in text
    assert "[Teaching] Grounding the plan before execution" in text
