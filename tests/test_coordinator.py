"""One interaction, end to end: record, transcribe, capture, ask, show, idle.

The coordinator owns the lifecycle only. These tests drive it with doubles for
the microphone, the transcriber, the screen and the model, so the whole flow can
be asserted synchronously - and they pin down what the model is given (the
transcript, the screen and the depth instruction) and what happens to its answer
(it is handed to the UI, and then nothing else happens).
"""

import logging

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from app.capture.provider import ScreenCaptureProvider, ScreenCaptureResult
from app.coordinator import ApplicationCoordinator
from app.llm.mock_provider import MockVisionLLMProvider
from app.llm.provider import LLMError, VisionLLMProvider, VisionLLMRequest
from app.llm.response import HeyArroResponse, ResponseContent, ResponseTone
from app.state import ApplicationState
from app.transcription.provider import TranscriptionProvider, TranscriptionResult

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    yield QApplication.instance() or QApplication([])


# --- test doubles -----------------------------------------------------------


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

    def waitForDone(self, timeout=0):
        return True


class FakeHotkey(QObject):
    pressed = Signal()
    released = Signal()

    def start(self):
        return None

    def stop(self):
        return None


class StubRecorder(QObject):
    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(self, error=None):
        super().__init__()
        self.started = 0
        self.stopped = 0
        self.aborted = 0
        self._recording = False
        self._error = error

    @property
    def recording(self):
        return self._recording

    def start(self):
        self.started += 1
        self._recording = True

    def stop(self):
        self.stopped += 1
        self._recording = False
        if self._error is not None:
            self.failed.emit(self._error)
            return
        self.finished.emit("audio.wav", {"duration_seconds": 1.5})

    def abort(self):
        self.aborted += 1
        self._recording = False


class StubTranscription(TranscriptionProvider):
    def __init__(self, text="hello world", error=None):
        self._text = text
        self._error = error
        self.calls = 0

    def transcribe(self, audio_file):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return TranscriptionResult(text=self._text, duration=1.5, language="en", confidence=0.9)


class ScriptedTranscription(StubTranscription):
    """Returns each scripted utterance in turn, one per interaction."""

    def __init__(self, texts):
        super().__init__()
        self._texts = list(texts)

    def transcribe(self, audio_file):
        self.calls += 1
        index = min(self.calls - 1, len(self._texts) - 1)
        return TranscriptionResult(
            text=self._texts[index], duration=1.5, language="en", confidence=0.9
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
    """Records the request it was given and answers with fixed text."""

    def __init__(self, text="mock reply", error=None):
        self._text = text
        self._error = error
        self.requests = []

    def process(self, request: VisionLLMRequest) -> HeyArroResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return HeyArroResponse(
            response=ResponseContent(text=self._text, tone=ResponseTone.NEUTRAL)
        )


# --- harness ----------------------------------------------------------------


def build(
    tmp_path,
    recorder=None,
    transcription=None,
    capture=None,
    llm=None,
    pool=None,
    **kwargs,
):
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
        thread_pool=pool if pool is not None else SyncPool(),
        screenshot_dir=tmp_path,
        **options,
    )
    return hotkey, recorder, coordinator


def watch(coordinator):
    """Collect the states, the inputs, the answers and the errors."""
    seen = []
    inputs = []
    responses = []
    errors = []
    coordinator.state_changed.connect(lambda old, new: seen.append(new))
    coordinator.input_ready.connect(lambda text, shot: inputs.append((text, shot)))
    coordinator.response_ready.connect(responses.append)
    coordinator.error_occurred.connect(errors.append)
    return seen, inputs, responses, errors


def ask(hotkey):
    hotkey.pressed.emit()
    hotkey.released.emit()


# --- the happy path ---------------------------------------------------------


def test_starts_idle(tmp_path):
    _hotkey, _recorder, coordinator = build(tmp_path)

    assert coordinator.state is ApplicationState.IDLE
    assert coordinator.last_response is None


def test_press_starts_listening_and_recording(tmp_path):
    hotkey, recorder, coordinator = build(tmp_path)

    hotkey.pressed.emit()

    assert coordinator.state is ApplicationState.LISTENING
    assert recorder.started == 1
    assert recorder.recording is True


def test_release_moves_to_processing_and_stops_recording(tmp_path):
    hotkey, recorder, coordinator = build(tmp_path, pool=DeferredPool())

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert coordinator.state is ApplicationState.PROCESSING
    assert recorder.stopped == 1


def test_a_full_interaction_ends_idle(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path)
    seen, _inputs, _responses, _errors = watch(coordinator)

    ask(hotkey)

    assert seen == [
        ApplicationState.LISTENING,
        ApplicationState.PROCESSING,
        ApplicationState.RESPONDING,
        ApplicationState.IDLE,
    ]


def test_the_answer_is_available_on_the_coordinator(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path, llm=StubLLM(text="It is a window."))

    ask(hotkey)

    assert coordinator.last_response.response.text == "It is a window."


def test_the_transcript_and_the_screenshot_are_handed_to_the_ui(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, transcription=StubTranscription("hello world")
    )
    _seen, inputs, responses, _errors = watch(coordinator)

    ask(hotkey)

    assert len(inputs) == 1
    transcript, screenshot = inputs[0]
    assert transcript == "hello world"
    assert screenshot.image == PNG
    assert (screenshot.width, screenshot.height) == (4, 4)
    assert len(responses) == 1


def test_the_screenshot_is_saved_for_each_interaction(tmp_path):
    hotkey, _recorder, _coordinator = build(tmp_path)

    ask(hotkey)

    assert len(list(tmp_path.glob("*.png"))) == 1


def test_a_second_question_is_answered_again(tmp_path):
    """Nothing is left over from the first interaction."""
    llm = StubLLM()
    hotkey, _recorder, coordinator = build(
        tmp_path, llm=llm, transcription=ScriptedTranscription(["first", "second"])
    )

    ask(hotkey)
    ask(hotkey)

    assert [request.transcript for request in llm.requests] == ["first", "second"]
    assert coordinator.state is ApplicationState.IDLE


# --- what the model is given ------------------------------------------------


def test_the_transcript_and_the_screenshot_reach_the_provider(tmp_path):
    llm = StubLLM()
    hotkey, _recorder, _coordinator = build(
        tmp_path, llm=llm, transcription=StubTranscription("how do I export this?")
    )

    ask(hotkey)

    request = llm.requests[0]
    assert request.transcript == "how do I export this?"
    assert request.screenshot == PNG


def test_the_depth_asked_for_reaches_the_request(tmp_path):
    llm = StubLLM()
    hotkey, _recorder, _coordinator = build(
        tmp_path, llm=llm, transcription=StubTranscription("Explain this like I'm a beginner.")
    )

    ask(hotkey)

    assert "plain language" in llm.requests[0].guidance
    assert len(llm.requests) == 1  # decided locally, not with a second call


def test_an_ordinary_request_gets_the_neutral_depth(tmp_path):
    llm = StubLLM()
    hotkey, _recorder, _coordinator = build(
        tmp_path, llm=llm, transcription=StubTranscription("What is this?")
    )

    ask(hotkey)

    assert "normal technical level" in llm.requests[0].guidance


def test_the_input_is_logged_before_it_is_sent(tmp_path, caplog):
    hotkey, _recorder, _coordinator = build(
        tmp_path, transcription=StubTranscription("what is on my screen?")
    )

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    text = caplog.text
    assert "[Input] transcript: 'what is on my screen?'" in text
    assert "[Input] screenshot: 4x4, 8 bytes, monitor 1" in text
    assert "[Input] depth: intermediate" in text
    assert "[Input] guidance:" in text
    assert "[Input] sending to StubLLM" in text
    assert "[Output] mock reply" in text
    # The input is logged first: "sending" is the last thing before the call.
    assert text.index("[Input] transcript") < text.index("[Input] sending to")


def test_a_capture_failure_is_reported_not_hidden(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, capture=StubCapture(error=RuntimeError("capture exploded"))
    )
    _seen, _inputs, _responses, errors = watch(coordinator)

    ask(hotkey)

    assert errors == ["capture exploded"]


# --- failures ---------------------------------------------------------------


def test_microphone_failure_enters_error_then_idle(tmp_path):
    recorder = StubRecorder(error="microphone exploded")
    hotkey, _recorder, coordinator = build(tmp_path, recorder=recorder)
    seen, _inputs, _responses, errors = watch(coordinator)

    ask(hotkey)

    assert errors == ["microphone exploded"]
    assert seen[-2:] == [ApplicationState.ERROR, ApplicationState.IDLE]


def test_transcription_failure_enters_error_then_idle(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, transcription=StubTranscription(error=RuntimeError("whisper exploded"))
    )
    seen, _inputs, _responses, errors = watch(coordinator)

    ask(hotkey)

    assert errors == ["whisper exploded"]
    assert seen[-2:] == [ApplicationState.ERROR, ApplicationState.IDLE]


def test_model_failure_enters_error_then_idle(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, llm=StubLLM(error=LLMError("model exploded"))
    )
    seen, _inputs, responses, errors = watch(coordinator)

    ask(hotkey)

    assert errors == ["model exploded"]
    assert responses == []
    assert coordinator.last_response is None
    assert seen[-2:] == [ApplicationState.ERROR, ApplicationState.IDLE]


def test_release_without_press_is_ignored(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path)

    hotkey.released.emit()

    assert coordinator.state is ApplicationState.IDLE


def test_a_second_press_while_listening_does_not_restart_recording(tmp_path):
    hotkey, recorder, coordinator = build(tmp_path)

    hotkey.pressed.emit()
    hotkey.pressed.emit()

    assert coordinator.state is ApplicationState.LISTENING
    assert recorder.started == 1


def test_press_recovers_from_the_error_state(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, llm=StubLLM(error=LLMError("model exploded"))
    )

    ask(hotkey)
    assert coordinator.state is ApplicationState.IDLE

    hotkey.pressed.emit()

    assert coordinator.state is ApplicationState.LISTENING


def test_press_and_release_while_responding_are_ignored(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path, responding_hold_ms=5000)

    ask(hotkey)
    assert coordinator.state is ApplicationState.RESPONDING

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert coordinator.state is ApplicationState.RESPONDING


def test_an_ignored_press_is_not_reported_as_an_invalid_transition(tmp_path, caplog):
    hotkey, _recorder, _coordinator = build(tmp_path, responding_hold_ms=5000)

    ask(hotkey)
    with caplog.at_level(logging.ERROR):
        hotkey.pressed.emit()

    assert "Rejected invalid transition" not in caplog.text


# --- provider independence --------------------------------------------------


def test_the_coordinator_works_with_any_provider_implementation(tmp_path):
    """The mock provider is an offline stand-in for the real one."""
    llm = MockVisionLLMProvider(delay_seconds=0)
    hotkey, _recorder, coordinator = build(tmp_path, llm=llm)

    ask(hotkey)

    assert "[mock response]" in coordinator.last_response.response.text
    assert coordinator.state is ApplicationState.IDLE
