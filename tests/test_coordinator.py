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
from app.llm.provider import (
    LLMError,
    VisionLLMProvider,
    VisionLLMRequest,
    VisionLLMResponse,
)
from app.state import ApplicationState
from app.transcription.provider import TranscriptionProvider, TranscriptionResult

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
    def __init__(self, text="mock reply", error=None):
        self._text = text
        self._error = error
        self.requests = []

    def process(self, request: VisionLLMRequest) -> VisionLLMResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return VisionLLMResponse(text=self._text, model="stub-llm")


def build(tmp_path, recorder=None, transcription=None, capture=None, llm=None, pool=None, **kwargs):
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
    assert [response.text for response in responses] == ["mock reply"]
    assert errors == []


def test_response_is_available_on_the_coordinator(tmp_path):
    hotkey, _recorder, coordinator = build(tmp_path)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert coordinator.last_response is not None
    assert coordinator.last_response.text == "mock reply"


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


def test_coordinator_works_with_any_provider_implementation(tmp_path):
    """The coordinator only needs the abstraction, not a specific provider."""
    provider = StubLLM(text="a different answer")
    hotkey, _recorder, coordinator = build(tmp_path, llm=provider)
    responses = []
    coordinator.response_ready.connect(responses.append)

    hotkey.pressed.emit()
    hotkey.released.emit()

    assert [response.text for response in responses] == ["a different answer"]


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
