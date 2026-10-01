"""One interaction, end to end: record, transcribe, capture, ask, log.

The coordinator drives the input side only. These tests use doubles for the
microphone, the transcriber, the screen and the model, so the whole flow can be
asserted synchronously - and they pin down what the model is given and what it
answers, which is all that leaves the application.
"""

import importlib.util
import logging
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

import app.coordinator as coordinator_module
from app.capture.provider import ScreenCaptureProvider, ScreenCaptureResult
from app.coordinator import ApplicationCoordinator
from app.llm.mock_provider import MockVisionLLMProvider
from app.llm.provider import LLMError, VisionLLMProvider, VisionLLMRequest
from app.llm.response import HeyArroResponse, ResponseContent, ResponseTone
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


def build(tmp_path, recorder=None, transcription=None, capture=None, llm=None, pool=None):
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
    )
    return hotkey, recorder, coordinator


class FakeHotkey(QObject):
    pressed = Signal()
    released = Signal()

    def start(self):
        return None

    def stop(self):
        return None


def watch(coordinator):
    errors = []
    coordinator.error_occurred.connect(errors.append)
    return errors


def ask(hotkey):
    hotkey.pressed.emit()
    hotkey.released.emit()


# --- the hotkey and the recording -------------------------------------------


def test_a_press_starts_recording(tmp_path):
    hotkey, recorder, coordinator = build(tmp_path)

    hotkey.pressed.emit()

    assert recorder.started == 1
    assert recorder.recording is True
    assert coordinator.working is False


def test_a_release_stops_recording_and_starts_the_rest(tmp_path):
    hotkey, recorder, _coordinator = build(tmp_path)

    ask(hotkey)

    assert recorder.stopped == 1
    assert recorder.recording is False


def test_a_press_while_recording_is_ignored(tmp_path, caplog):
    hotkey, recorder, _coordinator = build(tmp_path)

    with caplog.at_level(logging.INFO):
        hotkey.pressed.emit()
        hotkey.pressed.emit()

    assert recorder.started == 1
    assert "already running" in caplog.text


def test_a_press_while_the_question_is_in_flight_is_ignored(tmp_path, caplog):
    """One interaction at a time, so two runs can never interleave in the log."""
    seen = []
    llm = StubLLM()
    hotkey, recorder, coordinator = build(tmp_path, llm=llm)
    coordinator.error_occurred.connect(seen.append)
    original = llm.process

    def process(request):
        # While the model is being asked, another press arrives.
        hotkey.pressed.emit()
        return original(request)

    llm.process = process

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert recorder.started == 1
    assert "already running" in caplog.text


def test_a_release_without_a_press_is_ignored(tmp_path, caplog):
    hotkey, recorder, _coordinator = build(tmp_path)

    with caplog.at_level(logging.INFO):
        hotkey.released.emit()

    assert recorder.stopped == 0
    assert "nothing is being recorded" in caplog.text


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


def test_the_screenshot_is_saved_for_each_interaction(tmp_path):
    hotkey, _recorder, _coordinator = build(tmp_path)

    ask(hotkey)

    assert len(list(tmp_path.glob("*.png"))) == 1


def test_a_second_question_is_processed_after_the_first(tmp_path):
    llm = StubLLM()
    hotkey, _recorder, coordinator = build(
        tmp_path, llm=llm, transcription=ScriptedTranscription(["first", "second"])
    )

    ask(hotkey)
    assert coordinator.working is False

    ask(hotkey)

    assert [request.transcript for request in llm.requests] == ["first", "second"]


# --- what comes back out ----------------------------------------------------


def test_the_input_and_the_answer_are_logged(tmp_path, caplog):
    hotkey, _recorder, coordinator = build(
        tmp_path, llm=StubLLM(text="It is a window."),
        transcription=StubTranscription("what is on my screen?"),
    )

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    text = caplog.text
    assert "[Input] transcript: 'what is on my screen?'" in text
    assert "[Input] screenshot: 4x4, 8 bytes, monitor 1" in text
    assert "[Input] sending to StubLLM" in text
    assert "[Output] It is a window." in text
    assert coordinator.working is False


def test_the_answer_is_only_logged(tmp_path, caplog):
    """Nothing is retained after the answer: the log is the whole record."""
    hotkey, _recorder, coordinator = build(tmp_path, llm=StubLLM(text="the answer"))

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "[Output] the answer" in caplog.text
    assert not hasattr(coordinator, "last_response")
    assert coordinator.working is False


def test_the_coordinator_works_with_any_provider_implementation(tmp_path, caplog):
    """The mock provider is an offline stand-in for the real one."""
    llm = MockVisionLLMProvider(delay_seconds=0)
    hotkey, _recorder, _coordinator = build(tmp_path, llm=llm)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "[Output] [mock response]" in caplog.text


# --- failures ---------------------------------------------------------------


def test_a_microphone_failure_is_reported(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, recorder=StubRecorder(error="microphone exploded")
    )
    errors = watch(coordinator)

    ask(hotkey)

    assert errors == ["microphone exploded"]
    assert coordinator.working is False


def test_a_transcription_failure_is_reported(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, transcription=StubTranscription(error=RuntimeError("whisper exploded"))
    )
    errors = watch(coordinator)

    ask(hotkey)

    assert errors == ["whisper exploded"]
    assert coordinator.working is False


def test_a_capture_failure_is_reported(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, capture=StubCapture(error=RuntimeError("capture exploded"))
    )
    errors = watch(coordinator)

    ask(hotkey)

    assert errors == ["capture exploded"]
    assert coordinator.working is False


def test_a_model_failure_is_reported(tmp_path, caplog):
    hotkey, _recorder, coordinator = build(
        tmp_path, llm=StubLLM(error=LLMError("model exploded"))
    )
    errors = watch(coordinator)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert errors == ["model exploded"]
    assert coordinator.working is False
    assert "[Output]" not in caplog.text


def test_the_app_is_ready_again_after_a_failure(tmp_path):
    llm = StubLLM(error=LLMError("model exploded"))
    hotkey, _recorder, coordinator = build(tmp_path, llm=llm)

    ask(hotkey)
    ask(hotkey)

    assert len(llm.requests) == 2


# --- architecture -----------------------------------------------------------


def test_there_is_no_state_machine_and_no_output_stage():
    assert importlib.util.find_spec("app.state") is None
    assert importlib.util.find_spec("app.ui.caption") is None

    source = Path(coordinator_module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("applicationstate", "from app.state", "qwidget", "qpainter", "show_text"):
        assert forbidden not in source
