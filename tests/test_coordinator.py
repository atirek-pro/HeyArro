"""One interaction, end to end: record, transcribe, capture, log.

The coordinator drives the input side only. These tests use doubles for the
microphone, the transcriber and the screen, so the whole flow can be asserted
synchronously - and they pin down what is logged, which is all that leaves the
application.
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


# --- harness ----------------------------------------------------------------


def build(tmp_path, recorder=None, transcription=None, capture=None, pool=None):
    hotkey = FakeHotkey()
    recorder = recorder if recorder is not None else StubRecorder()
    coordinator = ApplicationCoordinator(
        hotkey=hotkey,
        recorder=recorder,
        transcription=transcription or StubTranscription(),
        capture=capture or StubCapture(),
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
    transcription = StubTranscription()
    hotkey, recorder, _coordinator = build(tmp_path, transcription=transcription)
    original = transcription.transcribe

    def transcribe(audio_file):
        # While the question is being processed, another press arrives.
        hotkey.pressed.emit()
        return original(audio_file)

    transcription.transcribe = transcribe

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


# --- what is logged ---------------------------------------------------------


def test_the_screenshot_is_saved_for_each_interaction(tmp_path):
    hotkey, _recorder, _coordinator = build(tmp_path)

    ask(hotkey)

    assert len(list(tmp_path.glob("*.png"))) == 1


def test_the_input_is_logged_and_the_interaction_ends(tmp_path, caplog):
    hotkey, _recorder, coordinator = build(
        tmp_path, transcription=StubTranscription("what is on my screen?")
    )

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    text = caplog.text
    assert "[Input] transcript: 'what is on my screen?'" in text
    assert "[Input] screenshot: 4x4, 8 bytes, monitor 1" in text
    assert "Interaction complete" in text
    assert coordinator.working is False


def test_a_second_question_is_processed_after_the_first(tmp_path, caplog):
    hotkey, _recorder, coordinator = build(
        tmp_path, transcription=ScriptedTranscription(["first", "second"])
    )

    with caplog.at_level(logging.INFO):
        ask(hotkey)
        assert coordinator.working is False
        ask(hotkey)

    assert caplog.text.index("[Input] transcript: 'first'") < caplog.text.index("[Input] transcript: 'second'")
    assert caplog.text.count("Interaction complete") == 2


def test_no_model_is_asked_for_an_answer(tmp_path, caplog):
    """The flow ends with the input and its refinement: there is no answer stage."""
    hotkey, _recorder, _coordinator = build(tmp_path)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "[Output]" not in caplog.text
    assert "sending to" not in caplog.text


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


def test_the_app_is_ready_again_after_a_failure(tmp_path):
    hotkey, _recorder, coordinator = build(
        tmp_path, capture=StubCapture(error=RuntimeError("capture exploded"))
    )
    errors = watch(coordinator)

    ask(hotkey)
    ask(hotkey)

    assert errors == ["capture exploded", "capture exploded"]
    assert coordinator.working is False


# --- architecture -----------------------------------------------------------


def test_there_is_no_state_machine_and_no_output_stage():
    assert importlib.util.find_spec("app.state") is None
    assert importlib.util.find_spec("app.ui.caption") is None

    source = Path(coordinator_module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("applicationstate", "from app.state", "qwidget", "qpainter", "show_text"):
        assert forbidden not in source


def test_the_coordinator_does_not_know_a_model_provider():
    source = Path(coordinator_module.__file__).read_text(encoding="utf-8")

    assert "app.llm" not in source
    assert "VisionLLM" not in source
