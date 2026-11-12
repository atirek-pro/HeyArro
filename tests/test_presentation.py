"""Presenting the final response: statuses, speech, marks on the screen and the glass pill.

Speech, the overlay and the thread pool are replaced by doubles, so nothing is
spoken or shown while the tests run.
"""

import json
import logging

import pytest
from PySide6.QtCore import QObject, QPointF, QRectF, Signal
from PySide6.QtGui import QImage

from app.coordinator import ApplicationCoordinator
from app.final_response.overlay import ScreenOverlay, map_box, map_point, scale_for, target_screen
from app.final_response.presenter import ResponsePresenter, reading_ms
from app.final_response.provider import FinalResponseResult
from app.final_response.speech import SpeechError, WindowsSpeechProvider
from app.refinement import ResponseCategory
from app.status import AssistantStatus
from app.ui.status_indicator import StatusIndicator
from tests.test_coordinator import (
    FakeHotkey,
    StubCapture,
    StubRecorder,
    StubTranscription,
    SyncPool,
    ask,
)
from tests.test_final_response import StubComposer
from tests.test_plan_grounding import StubGrounder
from tests.test_refinement import qt_app  # noqa: F401
from tests.test_response_planner import CategorisingSelector, StubPlanner
from tests.test_segment_selection import SegmentedRefinement

SEARCH_BOX = {"left": 600, "top": 1000, "width": 300, "height": 50}


# --- doubles ----------------------------------------------------------------


class FakeOverlay:
    def __init__(self, ready=True):
        self._ready = ready
        self.prepared = []
        self.marks = []
        self.cleared = 0

    def prepare(self, size):
        self.prepared.append(size)
        return self._ready

    def show_mark(self, action, box, point):
        self.marks.append((action, box, point))

    def clear(self):
        self.cleared += 1


class FakeSpeech:
    def __init__(self, fail_on=()):
        self.spoken = []
        self._fail_on = set(fail_on)

    def speak(self, text):
        if text in self._fail_on:
            raise SpeechError("no voice")
        self.spoken.append(text)


def _sentence(index, text, step=1, action="point", box=SEARCH_BOX):
    pointer = {"box": box, "point": {"x": box["left"] + box["width"] // 2, "y": box["top"] + box["height"] // 2}} if box else None
    return {"index": index, "text": text, "step": step, "visual_action": action, "target": "t", "pointer": pointer}


def _response_file(tmp_path, sentences=None):
    sentences = sentences or [
        _sentence(1, "Here is the search box.", step=0, action="none", box=None),
        _sentence(2, "It is at the bottom.", action="point"),
        _sentence(3, "Click it to search.", action="highlight"),
    ]
    path = tmp_path / "final_response.json"
    path.write_text(
        json.dumps({"screenshot_size": {"width": 1920, "height": 1080}, "sentences": sentences}), encoding="utf-8"
    )
    return FinalResponseResult(response_path=str(path), sentence_count=len(sentences))


def _presenter(overlay=None, speech=None):
    presenter = ResponsePresenter(
        overlay=overlay, speech=speech, thread_pool=SyncPool(), responding_seconds=0, hold_ms=lambda text: 0
    )
    statuses, finished = [], []
    presenter.status_changed.connect(statuses.append)
    presenter.finished.connect(lambda: finished.append(True))
    return presenter, statuses, finished


# --- the presenter ----------------------------------------------------------


def test_each_sentence_is_spoken_with_its_mark_in_order(tmp_path):
    overlay, speech = FakeOverlay(), FakeSpeech()
    presenter, statuses, finished = _presenter(overlay, speech)

    presenter.present(_response_file(tmp_path))

    assert overlay.prepared == [(1920, 1080)]
    assert speech.spoken == ["Here is the search box.", "It is at the bottom.", "Click it to search."]
    assert [mark[0] for mark in overlay.marks] == ["none", "point", "highlight"]
    assert overlay.marks[0][1] is None
    assert overlay.marks[1][1] == SEARCH_BOX
    assert overlay.marks[1][2] == {"x": 750, "y": 1025}
    assert overlay.cleared == 1
    assert finished == [True]


def test_it_says_responding_first_then_speaking(tmp_path):
    presenter, statuses, _finished = _presenter(FakeOverlay(), FakeSpeech())

    presenter.present(_response_file(tmp_path))

    assert statuses[0] is AssistantStatus.RESPONDING
    assert set(statuses[1:]) == {AssistantStatus.SPEAKING}


def test_without_speech_the_marks_are_still_shown(tmp_path):
    overlay = FakeOverlay()
    presenter, _statuses, finished = _presenter(overlay, None)

    presenter.present(_response_file(tmp_path))

    assert len(overlay.marks) == 3
    assert finished == [True]


def test_a_sentence_that_cannot_be_spoken_does_not_stop_the_rest(tmp_path, caplog):
    speech = FakeSpeech(fail_on={"It is at the bottom."})
    presenter, _statuses, finished = _presenter(FakeOverlay(), speech)

    with caplog.at_level(logging.INFO):
        presenter.present(_response_file(tmp_path))

    assert speech.spoken == ["Here is the search box.", "Click it to search."]
    assert "could not speak the sentence: no voice" in caplog.text
    assert finished == [True]


def test_no_marks_are_drawn_when_the_overlay_cannot_place_them(tmp_path):
    overlay, speech = FakeOverlay(ready=False), FakeSpeech()
    presenter, _statuses, finished = _presenter(overlay, speech)

    presenter.present(_response_file(tmp_path))

    assert overlay.marks == []
    assert len(speech.spoken) == 3
    assert finished == [True]


def test_without_an_overlay_the_sentences_are_still_spoken(tmp_path):
    speech = FakeSpeech()
    presenter, _statuses, finished = _presenter(None, speech)

    presenter.present(_response_file(tmp_path))

    assert len(speech.spoken) == 3
    assert finished == [True]


def test_an_unreadable_response_finishes_at_once(tmp_path, caplog):
    speech = FakeSpeech()
    presenter, statuses, finished = _presenter(FakeOverlay(), speech)

    with caplog.at_level(logging.INFO):
        presenter.present(FinalResponseResult(response_path=str(tmp_path / "missing.json")))

    assert speech.spoken == [] and statuses == []
    assert finished == [True]
    assert "could not read the final response" in caplog.text


def test_reading_time_grows_with_the_sentence_and_has_a_floor():
    assert reading_ms("Hi.") == 1500
    assert reading_ms(" ".join(["word"] * 26)) == 10000


# --- speech -----------------------------------------------------------------


class FakeEngine:
    def __init__(self, error=None):
        self.said = []
        self.properties = {}
        self.stopped = 0
        self._error = error

    def setProperty(self, name, value):
        self.properties[name] = value

    def say(self, text):
        self.said.append(text)

    def runAndWait(self):
        if self._error:
            raise self._error

    def stop(self):
        self.stopped += 1


def test_the_windows_voice_speaks_a_sentence_with_its_settings():
    engine = FakeEngine()
    speech = WindowsSpeechProvider(rate_wpm=180, volume=0.8, voice_id="v1", engine_factory=lambda: engine)

    speech.speak("  Hello\n there ")

    assert engine.said == ["Hello there"]
    assert engine.properties == {"rate": 180, "volume": 0.8, "voice": "v1"}
    assert engine.stopped == 1


def test_an_empty_sentence_is_not_spoken():
    created = []
    speech = WindowsSpeechProvider(rate_wpm=0, volume=None, voice_id="", engine_factory=lambda: created.append(1))

    speech.speak("   ")

    assert created == []


def test_a_speech_failure_is_a_speech_error():
    speech = WindowsSpeechProvider(
        rate_wpm=0, volume=None, voice_id="", engine_factory=lambda: FakeEngine(error=RuntimeError("audio device"))
    )

    with pytest.raises(SpeechError, match="audio device"):
        speech.speak("Hello")


# --- the Kokoro voice ---------------------------------------------------------


class FakeKokoro:
    def __init__(self, error=None):
        self.made = []
        self._error = error

    def create(self, text, voice, speed, lang):
        if self._error:
            raise self._error
        self.made.append((text, voice, speed, lang))
        return [0.0] * 10, 24000


class FakePlayer:
    def __init__(self):
        self.played = []
        self.stopped = 0

    def play(self, samples, sample_rate):
        self.played.append((len(samples), sample_rate))

    def stop(self):
        self.stopped += 1


def _kokoro(tmp_path, kokoro=None, voice="af_heart", language=None):
    from app.final_response.kokoro_speech import KokoroSpeechProvider

    kokoro = kokoro or FakeKokoro()
    player = FakePlayer()
    loaded = []

    def loader(model_path, voices_path):
        loaded.append((model_path, voices_path))
        return kokoro

    speech = KokoroSpeechProvider(
        voice=voice,
        speed=1.0,
        language=language,
        model_dir=tmp_path,
        loader=loader,
        player=player,
        prepare_files=lambda folder: (folder / "m.onnx", folder / "v.bin"),
    )
    return speech, kokoro, player, loaded


def test_kokoro_makes_and_plays_a_sentence(tmp_path):
    speech, kokoro, player, loaded = _kokoro(tmp_path)

    speech.speak("  Hello\n there ")

    assert kokoro.made == [("Ready.", "af_heart", 1.0, "en-us"), ("Hello there", "af_heart", 1.0, "en-us")]
    assert player.played == [(10, 24000)]
    assert loaded == [(tmp_path / "m.onnx", tmp_path / "v.bin")]


def test_kokoro_prepares_sentences_ahead_and_speaks_them_in_order(tmp_path):
    speech, kokoro, player, _loaded = _kokoro(tmp_path)

    speech.prepare(["One.", "Two.", "One."])
    speech._maker.submit(lambda: None).result()  # every queued sentence is made
    assert [made[0] for made in kokoro.made] == ["Ready.", "One.", "Two.", "One."]

    for text in ("One.", "Two.", "One."):
        speech.speak(text)

    assert len(kokoro.made) == 4  # the warm-up, then nothing was made twice
    assert len(player.played) == 3
    assert speech._pending == {}


def test_kokoro_picks_the_pronunciation_from_the_voice(tmp_path):
    from app.final_response.kokoro_speech import language_for

    assert language_for("af_heart") == "en-us"
    assert language_for("bf_emma") == "en-gb"
    assert language_for("hf_alpha") == "en-us"
    assert language_for("bf_emma", "en-us") == "en-us"


def test_a_kokoro_failure_is_a_speech_error(tmp_path):
    speech, _kokoro_engine, player, _loaded = _kokoro(tmp_path, kokoro=FakeKokoro(error=RuntimeError("bad phonemes")))

    with pytest.raises(SpeechError, match="bad phonemes"):
        speech.speak("Hello")
    assert player.played == []


def test_a_missing_model_is_downloaded_into_the_model_folder(tmp_path):
    from app.final_response.kokoro_speech import MODEL_FILE, VOICES_FILE, ensure_model_files

    (tmp_path / VOICES_FILE).write_bytes(b"voices")
    requested = []

    class Response:
        def __init__(self, data):
            self._chunks = [data, b""]

        def read(self, _size):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(url):
        requested.append(url)
        return Response(b"model-bytes")

    model, voices = ensure_model_files(tmp_path, opener=opener)

    assert requested == [
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    ]
    assert model == tmp_path / MODEL_FILE and model.read_bytes() == b"model-bytes"
    assert voices.read_bytes() == b"voices"  # an existing file is used as it is


def test_a_failed_download_leaves_no_partial_file(tmp_path):
    from app.final_response.kokoro_speech import ensure_model_files

    def opener(url):
        raise OSError("offline")

    with pytest.raises(SpeechError, match="offline"):
        ensure_model_files(tmp_path, opener=opener)
    assert list(tmp_path.iterdir()) == []


def test_the_factory_picks_the_voice_from_the_config(monkeypatch):
    from app import config
    from app.final_response import kokoro_speech
    from app.final_response.speech import create_speech_provider

    monkeypatch.setattr(config, "TTS_ENABLED", True)
    monkeypatch.setattr(config, "TTS_PROVIDER", "windows")
    assert isinstance(create_speech_provider(), WindowsSpeechProvider)

    monkeypatch.setattr(config, "TTS_PROVIDER", "kokoro")
    created = []
    monkeypatch.setattr(kokoro_speech, "KokoroSpeechProvider", lambda: created.append(True) or "kokoro")
    assert create_speech_provider() == "kokoro"
    assert created == [True]

    monkeypatch.setattr(config, "TTS_ENABLED", False)
    assert create_speech_provider() is None


def test_the_presenter_lets_the_voice_prepare_every_sentence_first(tmp_path):
    class PreparingSpeech(FakeSpeech):
        def __init__(self):
            super().__init__()
            self.prepared = []

        def prepare(self, texts):
            self.prepared.append(list(texts))

    speech = PreparingSpeech()
    presenter, _statuses, finished = _presenter(FakeOverlay(), speech)

    presenter.present(_response_file(tmp_path))

    assert speech.prepared == [["Here is the search box.", "It is at the bottom.", "Click it to search."]]
    assert speech.spoken == speech.prepared[0]
    assert finished == [True]


def test_kokoro_starts_loading_as_soon_as_it_is_created(tmp_path):
    import threading

    from app.final_response.kokoro_speech import KokoroSpeechProvider

    started, release = threading.Event(), threading.Event()

    def slow_files(folder):
        started.set()
        release.wait(5)
        return folder / "m.onnx", folder / "v.bin"

    speech = KokoroSpeechProvider(
        voice="af_heart", model_dir=tmp_path, loader=lambda m, v: FakeKokoro(), player=FakePlayer(),
        prepare_files=slow_files,
    )

    assert started.wait(5)  # loading began before any sentence was asked for
    assert not speech.is_loaded()
    release.set()
    assert speech.wait_until_loaded(5) is True
    assert speech.is_loaded() and speech.load_error() is None


def test_when_kokoro_cannot_load_the_windows_voice_speaks(tmp_path):
    from app.final_response.kokoro_speech import KokoroSpeechProvider

    def no_files(folder):
        raise SpeechError("offline")

    fallback = FakeSpeech()
    speech = KokoroSpeechProvider(
        voice="af_heart", model_dir=tmp_path, loader=lambda m, v: FakeKokoro(), player=FakePlayer(),
        prepare_files=no_files, fallback=lambda: fallback,
    )

    assert speech.wait_until_loaded(5) is False
    assert "offline" in str(speech.load_error())
    speech.prepare(["Hello."])
    assert speech.is_ready("Hello.")
    speech.speak("Hello.")
    assert fallback.spoken == ["Hello."]


def test_a_missing_model_is_reported_before_loading(tmp_path):
    from app.final_response.kokoro_speech import MODEL_FILE, VOICES_FILE, missing_model_files

    assert missing_model_files(tmp_path) == [MODEL_FILE, VOICES_FILE]
    (tmp_path / MODEL_FILE).write_bytes(b"m")
    (tmp_path / VOICES_FILE).write_bytes(b"v")
    assert missing_model_files(tmp_path) == []


class FakeTray:
    def __init__(self):
        self.messages = []
        self.tooltips = []

    def setToolTip(self, text):
        self.tooltips.append(text)

    def showMessage(self, title, message, icon, ms):
        self.messages.append(message)


class LoadingSpeech:
    def __init__(self, needs_download=False, error=None):
        self.needs_download = needs_download
        self.loaded = False
        self._error = error

    def is_loaded(self):
        return self.loaded

    def load_error(self):
        return self._error if self.loaded else None


def test_questions_are_accepted_only_after_the_voice_is_ready(qt_app):  # noqa: F811
    from app.main import when_voice_ready

    speech, tray, started = LoadingSpeech(needs_download=True), FakeTray(), []
    timer = when_voice_ready(speech, tray, lambda: started.append(True))

    assert started == []  # still downloading: the hotkey is not listening yet
    assert "Downloading the voice model" in tray.messages[0]

    timer.timeout.emit()
    assert started == []

    speech.loaded = True
    timer.timeout.emit()
    assert started == [True]
    assert not timer.isActive()
    assert tray.tooltips[-1] == "Hey Arro"
    assert "Voice ready" in tray.messages[-1]


def test_hey_arro_still_starts_when_the_voice_fails_to_load(qt_app):  # noqa: F811
    from app.main import when_voice_ready

    speech, tray, started = LoadingSpeech(error=SpeechError("offline")), FakeTray(), []
    timer = when_voice_ready(speech, tray, lambda: started.append(True))
    speech.loaded = True
    timer.timeout.emit()

    assert started == [True]
    assert "Windows voice" in tray.messages[-1]


def test_a_voice_without_loading_starts_at_once():
    from app.main import when_voice_ready

    started = []
    assert when_voice_ready(None, FakeTray(), lambda: started.append(1)) is None
    assert when_voice_ready(FakeSpeech(), FakeTray(), lambda: started.append(2)) is None
    assert started == [1, 2]


def test_the_mark_waits_until_its_sentence_can_be_heard(tmp_path, monkeypatch):
    from app.final_response import presenter as presenter_module

    monkeypatch.setattr(presenter_module, "READY_POLL_MS", 0)
    overlay = FakeOverlay()

    class SlowSpeech(FakeSpeech):
        def __init__(self):
            super().__init__()
            self.checks = 0
            self.marks_when_ready = []

        def is_ready(self, text):
            self.checks += 1
            if self.checks % 3:  # ready on every third check
                return False
            self.marks_when_ready.append(len(overlay.marks))
            return True

    speech = SlowSpeech()
    presenter, _statuses, finished = _presenter(overlay, speech)

    presenter.present(_response_file(tmp_path))

    # Each mark was drawn only after its sentence was ready, never before.
    assert speech.marks_when_ready == [0, 1, 2]
    assert len(speech.spoken) == 3
    assert finished == [True]


# --- the overlay --------------------------------------------------------------


def test_screenshot_pixels_are_scaled_to_logical_pixels():
    scale = scale_for((1920, 1080), (1280, 720))

    assert scale == pytest.approx((2 / 3, 2 / 3))
    assert map_box(SEARCH_BOX, scale) == QRectF(400, 1000 * 2 / 3, 200, 50 * 2 / 3)
    assert map_point({"x": 750, "y": 1025}, scale) == QPointF(500, 1025 * 2 / 3)


def test_the_overlay_skips_the_combined_monitor_and_uses_the_primary_for_monitor_one():
    assert target_screen((1920, 1080), 0) is None
    assert target_screen((1920, 1080), 1) is not None


@pytest.mark.parametrize("action", ["point", "circle", "underline", "arrow", "highlight", "draw"])
def test_every_mark_can_be_painted(action):
    overlay = ScreenOverlay(monitor_index=1)
    overlay.resize(800, 600)
    overlay._mark = (action, QRectF(300, 250, 120, 40), QPointF(360, 270))
    overlay._progress = 1.0
    image = QImage(800, 600, QImage.Format_ARGB32_Premultiplied)
    image.fill(0)

    overlay.render(image)

    assert any(image.pixelColor(x, y).alpha() > 0 for x in range(250, 480, 4) for y in range(200, 340, 4))


def test_a_mark_without_a_box_or_action_draws_nothing():
    overlay = ScreenOverlay(monitor_index=1)
    overlay._scale = (1.0, 1.0)

    overlay.show_mark("none", SEARCH_BOX, None)
    assert overlay._mark is None
    overlay.show_mark("point", None, None)
    assert overlay._mark is None


# --- the status indicator ---------------------------------------------------


@pytest.mark.parametrize("status", [s for s in AssistantStatus if s is not AssistantStatus.IDLE])
def test_every_status_has_a_label_and_can_be_painted(status):
    indicator = StatusIndicator()
    indicator._status = status
    image = QImage(indicator.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(0)

    indicator.render(image)

    assert status.label
    assert image.pixelColor(image.width() // 2, image.height() // 2).alpha() > 0


# --- the coordinator --------------------------------------------------------


class FakePresenter(QObject):
    status_changed = Signal(object)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.presented = []

    def present(self, response):
        self.presented.append(response)
        self.status_changed.emit(AssistantStatus.RESPONDING)
        self.status_changed.emit(AssistantStatus.SPEAKING)
        self.finished.emit()


def _build(tmp_path, presenter=None, category=ResponseCategory.GUIDING, composer=None):
    hotkey = FakeHotkey()
    folder = tmp_path / "shot"
    folder.mkdir()
    coordinator = ApplicationCoordinator(
        hotkey=hotkey,
        recorder=StubRecorder(),
        transcription=StubTranscription(text="where is the search box?"),
        capture=StubCapture(),
        thread_pool=SyncPool(),
        screenshot_dir=tmp_path / "plain",
        refinement=SegmentedRefinement(folder),
        refinement_dir=tmp_path / "segments",
        segment_selector=CategorisingSelector(category),
        response_planner=StubPlanner(),
        plan_grounder=StubGrounder(),
        response_composer=composer or StubComposer(),
        response_presenter=presenter,
    )
    statuses = []
    coordinator.status_changed.connect(statuses.append)
    return hotkey, coordinator, statuses


def test_the_status_follows_the_flow(tmp_path):
    presenter = FakePresenter()
    hotkey, coordinator, statuses = _build(tmp_path, presenter)

    ask(hotkey)

    assert statuses == [
        AssistantStatus.LISTENING,
        AssistantStatus.TRANSCRIBING,
        AssistantStatus.REFINING,
        AssistantStatus.PLANNING,
        AssistantStatus.RESPONDING,
        AssistantStatus.SPEAKING,
        AssistantStatus.IDLE,
    ]
    assert coordinator.status is AssistantStatus.IDLE


def test_a_press_shows_listening_until_release(tmp_path):
    hotkey, coordinator, statuses = _build(tmp_path)

    hotkey.pressed.emit()

    assert statuses == [AssistantStatus.LISTENING]
    assert coordinator.status is AssistantStatus.LISTENING


def test_the_interaction_ends_only_after_the_response_is_presented(tmp_path, caplog):
    presenter = FakePresenter()
    hotkey, coordinator, _statuses = _build(tmp_path, presenter)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert [r.response_path for r in presenter.presented] == ["final_response.json"]
    assert caplog.text.index("Final response complete") < caplog.text.index("Final response presented")
    assert caplog.text.index("Final response presented") < caplog.text.index("Interaction complete")
    assert coordinator.working is False


def test_a_fallback_question_goes_back_to_idle_without_presenting(tmp_path):
    presenter = FakePresenter()
    hotkey, coordinator, statuses = _build(tmp_path, presenter, category=ResponseCategory.FALLBACK)

    ask(hotkey)

    assert presenter.presented == []
    assert statuses[-1] is AssistantStatus.IDLE
    assert AssistantStatus.PLANNING not in statuses


def test_a_failure_goes_back_to_idle(tmp_path):
    hotkey = FakeHotkey()
    coordinator = ApplicationCoordinator(
        hotkey=hotkey,
        recorder=StubRecorder(),
        transcription=StubTranscription(),
        capture=StubCapture(error=RuntimeError("capture exploded")),
        thread_pool=SyncPool(),
        screenshot_dir=tmp_path,
    )
    statuses = []
    coordinator.status_changed.connect(statuses.append)

    ask(hotkey)

    assert statuses[-1] is AssistantStatus.IDLE
    assert coordinator.working is False


def test_a_failed_response_does_not_present_and_ends_idle(tmp_path):
    presenter = FakePresenter()
    hotkey, coordinator, statuses = _build(tmp_path, presenter, composer=StubComposer(error=RuntimeError("quota")))

    ask(hotkey)

    assert presenter.presented == []
    assert statuses[-1] is AssistantStatus.IDLE
