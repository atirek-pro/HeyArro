import builtins
import threading
from pathlib import Path

import pytest

import app.coordinator
import app.tts.provider as tts_provider_module
from app import config
from app.tts import (
    TTSError,
    TTSNotImplementedError,
    TTSProvider,
    UnsupportedTTSProviderError,
    available_tts_providers,
    get_tts_provider,
    normalize_text,
)
from app.tts.elevenlabs_provider import ElevenLabsTTSProvider
from app.tts.mock_provider import MockTTSProvider
from app.tts.windows_provider import WindowsTTSProvider, create_windows_engine
from app.tts.worker import TTSWorker


# --- test doubles -----------------------------------------------------------


class FakeEngine:
    """Stands in for a pyttsx3 engine without making any sound."""

    def __init__(self, fail_on_run=False):
        self.said = []
        self.properties = {}
        self.ran = 0
        self.stop_calls = 0
        self._fail_on_run = fail_on_run
        self.started = None
        self.release = None

    def say(self, text):
        self.said.append(text)

    def runAndWait(self):
        self.ran += 1
        if self._fail_on_run:
            raise RuntimeError("engine exploded")
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(2)

    def setProperty(self, name, value):
        self.properties[name] = value

    def stop(self):
        self.stop_calls += 1
        if self.release is not None:
            self.release.set()


def windows_provider(engine=None, error=None, **kwargs):
    """A Windows provider wired to a fake engine factory."""
    engine = engine if engine is not None else FakeEngine()
    calls = {"count": 0}

    def factory():
        calls["count"] += 1
        if error is not None:
            raise error
        return engine

    provider = WindowsTTSProvider(engine_factory=factory, **kwargs)
    return provider, engine, calls


class RecordingProvider(TTSProvider):
    name = "recording"

    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)

    def stop(self):
        return None


# --- windows provider -------------------------------------------------------


def test_windows_provider_can_be_instantiated():
    provider = WindowsTTSProvider()

    assert isinstance(provider, TTSProvider)
    assert provider.name == "windows"


def test_windows_provider_speaks_the_supplied_text():
    provider, engine, _calls = windows_provider()

    provider.speak("Hello there.")

    assert engine.said == ["Hello there."]
    assert engine.ran == 1


def test_windows_provider_normalises_whitespace():
    provider, engine, _calls = windows_provider()

    provider.speak("  Hello\n\tthere  ")

    assert engine.said == ["Hello there"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_windows_provider_treats_blank_text_as_a_no_op(text):
    provider, _engine, calls = windows_provider()

    provider.speak(text)

    assert calls["count"] == 0


def test_windows_provider_stop_without_speech_is_safe():
    provider, engine, _calls = windows_provider()

    provider.stop()

    assert engine.stop_calls == 0


def test_windows_provider_stop_interrupts_active_speech():
    engine = FakeEngine()
    engine.started = threading.Event()
    engine.release = threading.Event()
    provider = WindowsTTSProvider(engine_factory=lambda: engine)

    thread = threading.Thread(target=provider.speak, args=("hold on",))
    thread.start()
    assert engine.started.wait(2), "speech never started"

    provider.stop()
    assert engine.stop_calls >= 1

    engine.release.set()
    thread.join(2)
    assert not thread.is_alive()


def test_windows_provider_engine_initialization_failure_is_reported():
    provider, _engine, _calls = windows_provider(error=TTSError("Could not initialize the engine"))

    with pytest.raises(TTSError):
        provider.speak("hello")


def test_windows_provider_speech_failure_is_reported():
    provider, _engine, _calls = windows_provider(engine=FakeEngine(fail_on_run=True))

    with pytest.raises(TTSError) as info:
        provider.speak("hello")

    assert "engine exploded" in str(info.value)


def test_missing_pyttsx3_is_reported_as_a_tts_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyttsx3":
            raise ImportError("no module named pyttsx3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(TTSError) as info:
        create_windows_engine()

    assert "pyttsx3" in str(info.value)


def test_configured_voice_settings_are_applied():
    provider, engine, _calls = windows_provider(rate_wpm=180, volume=0.5, voice_id="voice-1")

    provider.speak("hi")

    assert engine.properties == {"rate": 180, "volume": 0.5, "voice": "voice-1"}


# --- text normalisation -----------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [("  a  b ", "a b"), ("", ""), (None, ""), ("\n\t", ""), ("plain", "plain")],
)
def test_normalize_text(value, expected):
    assert normalize_text(value) == expected


# --- mock provider ----------------------------------------------------------


def test_mock_provider_records_spoken_text():
    tts = MockTTSProvider()

    tts.speak("Open Settings and select Network.")

    assert tts.spoken_texts == ["Open Settings and select Network."]
    assert tts.spoken is True


def test_mock_provider_ignores_blank_text():
    tts = MockTTSProvider()

    tts.speak("   ")

    assert tts.spoken_texts == []
    assert tts.spoken is False


def test_mock_provider_stop_is_safe_and_counted():
    tts = MockTTSProvider()

    tts.stop()
    tts.stop()

    assert tts.stop_calls == 2


# --- elevenlabs placeholder -------------------------------------------------


def test_elevenlabs_is_a_registered_placeholder():
    provider = ElevenLabsTTSProvider()

    assert isinstance(provider, TTSProvider)
    assert provider.name == "elevenlabs"

    with pytest.raises(TTSNotImplementedError) as info:
        provider.speak("hello")

    assert "not implemented" in str(info.value).lower()


def test_elevenlabs_also_raises_plain_not_implemented_error():
    with pytest.raises(NotImplementedError):
        ElevenLabsTTSProvider().speak("hello")


def test_elevenlabs_stop_is_safe():
    assert ElevenLabsTTSProvider().stop() is None


# --- provider factory -------------------------------------------------------


def test_factory_can_build_every_registered_provider():
    expected = {
        "windows": WindowsTTSProvider,
        "elevenlabs": ElevenLabsTTSProvider,
        "mock": MockTTSProvider,
    }

    for name, provider_class in expected.items():
        assert isinstance(get_tts_provider(name), provider_class)


def test_factory_lists_every_provider():
    assert available_tts_providers() == ["elevenlabs", "mock", "windows"]


def test_factory_uses_the_configured_provider(monkeypatch):
    monkeypatch.setattr(config, "TTS_PROVIDER", "mock")

    assert isinstance(get_tts_provider(), MockTTSProvider)


def test_factory_is_case_and_whitespace_insensitive():
    assert isinstance(get_tts_provider("  MoCk "), MockTTSProvider)


def test_unsupported_provider_name_produces_a_clear_error():
    with pytest.raises(UnsupportedTTSProviderError) as info:
        get_tts_provider("hal9000")

    message = str(info.value)
    assert "hal9000" in message
    assert "windows" in message


def test_missing_provider_name_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "TTS_PROVIDER", "")

    with pytest.raises(UnsupportedTTSProviderError):
        get_tts_provider()


# --- worker -----------------------------------------------------------------


def test_worker_forwards_text_and_emits_finished():
    provider = RecordingProvider()
    worker = TTSWorker(provider, "hello")
    finished = []
    worker.signals.finished.connect(lambda: finished.append(True))

    worker.run()

    assert provider.spoken == ["hello"]
    assert finished == [True]


def test_worker_reports_failures_without_raising():
    worker = TTSWorker(ElevenLabsTTSProvider(), "hello")
    failed = []
    worker.signals.failed.connect(failed.append)

    worker.run()

    assert len(failed) == 1
    assert "not implemented" in failed[0].lower()


# --- architecture -----------------------------------------------------------


def test_tts_contract_is_independent_of_the_response_layers():
    source = Path(tts_provider_module.__file__).read_text(encoding="utf-8").lower()

    for forbidden in ("gemini", "openai", "claude", "pyttsx3"):
        assert forbidden not in source

    # The TTS contract never imports the vision or capture layers.
    assert "app.llm" not in source
    assert "app.capture" not in source


def test_coordinator_contains_no_engine_specific_code():
    source = Path(app.coordinator.__file__).read_text(encoding="utf-8").lower()

    assert "pyttsx3" not in source
    assert "windows_provider" not in source
    assert "elevenlabs" not in source
