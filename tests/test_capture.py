import pytest

from app.capture import (
    MonitorInfo,
    ScreenCaptureResult,
    create_screen_capture_provider,
    save_png,
)
from app.capture.provider import ScreenCaptureProvider
from app.pipeline import ProcessingCoordinator

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _fake_png(size=64):
    return PNG_SIGNATURE + b"\x00" * size


def _provider_or_skip():
    provider = create_screen_capture_provider()
    try:
        monitors = provider.list_monitors()
    except Exception as exc:  # pragma: no cover - only on a machine without a display
        pytest.skip(f"screen capture unavailable: {exc}")
    return provider, monitors


def test_factory_returns_a_screen_capture_provider():
    assert isinstance(create_screen_capture_provider(), ScreenCaptureProvider)


def test_result_exposes_size_and_resolution():
    result = ScreenCaptureResult(
        image=_fake_png(10),
        monitor_index=1,
        width=1920,
        height=1080,
        timestamp=123.0,
        monitor=MonitorInfo(1, 0, 0, 1920, 1080, is_primary=True),
    )
    assert result.size_bytes == len(PNG_SIGNATURE) + 10
    assert result.resolution == "1920x1080"
    assert result.monitor.resolution == "1920x1080"
    assert result.path is None


def test_save_png_writes_an_openable_png(tmp_path):
    result = ScreenCaptureResult(
        image=_fake_png(),
        monitor_index=1,
        width=8,
        height=8,
        timestamp=1.0,
    )

    path = save_png(result, tmp_path)

    assert path.exists()
    assert path.suffix == ".png"
    assert path.read_bytes() == result.image
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_list_monitors_includes_virtual_screen_and_primary():
    _provider, monitors = _provider_or_skip()

    assert len(monitors) >= 2
    assert monitors[0].index == 0
    assert monitors[1].is_primary is True
    assert all(monitor.width > 0 and monitor.height > 0 for monitor in monitors)


def test_capture_returns_png_matching_reported_geometry():
    provider, monitors = _provider_or_skip()

    result = provider.capture_screen()
    assert result.image.startswith(PNG_SIGNATURE)
    assert result.size_bytes > len(PNG_SIGNATURE)
    assert result.resolution == monitors[provider.monitor_index].resolution
    assert result.monitor_index == provider.monitor_index
    assert result.timestamp > 0


def test_every_monitor_can_be_captured():
    provider, monitors = _provider_or_skip()

    for monitor in monitors:
        result = provider.capture_screen(monitor.index)
        assert result.monitor_index == monitor.index
        assert (result.width, result.height) == (monitor.width, monitor.height)
        assert result.image.startswith(PNG_SIGNATURE)


def test_out_of_range_monitor_is_rejected():
    provider, monitors = _provider_or_skip()

    with pytest.raises(Exception):
        provider.capture_screen(len(monitors) + 5)


def test_coordinator_waits_for_both_sides():
    coordinator = ProcessingCoordinator()
    completed = []
    coordinator.completed.connect(completed.append)

    coordinator.start("audio.wav")
    coordinator.set_transcript("hello world")
    assert completed == []

    marker = object()
    coordinator.set_screenshot(marker)
    assert len(completed) == 1
    assert completed[0].transcript == "hello world"
    assert completed[0].screenshot is marker
    assert completed[0].audio_path == "audio.wav"
    assert completed[0].has_transcript and completed[0].has_screenshot


def test_coordinator_completes_even_when_capture_fails():
    coordinator = ProcessingCoordinator()
    completed = []
    coordinator.completed.connect(completed.append)

    coordinator.start("audio.wav")
    coordinator.set_screenshot(object())
    coordinator.set_transcript_error("no speech")

    assert len(completed) == 1
    assert completed[0].transcript_error == "no speech"
    assert completed[0].has_transcript is False
    assert completed[0].has_screenshot is True


def test_coordinator_ignores_results_without_an_interaction():
    coordinator = ProcessingCoordinator()
    completed = []
    coordinator.completed.connect(completed.append)

    coordinator.set_transcript("stray")

    assert completed == []
