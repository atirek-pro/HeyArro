"""Screenshot refinement: the folder it writes, the segments it finds, the logs.

The unstructured library is replaced by a fake partitioner, so these tests need
neither Tesseract nor a layout model.
"""

import io
import logging
from types import SimpleNamespace

import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

import app.refinement.unstructured_provider as module
from app.capture.provider import ScreenCaptureResult
from app.coordinator import ApplicationCoordinator
from app.refinement import (
    RefinementResult,
    ScreenRefinementError,
    ScreenRefinementProvider,
    create_refinement_provider,
)
from app.refinement.unstructured_provider import UnstructuredRefinementProvider
from tests.test_coordinator import (
    FakeHotkey,
    StubCapture,
    StubLLM,
    StubRecorder,
    StubTranscription,
    SyncPool,
    ask,
)


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    yield QApplication.instance() or QApplication([])


def _png(width=200, height=100):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _capture(width=200, height=100):
    return ScreenCaptureResult(
        image=_png(width, height), monitor_index=1, width=width, height=height, timestamp=1.0
    )


class Title:
    """Named like unstructured's element classes, which the provider uses as the kind."""

    def __init__(self, text, points):
        self.text = text
        coordinates = SimpleNamespace(points=points) if points else None
        self.metadata = SimpleNamespace(coordinates=coordinates)


def _box(left, top, right, bottom):
    return ((left, top), (left, bottom), (right, bottom), (right, top))


def _patch_partition(monkeypatch, elements):
    monkeypatch.setattr(module, "_import_partition_image", lambda: lambda **kwargs: elements)


# --- the provider -----------------------------------------------------------


def test_factory_returns_a_refinement_provider():
    assert isinstance(create_refinement_provider(), ScreenRefinementProvider)


def test_the_screenshot_and_its_segments_live_in_one_folder(tmp_path, monkeypatch):
    _patch_partition(monkeypatch, [Title("Hello", _box(10, 10, 110, 40))])

    result = UnstructuredRefinementProvider().refine(_capture(), tmp_path)

    folders = list(tmp_path.iterdir())
    assert len(folders) == 1
    assert str(folders[0]) == result.folder
    assert (folders[0] / "screenshot.png").exists()
    assert [p.name for p in (folders[0] / "segments").iterdir()] == ["segment-001-title.png"]
    assert not list(tmp_path.glob("*.png"))


def test_a_segment_is_cropped_to_its_box(tmp_path, monkeypatch):
    _patch_partition(monkeypatch, [Title("Hello", _box(10, 20, 110, 60))])

    result = UnstructuredRefinementProvider().refine(_capture(), tmp_path)

    segment = result.segments[0]
    assert (segment.kind, segment.text) == ("Title", "Hello")
    assert segment.box == (10, 20, 110, 60)
    with Image.open(segment.path) as cropped:
        assert cropped.size == (100, 40)


def test_boxes_are_clamped_to_the_screenshot(tmp_path, monkeypatch):
    _patch_partition(monkeypatch, [Title("Edge", _box(-5, -5, 500, 500))])

    result = UnstructuredRefinementProvider().refine(_capture(), tmp_path)

    assert result.segments[0].box == (0, 0, 200, 100)


def test_elements_without_coordinates_and_tiny_ones_are_skipped(tmp_path, monkeypatch, caplog):
    _patch_partition(
        monkeypatch,
        [
            Title("no box", None),
            Title("speck", _box(0, 0, 3, 3)),
            Title("kept", _box(0, 0, 50, 50)),
        ],
    )

    with caplog.at_level(logging.INFO):
        result = UnstructuredRefinementProvider().refine(_capture(), tmp_path)

    assert [s.text for s in result.segments] == ["kept"]
    assert result.segments[0].index == 1
    assert "has no coordinates; skipped" in caplog.text
    assert "is below" in caplog.text


def test_every_step_is_logged(tmp_path, monkeypatch, caplog):
    _patch_partition(monkeypatch, [Title("Hello", _box(10, 10, 110, 40))])

    with caplog.at_level(logging.INFO):
        UnstructuredRefinementProvider().refine(_capture(), tmp_path)

    for step in (
        "[Refinement] received screenshot: 200x100",
        "[Refinement] created folder",
        "[Refinement] saved screenshot to",
        "[Refinement] partitioning with unstructured",
        "[Refinement] unstructured returned 1 element(s)",
        "[Refinement] segment 1: kind=Title",
        "[Refinement] done in",
    ):
        assert step in caplog.text


def test_a_missing_library_is_a_refinement_error(tmp_path, monkeypatch):
    def missing():
        raise ScreenRefinementError("unstructured is not available: boom")

    monkeypatch.setattr(module, "_import_partition_image", missing)

    with pytest.raises(ScreenRefinementError, match="not available"):
        UnstructuredRefinementProvider().refine(_capture(), tmp_path)


def test_a_failing_partition_is_a_refinement_error(tmp_path, monkeypatch):
    def explode(**kwargs):
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(module, "_import_partition_image", lambda: explode)

    with pytest.raises(ScreenRefinementError, match="tesseract not found"):
        UnstructuredRefinementProvider().refine(_capture(), tmp_path)


# --- the coordinator --------------------------------------------------------


class StubRefinement(ScreenRefinementProvider):
    def __init__(self, error=None):
        self._error = error
        self.captures = []

    def refine(self, capture, output_dir):
        self.captures.append(capture)
        if self._error is not None:
            raise self._error
        return RefinementResult(
            folder=str(output_dir / "shot"),
            screenshot_path=str(output_dir / "shot" / "screenshot.png"),
            segments=[],
            strategy="stub",
        )


def _build(tmp_path, refinement):
    hotkey = FakeHotkey()
    llm = StubLLM()
    coordinator = ApplicationCoordinator(
        hotkey=hotkey,
        recorder=StubRecorder(),
        transcription=StubTranscription(),
        capture=StubCapture(),
        llm=llm,
        thread_pool=SyncPool(),
        screenshot_dir=tmp_path / "plain",
        refinement=refinement,
        refinement_dir=tmp_path / "segments",
    )
    return hotkey, llm, coordinator


def test_the_screenshot_is_refined_before_the_model_is_asked(tmp_path, caplog):
    refinement = StubRefinement()
    hotkey, llm, _coordinator = _build(tmp_path, refinement)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert len(refinement.captures) == 1
    assert len(llm.requests) == 1
    assert "Refinement complete: 0 segment(s), strategy=stub" in caplog.text
    assert not (tmp_path / "plain").exists()
    assert caplog.text.index("Refining screenshot") < caplog.text.index("[Input] transcript")


def test_a_failed_refinement_still_answers_the_question(tmp_path, caplog):
    hotkey, llm, coordinator = _build(tmp_path, StubRefinement(error=RuntimeError("boom")))

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "Refinement failed, continuing with the plain screenshot: boom" in caplog.text
    assert len(llm.requests) == 1
    assert len(list((tmp_path / "plain").glob("*.png"))) == 1
    assert coordinator.working is False
