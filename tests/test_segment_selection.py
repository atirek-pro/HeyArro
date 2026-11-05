"""Segment manifests and relevant segment selection.

Gemini is replaced by a fake client and unstructured by a fake partitioner, so
these tests need neither a key nor a layout model.
"""

import json
import logging
from types import SimpleNamespace

import pytest

from app.coordinator import ApplicationCoordinator
from app.refinement import (
    RefinementResult,
    ResponseCategory,
    ScreenRefinementProvider,
    ScreenSegment,
    SegmentSelectionError,
    SegmentSelector,
    SelectionResult,
)
from app.refinement.gemini_selector import GeminiSegmentSelector
from app.refinement.manifest import (
    MANIFEST_NAME,
    RELEVANT_MANIFEST_NAME,
    write_relevant_manifest,
    write_segment_manifest,
)
from app.refinement.unstructured_provider import UnstructuredRefinementProvider
from tests.test_coordinator import (
    FakeHotkey,
    StubCapture,
    StubRecorder,
    StubTranscription,
    SyncPool,
    ask,
)
from tests.test_refinement import Title, _box, _capture, _patch_partition, qt_app  # noqa: F401


def _refinement(tmp_path):
    """A refined screenshot with two segments, laid out as the provider writes it."""
    (tmp_path / "segments").mkdir()
    (tmp_path / "screenshot.png").write_bytes(b"png")
    segments = [
        ScreenSegment(1, "Title", "Explorer", 0, 0, 100, 20, str(tmp_path / "segments" / "segment-001-title.png")),
        ScreenSegment(2, "Table", "app audio tests", 0, 30, 100, 200, str(tmp_path / "segments" / "segment-002-table.png")),
        ScreenSegment(3, "Footer", "29C Sunny", 0, 500, 100, 20, None),
    ]
    return RefinementResult(
        folder=str(tmp_path), screenshot_path=str(tmp_path / "screenshot.png"), segments=segments, strategy="stub"
    )


class FakeModels:
    def __init__(self, answer, error=None):
        self._answer = answer
        self._error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        text = self._answer if isinstance(self._answer, str) else json.dumps(self._answer)
        return SimpleNamespace(parsed=None, text=text, usage_metadata=None)


def _selector(answer, error=None):
    models = FakeModels(answer, error)
    return GeminiSegmentSelector(client=SimpleNamespace(models=models), model="fake-model"), models


def _answer(segments, category="instructing", confidence=0.9, reason="The user wants to do a task."):
    return {"response_category": category, "confidence": confidence, "reason": reason, "segments": segments}


# --- ss_segment_manifest.json -----------------------------------------------


def test_refinement_writes_the_segment_manifest_next_to_the_screenshot(tmp_path, monkeypatch):
    _patch_partition(monkeypatch, [Title("Hello", _box(10, 20, 110, 60))])

    result = UnstructuredRefinementProvider().refine(_capture(), tmp_path)

    manifest = json.loads((tmp_path / result.folder.split("\\")[-1] / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["screenshot"] == "screenshot.png"
    assert manifest["segment_count"] == 1
    assert manifest["segments"] == [
        {
            "id": 1,
            "kind": "Title",
            "text": "Hello",
            "box": {"left": 10, "top": 20, "width": 100, "height": 40},
            "file": "segments/segment-001-title.png",
        }
    ]


def test_a_manifest_that_cannot_be_written_does_not_fail_refinement(tmp_path, monkeypatch, caplog):
    _patch_partition(monkeypatch, [Title("Hello", _box(10, 20, 110, 60))])

    def explode(result):
        raise OSError("disk full")

    monkeypatch.setattr("app.refinement.unstructured_provider.write_segment_manifest", explode)

    with caplog.at_level(logging.INFO):
        result = UnstructuredRefinementProvider().refine(_capture(), tmp_path)

    assert result.segment_count == 1
    assert "could not write the segment manifest: disk full" in caplog.text


# --- relevant_ss_manifest.json ----------------------------------------------


def test_the_relevant_manifest_keeps_scored_segments_highest_first(tmp_path):
    refinement = _refinement(tmp_path)

    path = write_relevant_manifest(
        refinement, "where is it?", "fake-model", {1: 3, 2: 10}, 1.23456, "guiding", 0.8, "It asks where."
    )

    assert path.name == RELEVANT_MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["query"] == "where is it?"
    assert manifest["response_category"] == "guiding"
    assert manifest["confidence"] == 0.8
    assert manifest["category_reason"] == "It asks where."
    assert manifest["model"] == "fake-model"
    assert manifest["source_manifest"] == MANIFEST_NAME
    assert (manifest["total_segments"], manifest["selected_count"]) == (3, 2)
    assert manifest["selection_seconds"] == 1.235
    assert [(s["id"], s["score"]) for s in manifest["segments"]] == [(2, 10), (1, 3)]
    assert manifest["segments"][0]["kind"] == "Table"
    assert manifest["segments"][0]["file"] == "segments/segment-002-table.png"


def test_the_full_manifest_lists_every_segment(tmp_path):
    path = write_segment_manifest(_refinement(tmp_path))

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert [s["id"] for s in manifest["segments"]] == [1, 2, 3]
    assert manifest["segments"][2]["file"] is None


# --- the Gemini selector ----------------------------------------------------


def test_the_selector_scores_segments_and_writes_the_manifest(tmp_path):
    selector, models = _selector(_answer([{"segment_id": 2, "score": 10}, {"segment_id": 1, "score": 4}]))

    result = selector.select(_refinement(tmp_path), "how many folders?")

    assert result.scores == {2: 10, 1: 4}
    assert result.selected_count == 2
    saved = json.loads((tmp_path / RELEVANT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [s["id"] for s in saved["segments"]] == [2, 1]

    call = models.calls[0]
    assert call["model"] == "fake-model"
    prompt = call["contents"].parts[1].text
    assert "how many folders?" in prompt
    assert '"id":2' in prompt and "app audio tests" in prompt
    assert "segment-002-table.png" not in prompt


def test_unknown_ids_are_dropped_and_scores_are_clamped(tmp_path):
    selector, _models = _selector(
        _answer(
            [
                {"segment_id": 99, "score": 9},
                {"segment_id": 1, "score": 50},
                {"segment_id": 2, "score": 0},
                {"segment_id": 3, "score": 2},
                {"segment_id": 3, "score": 6},
            ]
        )
    )

    result = selector.select(_refinement(tmp_path), "q")

    assert result.scores == {1: 10, 3: 6}


def test_nothing_relevant_still_writes_an_empty_manifest(tmp_path):
    selector, _models = _selector(_answer([], category="fallback", reason="Unrelated to the screen."))

    result = selector.select(_refinement(tmp_path), "something unrelated")

    assert result.scores == {}
    saved = json.loads((tmp_path / RELEVANT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert saved["segments"] == []
    assert saved["response_category"] == "fallback"


# --- the response category --------------------------------------------------


@pytest.mark.parametrize("category", ["teaching", "instructing", "guiding", "informative", "fallback"])
def test_the_category_confidence_and_reason_are_saved_with_the_segments(tmp_path, category):
    selector, _models = _selector(
        _answer([{"segment_id": 2, "score": 9}], category=category, confidence=0.75, reason="  Because\nof the verb. ")
    )

    result = selector.select(_refinement(tmp_path), "q")

    assert result.response_category == ResponseCategory(category)
    assert (result.confidence, result.category_reason) == (0.75, "Because of the verb.")
    saved = json.loads((tmp_path / RELEVANT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert saved["response_category"] == category
    assert saved["confidence"] == 0.75
    assert saved["category_reason"] == "Because of the verb."
    assert [s["id"] for s in saved["segments"]] == [2]


def test_confidence_is_clamped_to_zero_and_one(tmp_path):
    refinement = _refinement(tmp_path)
    high, _ = _selector(_answer([], confidence=7))
    low, _ = _selector(_answer([], confidence=-2))

    assert high.select(refinement, "q").confidence == 1.0
    assert low.select(refinement, "q").confidence == 0.0


def test_an_unknown_category_is_a_selection_error(tmp_path):
    selector, _models = _selector(_answer([], category="chatting"))

    with pytest.raises(SegmentSelectionError, match="invalid selection"):
        selector.select(_refinement(tmp_path), "q")


def test_the_prompt_defines_every_category():
    from app.refinement.gemini_selector import SYSTEM_INSTRUCTION

    for category in ResponseCategory:
        assert f"- {category.value}:" in SYSTEM_INSTRUCTION

    # Counting or reading what is on screen is informative, not a fallback.
    fallback_definition = SYSTEM_INSTRUCTION.split("- fallback:")[1].split("Tie-breaks")[0]
    assert "counting" not in fallback_definition and "How many" not in fallback_definition


def test_a_malformed_answer_is_a_selection_error(tmp_path):
    selector, _models = _selector("not json")

    with pytest.raises(SegmentSelectionError, match="invalid selection"):
        selector.select(_refinement(tmp_path), "q")


def test_a_failing_request_is_a_selection_error(tmp_path):
    selector, _models = _selector({}, error=RuntimeError("quota"))

    with pytest.raises(SegmentSelectionError, match="quota"):
        selector.select(_refinement(tmp_path), "q")


def test_a_missing_key_is_a_selection_error(tmp_path):
    selector = GeminiSegmentSelector(api_key="", model="fake-model")

    with pytest.raises(SegmentSelectionError, match="GEMINI_API_KEY"):
        selector.select(_refinement(tmp_path), "q")


# --- the coordinator --------------------------------------------------------


class SegmentedRefinement(ScreenRefinementProvider):
    def __init__(self, tmp_path):
        self._tmp_path = tmp_path

    def refine(self, capture, output_dir):
        return _refinement(self._tmp_path)


class StubSelector(SegmentSelector):
    def __init__(self, error=None):
        self._error = error
        self.calls = []

    def select(self, refinement, transcript):
        self.calls.append((refinement, transcript))
        if self._error is not None:
            raise self._error
        return SelectionResult(manifest_path="relevant.json", scores={2: 9}, seconds=0.5)


def _build(tmp_path, selector, refinement=True):
    hotkey = FakeHotkey()
    folder = tmp_path / "shot"
    folder.mkdir()
    coordinator = ApplicationCoordinator(
        hotkey=hotkey,
        recorder=StubRecorder(),
        transcription=StubTranscription(text="how many folders?"),
        capture=StubCapture(),
        thread_pool=SyncPool(),
        screenshot_dir=tmp_path / "plain",
        refinement=SegmentedRefinement(folder) if refinement else None,
        refinement_dir=tmp_path / "segments",
        segment_selector=selector,
    )
    return hotkey, coordinator


def test_segments_are_selected_against_the_transcript_and_the_interaction_ends(tmp_path, caplog):
    selector = StubSelector()
    hotkey, coordinator = _build(tmp_path, selector)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert len(selector.calls) == 1
    refinement, transcript = selector.calls[0]
    assert transcript == "how many folders?"
    assert refinement.segment_count == 3
    assert "Segment selection complete: 1 relevant" in caplog.text
    assert caplog.text.index("[Input] transcript") < caplog.text.index("Selecting relevant segments")
    assert caplog.text.index("Segment selection complete") < caplog.text.index("Interaction complete")
    assert coordinator.working is False


def test_a_failed_selection_still_completes_the_interaction(tmp_path, caplog):
    hotkey, coordinator = _build(tmp_path, StubSelector(error=RuntimeError("quota")))

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "Segment selection failed: quota" in caplog.text
    assert "Interaction complete" in caplog.text
    assert coordinator.working is False


def test_without_refinement_there_is_nothing_to_select(tmp_path, caplog):
    selector = StubSelector()
    hotkey, coordinator = _build(tmp_path, selector, refinement=False)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert selector.calls == []
    assert "Interaction complete" in caplog.text
    assert coordinator.working is False
