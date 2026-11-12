"""Response planning: the prompt per category, the plan file and its place in the flow.

Gemini is replaced by a fake SDK, so these tests need neither a key nor a network.
"""

import json
import logging
from types import SimpleNamespace

import pytest
from PIL import Image

from app.coordinator import ApplicationCoordinator
from app.planner import ResponsePlanner, ResponsePlanningError, create_response_planner
from app.planner.gemini_planner import GeminiResponsePlanner
from app.planner.manifest import PLAN_NAME
from app.planner.prompts import BASE_INSTRUCTION, CATEGORY_PROMPTS, system_instruction
from app.planner.provider import PlanResult
from app.refinement import ResponseCategory, SelectionResult
from tests.test_coordinator import (
    FakeHotkey,
    StubCapture,
    StubRecorder,
    StubTranscription,
    SyncPool,
    ask,
)
from tests.test_refinement import qt_app  # noqa: F401
from tests.test_segment_selection import SegmentedRefinement, StubSelector

PLANNED = [
    ResponseCategory.TEACHING,
    ResponseCategory.INSTRUCTING,
    ResponseCategory.GUIDING,
    ResponseCategory.INFORMATIVE,
]


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


def _folder(tmp_path, category="informative"):
    """A screenshot folder as selection leaves it: the image and relevant_ss_manifest.json."""
    Image.new("RGB", (1920, 1080), "white").save(tmp_path / "screenshot.png")
    (tmp_path / "segments").mkdir()
    Image.new("RGB", (100, 200), "black").save(tmp_path / "segments" / "segment-002-table.png")
    manifest = {
        "query": "how many folders are on my screen?",
        "response_category": category,
        "confidence": 0.95,
        "category_reason": "The user asks for a fact shown on screen.",
        "model": "selector-model",
        "screenshot": "screenshot.png",
        "source_manifest": "ss_segment_manifest.json",
        "total_segments": 3,
        "selected_count": 2,
        "selection_seconds": 1.0,
        "segments": [
            {
                "id": 2,
                "kind": "Table",
                "text": "app audio tests",
                "box": {"left": 0, "top": 30, "width": 100, "height": 200},
                "file": "segments/segment-002-table.png",
                "score": 9,
            },
            {
                "id": 1,
                "kind": "Title",
                "text": "Explorer",
                "box": {"left": 0, "top": 0, "width": 100, "height": 20},
                "file": None,
                "score": 4,
            },
        ],
    }
    path = tmp_path / "relevant_ss_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _selection(manifest_path, category=ResponseCategory.INFORMATIVE):
    return SelectionResult(manifest_path=str(manifest_path), scores={2: 9, 1: 4}, response_category=category)


def _step(
    purpose="Read the folders",
    segment_ids=(2,),
    action="highlight",
    key_points=("count", "names"),
    target="the folder list in the Explorer panel",
):
    return {
        "purpose": purpose,
        "target": target,
        "segment_ids": list(segment_ids),
        "visual_action": action,
        "key_points": list(key_points),
    }


def _answer(steps=None, goal="Tell the user how many folders there are.", caveats=()):
    return {
        "goal": goal,
        "approach": "Read the file tree.",
        "steps": [_step()] if steps is None else steps,
        "caveats": list(caveats),
    }


def _planner(answer, error=None):
    models = FakeModels(answer, error)
    return GeminiResponsePlanner(client=SimpleNamespace(models=models), model="fake-model"), models


# --- the prompts ------------------------------------------------------------


@pytest.mark.parametrize("category", PLANNED)
def test_every_planned_category_has_its_own_prompt(category):
    text = system_instruction(category)

    assert text.startswith(BASE_INSTRUCTION)
    assert f"Category: {category.value}." in text
    others = [other for other in PLANNED if other is not category]
    assert all(f"Category: {other.value}." not in text for other in others)


def test_the_prompts_differ_by_category():
    assert len({CATEGORY_PROMPTS[category] for category in PLANNED}) == len(PLANNED)


def test_a_fallback_question_has_no_prompt():
    assert ResponseCategory.FALLBACK not in CATEGORY_PROMPTS
    with pytest.raises(ValueError):
        system_instruction(ResponseCategory.FALLBACK)


def test_the_brief_asks_for_a_plan_and_not_the_answer():
    assert "Write a PLAN, not the answer" in BASE_INSTRUCTION
    assert "pixel box" in BASE_INSTRUCTION


# --- the planner ------------------------------------------------------------


def test_the_plan_is_saved_next_to_the_manifest(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer())

    result = planner.plan(_selection(manifest))

    assert result.plan_path == str(tmp_path / PLAN_NAME)
    assert (result.response_category, result.step_count) == (ResponseCategory.INFORMATIVE, 1)
    plan = json.loads((tmp_path / PLAN_NAME).read_text(encoding="utf-8"))
    assert plan["query"] == "how many folders are on my screen?"
    assert plan["response_category"] == "informative"
    assert plan["screenshot"] == "screenshot.png"
    assert plan["screenshot_size"] == {"width": 1920, "height": 1080}
    assert plan["source_manifest"] == "relevant_ss_manifest.json"
    assert plan["model"] == "fake-model"
    assert plan["goal"] == "Tell the user how many folders there are."


def test_a_step_carries_the_boxes_of_its_segments(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer([_step(segment_ids=[2, 1, 2])]))

    planner.plan(_selection(manifest))

    step = json.loads((tmp_path / PLAN_NAME).read_text(encoding="utf-8"))["steps"][0]
    assert step["order"] == 1
    assert step["visual_action"] == "highlight"
    assert step["target"] == "the folder list in the Explorer panel"
    assert step["key_points"] == ["count", "names"]
    assert step["segment_ids"] == [2, 1]
    assert step["targets"] == [
        {"id": 2, "kind": "Table", "box": {"left": 0, "top": 30, "width": 100, "height": 200}},
        {"id": 1, "kind": "Title", "box": {"left": 0, "top": 0, "width": 100, "height": 20}},
    ]


def test_a_step_without_a_segment_keeps_its_target_description(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer([_step(segment_ids=[], target="  the Windows taskbar\n search box ")]))

    planner.plan(_selection(manifest))

    step = json.loads((tmp_path / PLAN_NAME).read_text(encoding="utf-8"))["steps"][0]
    assert step["segment_ids"] == []
    assert step["targets"] == []
    assert step["target"] == "the Windows taskbar search box"


def test_a_step_without_a_target_is_an_error(tmp_path):
    manifest = _folder(tmp_path)
    step = _step()
    del step["target"]
    planner, _models = _planner(_answer([step]))

    with pytest.raises(ResponsePlanningError, match="invalid plan"):
        planner.plan(_selection(manifest))


def test_steps_are_numbered_in_order(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer([_step("first"), _step("second"), _step("third")]))

    planner.plan(_selection(manifest))

    steps = json.loads((tmp_path / PLAN_NAME).read_text(encoding="utf-8"))["steps"]
    assert [(step["order"], step["purpose"]) for step in steps] == [(1, "first"), (2, "second"), (3, "third")]


def test_unknown_segment_ids_are_dropped(tmp_path, caplog):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer([_step(segment_ids=[2, 99])]))

    with caplog.at_level(logging.INFO):
        planner.plan(_selection(manifest))

    step = json.loads((tmp_path / PLAN_NAME).read_text(encoding="utf-8"))["steps"][0]
    assert step["segment_ids"] == [2]
    assert "unknown segment id 99" in caplog.text


def test_caveats_are_kept(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer(caveats=["The tree is partly cut off."]))

    planner.plan(_selection(manifest))

    plan = json.loads((tmp_path / PLAN_NAME).read_text(encoding="utf-8"))
    assert plan["caveats"] == ["The tree is partly cut off."]


@pytest.mark.parametrize("category", PLANNED)
def test_the_request_has_the_screenshot_the_segments_and_the_category_prompt(tmp_path, category):
    manifest = _folder(tmp_path, category.value)
    planner, models = _planner(_answer())

    planner.plan(_selection(manifest, category))

    request = models.calls[0]
    parts = request["contents"].parts
    assert parts[0].inline_data.data == (tmp_path / "screenshot.png").read_bytes()
    assert parts[0].inline_data.mime_type == "image/png"
    prompt = parts[1].text
    assert "how many folders are on my screen?" in prompt
    assert f"Response category: {category.value}" in prompt
    assert "1920x1080" in prompt
    assert '"box":{"left":0,"top":30,"width":100,"height":200}' in prompt
    assert '"score":9' in prompt
    assert '"file"' not in prompt
    assert request["config"].system_instruction == system_instruction(category)


def test_each_segment_crop_is_sent_after_the_prompt_with_its_id(tmp_path):
    manifest = _folder(tmp_path)
    planner, models = _planner(_answer())

    planner.plan(_selection(manifest))

    parts = models.calls[0]["contents"].parts
    assert len(parts) == 4  # screenshot, prompt, then the label and image of segment 2
    assert parts[2].text == "Segment 2:"
    assert parts[3].inline_data.data == (tmp_path / "segments" / "segment-002-table.png").read_bytes()
    assert parts[3].inline_data.mime_type == "image/png"
    assert "1 of 2 available" in parts[1].text


def test_a_segment_without_a_readable_crop_is_sent_as_text_only(tmp_path, caplog):
    manifest = _folder(tmp_path)
    (tmp_path / "segments" / "segment-002-table.png").unlink()
    planner, models = _planner(_answer())

    with caplog.at_level(logging.INFO):
        planner.plan(_selection(manifest))

    assert len(models.calls[0]["contents"].parts) == 2
    assert "could not read the crop of segment 2" in caplog.text
    assert (tmp_path / PLAN_NAME).exists()


def test_the_brief_mentions_the_close_up_images():
    assert "close-up image of each segment" in BASE_INSTRUCTION


def test_the_brief_asks_for_a_target_description_on_every_step():
    assert "target (a short description of the exact element or area" in BASE_INSTRUCTION
    assert "even when segment_ids is empty" in BASE_INSTRUCTION


def test_the_screenshot_is_the_main_reference_and_segments_only_support_it():
    assert "The main reference is the full screenshot" in BASE_INSTRUCTION
    assert "supporting references" in BASE_INSTRUCTION
    assert "Use a segment only when it improves the plan" in BASE_INSTRUCTION
    assert "leave its segment_ids empty" in BASE_INSTRUCTION


def test_a_fallback_question_is_not_planned(tmp_path):
    manifest = _folder(tmp_path, "fallback")
    planner, models = _planner(_answer())

    with pytest.raises(ResponsePlanningError, match="fallback"):
        planner.plan(_selection(manifest, ResponseCategory.FALLBACK))

    assert models.calls == []
    assert not (tmp_path / PLAN_NAME).exists()


def test_a_plan_without_steps_is_an_error(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer(steps=[]))

    with pytest.raises(ResponsePlanningError, match="without steps"):
        planner.plan(_selection(manifest))

    assert not (tmp_path / PLAN_NAME).exists()


def test_a_malformed_answer_is_an_error(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner("not json")

    with pytest.raises(ResponsePlanningError, match="invalid plan"):
        planner.plan(_selection(manifest))


def test_an_unknown_visual_action_is_an_error(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer([_step(action="sparkle")]))

    with pytest.raises(ResponsePlanningError, match="invalid plan"):
        planner.plan(_selection(manifest))


def test_a_failing_request_is_an_error(tmp_path):
    manifest = _folder(tmp_path)
    planner, _models = _planner(_answer(), error=RuntimeError("quota"))

    with pytest.raises(ResponsePlanningError, match="quota"):
        planner.plan(_selection(manifest))


def test_a_missing_key_is_an_error(tmp_path):
    manifest = _folder(tmp_path)
    planner = GeminiResponsePlanner(api_key="", model="fake-model")

    with pytest.raises(ResponsePlanningError, match="GEMINI_API_KEY"):
        planner.plan(_selection(manifest))


def test_a_missing_manifest_is_an_error(tmp_path):
    planner, _models = _planner(_answer())

    with pytest.raises(ResponsePlanningError, match="relevant_ss_manifest.json"):
        planner.plan(_selection(tmp_path / "relevant_ss_manifest.json"))


def test_a_missing_screenshot_is_an_error(tmp_path):
    manifest = _folder(tmp_path)
    (tmp_path / "screenshot.png").unlink()
    planner, _models = _planner(_answer())

    with pytest.raises(ResponsePlanningError, match="screenshot"):
        planner.plan(_selection(manifest))


def test_the_factory_is_off_without_a_key(monkeypatch):
    monkeypatch.setattr("app.planner.config.GEMINI_API_KEY", None)

    assert create_response_planner() is None


def test_the_factory_is_off_when_disabled(monkeypatch):
    monkeypatch.setattr("app.planner.config.RESPONSE_PLANNING_ENABLED", False)

    assert create_response_planner() is None


def test_the_factory_builds_a_planner_with_a_key(monkeypatch):
    monkeypatch.setattr("app.planner.config.GEMINI_API_KEY", "key")
    monkeypatch.setattr("app.planner.gemini_planner.config.GEMINI_API_KEY", "key")

    assert isinstance(create_response_planner(), ResponsePlanner)


# --- the coordinator --------------------------------------------------------


class StubPlanner(ResponsePlanner):
    def __init__(self, error=None):
        self._error = error
        self.calls = []

    def plan(self, selection):
        self.calls.append(selection)
        if self._error is not None:
            raise self._error
        return PlanResult(
            plan_path="response_plan.json",
            response_category=selection.response_category,
            step_count=3,
            seconds=0.7,
        )


class CategorisingSelector(StubSelector):
    def __init__(self, category):
        super().__init__()
        self._category = category

    def select(self, refinement, transcript):
        super().select(refinement, transcript)
        return SelectionResult(
            manifest_path="relevant.json", scores={2: 9}, seconds=0.5, response_category=self._category
        )


def _build(tmp_path, category, planner):
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
        refinement=SegmentedRefinement(folder),
        refinement_dir=tmp_path / "segments",
        segment_selector=CategorisingSelector(category),
        response_planner=planner,
    )
    return hotkey, coordinator


@pytest.mark.parametrize("category", PLANNED)
def test_a_plan_is_made_after_the_segments_are_selected(tmp_path, caplog, category):
    planner = StubPlanner()
    hotkey, coordinator = _build(tmp_path, category, planner)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert [call.response_category for call in planner.calls] == [category]
    assert f"Planning a {category.value} response" in caplog.text
    assert "Response plan complete: 3 step(s)" in caplog.text
    assert caplog.text.index("Segment selection complete") < caplog.text.index("Response plan complete")
    assert caplog.text.index("Response plan complete") < caplog.text.index("Interaction complete")
    assert coordinator.working is False


def test_a_fallback_question_is_not_planned_in_the_flow(tmp_path, caplog):
    planner = StubPlanner()
    hotkey, coordinator = _build(tmp_path, ResponseCategory.FALLBACK, planner)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert planner.calls == []
    assert "no response plan is needed" in caplog.text
    assert "Interaction complete" in caplog.text
    assert coordinator.working is False


def test_a_failed_plan_still_completes_the_interaction(tmp_path, caplog):
    hotkey, coordinator = _build(tmp_path, ResponseCategory.TEACHING, StubPlanner(error=RuntimeError("quota")))

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "Response planning failed: quota" in caplog.text
    assert "Interaction complete" in caplog.text
    assert coordinator.working is False


def test_without_a_planner_the_flow_ends_after_selection(tmp_path, caplog):
    hotkey, coordinator = _build(tmp_path, ResponseCategory.TEACHING, None)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "Planning a" not in caplog.text
    assert "Interaction complete" in caplog.text
    assert coordinator.working is False
