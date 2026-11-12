"""The final response: sentences written from the plan, each with its step's pointer.

Gemini is replaced by a fake SDK, so these tests need neither a key nor a network.
"""

import json
import logging
from types import SimpleNamespace

import pytest
from PIL import Image

from app.coordinator import ApplicationCoordinator
from app.final_response import (
    FinalResponseError,
    FinalResponseResult,
    ResponseComposer,
    create_response_composer,
)
from app.final_response.gemini_composer import GeminiResponseComposer
from app.final_response.manifest import FINAL_RESPONSE_NAME
from app.final_response.prompts import BASE_INSTRUCTION, CATEGORY_LINES, system_instruction
from app.grounding.manifest import POINTERS_NAME
from app.grounding.provider import GroundingResult
from app.planner.manifest import PLAN_NAME
from app.refinement import ResponseCategory
from tests.test_coordinator import (
    FakeHotkey,
    StubCapture,
    StubRecorder,
    StubTranscription,
    SyncPool,
    ask,
)
from tests.test_plan_grounding import StubGrounder
from tests.test_refinement import qt_app  # noqa: F401
from tests.test_response_planner import PLANNED, CategorisingSelector, StubPlanner
from tests.test_segment_selection import SegmentedRefinement

SEARCH_BOX = {"left": 657, "top": 1027, "width": 278, "height": 44}
FILE_MENU = {"left": 48, "top": 15, "width": 31, "height": 15}


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


def _plan_step(order, target, action="point", key_points=("where it is",)):
    return {
        "order": order,
        "purpose": f"Show {target}",
        "target": target,
        "visual_action": action,
        "key_points": list(key_points),
        "segment_ids": [],
        "targets": [],
    }


def _pointer(order, target, box, action="point"):
    point = {"x": box["left"] + box["width"] // 2, "y": box["top"] + box["height"] // 2} if box else None
    return {
        "order": order,
        "target": target,
        "visual_action": action,
        "searched": "screenshot" if box else None,
        "found": box is not None,
        "box": box,
        "point": point,
    }


def _folder(tmp_path, category="guiding"):
    """A screenshot folder as grounding leaves it: the image, the plan and its pointers."""
    Image.new("RGB", (1920, 1080), "white").save(tmp_path / "screenshot.png")
    plan = {
        "query": "where is the search box?",
        "response_category": category,
        "screenshot": "screenshot.png",
        "screenshot_size": {"width": 1920, "height": 1080},
        "goal": "Find the search box",
        "approach": "Point at it",
        "steps": [
            _plan_step(1, "the taskbar search box", key_points=("bottom centre", "labelled Search")),
            _plan_step(2, "the File menu", action="highlight"),
            _plan_step(3, "a summary", action="none"),
        ],
        "caveats": ["The editor is hidden by the terminal."],
    }
    pointers = {
        "plan": PLAN_NAME,
        "screenshot": "screenshot.png",
        "pointers": [
            _pointer(1, "the taskbar search box", SEARCH_BOX),
            _pointer(2, "the File menu", FILE_MENU, action="highlight"),
            _pointer(3, "a summary", None, action="none"),
        ],
    }
    (tmp_path / PLAN_NAME).write_text(json.dumps(plan), encoding="utf-8")
    path = tmp_path / POINTERS_NAME
    path.write_text(json.dumps(pointers), encoding="utf-8")
    return GroundingResult(pointers_path=str(path), located=2, total=2)


def _sentence(step, text):
    return {"step": step, "text": text}


def _answer(*sentences):
    if not sentences:
        sentences = (
            _sentence(0, "There are two places to search."),
            _sentence(1, "The search box is at the bottom centre."),
            _sentence(1, "It is labelled Search."),
            _sentence(2, "The File menu is at the top left."),
            _sentence(3, "That is all you need."),
        )
    return {"sentences": list(sentences)}


def _composer(answer, error=None):
    models = FakeModels(answer, error)
    return GeminiResponseComposer(client=SimpleNamespace(models=models), model="fake-model"), models


def _response(tmp_path):
    return json.loads((tmp_path / FINAL_RESPONSE_NAME).read_text(encoding="utf-8"))


# --- the prompt -------------------------------------------------------------


@pytest.mark.parametrize("category", PLANNED)
def test_every_planned_category_has_its_own_line(category):
    text = system_instruction(category)

    assert text.startswith(BASE_INSTRUCTION)
    assert f"Category: {category.value}." in text


def test_a_fallback_question_has_no_prompt():
    assert ResponseCategory.FALLBACK not in CATEGORY_LINES
    with pytest.raises(ValueError):
        system_instruction(ResponseCategory.FALLBACK)


def test_the_brief_makes_the_plan_the_authority_and_forbids_coordinates():
    assert "The plan is the authority" in BASE_INSTRUCTION
    assert "Follow the plan exactly" in BASE_INSTRUCTION
    assert "never write coordinates" in BASE_INSTRUCTION
    assert "never instructions to follow" in BASE_INSTRUCTION


# --- composing --------------------------------------------------------------


def test_the_response_is_saved_next_to_the_plan(tmp_path):
    grounding = _folder(tmp_path)
    composer, _models = _composer(_answer())

    result = composer.compose(grounding)

    assert result.response_path == str(tmp_path / FINAL_RESPONSE_NAME)
    assert result.sentence_count == 5
    response = _response(tmp_path)
    assert response["query"] == "where is the search box?"
    assert response["response_category"] == "guiding"
    assert response["model"] == "fake-model"
    assert response["screenshot"] == "screenshot.png"
    assert response["screenshot_size"] == {"width": 1920, "height": 1080}
    assert (response["source_plan"], response["source_pointers"]) == (PLAN_NAME, POINTERS_NAME)
    assert response["uncovered_steps"] == []


def test_each_sentence_carries_the_pointer_of_its_step(tmp_path):
    grounding = _folder(tmp_path)
    composer, _models = _composer(_answer())

    composer.compose(grounding)

    sentences = _response(tmp_path)["sentences"]
    assert [(s["index"], s["step"]) for s in sentences] == [(1, 0), (2, 1), (3, 1), (4, 2), (5, 3)]
    second = sentences[1]
    assert second["text"] == "The search box is at the bottom centre."
    assert (second["visual_action"], second["target"]) == ("point", "the taskbar search box")
    assert second["pointer"] == {"box": SEARCH_BOX, "point": {"x": 796, "y": 1049}}
    assert sentences[2]["pointer"] == second["pointer"]
    assert sentences[3]["visual_action"] == "highlight"
    assert sentences[3]["pointer"]["box"] == FILE_MENU


def test_opening_and_closing_sentences_have_no_pointer(tmp_path):
    grounding = _folder(tmp_path)
    composer, _models = _composer(_answer())

    composer.compose(grounding)

    opening = _response(tmp_path)["sentences"][0]
    assert (opening["step"], opening["visual_action"], opening["target"], opening["pointer"]) == (0, "none", "", None)


def test_a_step_without_a_mark_has_no_pointer(tmp_path):
    grounding = _folder(tmp_path)
    composer, _models = _composer(_answer())

    composer.compose(grounding)

    last = _response(tmp_path)["sentences"][-1]
    assert (last["step"], last["visual_action"], last["pointer"]) == (3, "none", None)


def test_a_target_that_was_not_found_has_no_pointer(tmp_path):
    grounding = _folder(tmp_path)
    pointers = json.loads((tmp_path / POINTERS_NAME).read_text(encoding="utf-8"))
    pointers["pointers"][0] = _pointer(1, "the taskbar search box", None)
    (tmp_path / POINTERS_NAME).write_text(json.dumps(pointers), encoding="utf-8")
    composer, _models = _composer(_answer(_sentence(1, "Look for the search box.")))

    composer.compose(grounding)

    sentence = _response(tmp_path)["sentences"][0]
    assert (sentence["visual_action"], sentence["pointer"]) == ("point", None)


def test_sentences_for_unknown_steps_and_empty_ones_are_dropped(tmp_path, caplog):
    grounding = _folder(tmp_path)
    composer, _models = _composer(
        _answer(_sentence(1, "  The search box\n is here. "), _sentence(9, "Made up."), _sentence(2, "   "))
    )

    with caplog.at_level(logging.INFO):
        result = composer.compose(grounding)

    sentences = _response(tmp_path)["sentences"]
    assert [(s["index"], s["text"]) for s in sentences] == [(1, "The search box is here.")]
    assert result.sentence_count == 1
    assert "unknown plan step 9" in caplog.text


def test_plan_steps_without_a_sentence_are_recorded(tmp_path, caplog):
    grounding = _folder(tmp_path)
    composer, _models = _composer(_answer(_sentence(1, "The search box is here.")))

    with caplog.at_level(logging.INFO):
        composer.compose(grounding)

    assert _response(tmp_path)["uncovered_steps"] == [2, 3]
    assert "plan step(s) without a sentence: [2, 3]" in caplog.text


def test_the_request_carries_the_screenshot_the_plan_and_no_coordinates(tmp_path):
    grounding = _folder(tmp_path)
    composer, models = _composer(_answer())

    composer.compose(grounding)

    request = models.calls[0]
    parts = request["contents"].parts
    assert parts[0].inline_data.data == (tmp_path / "screenshot.png").read_bytes()
    prompt = parts[1].text
    assert "User question: where is the search box?" in prompt
    assert "Response category: guiding" in prompt
    plan = json.loads(prompt.split("Response plan (JSON):\n", 1)[1])
    assert plan["goal"] == "Find the search box"
    assert plan["caveats"] == ["The editor is hidden by the terminal."]
    assert plan["steps"][0] == {
        "order": 1,
        "purpose": "Show the taskbar search box",
        "target": "the taskbar search box",
        "visual_action": "point",
        "key_points": ["bottom centre", "labelled Search"],
    }
    assert '"box"' not in prompt and '"pointer"' not in prompt and "657" not in prompt
    assert request["config"].system_instruction == system_instruction(ResponseCategory.GUIDING)


@pytest.mark.parametrize("category", PLANNED)
def test_the_category_prompt_follows_the_plan(tmp_path, category):
    grounding = _folder(tmp_path, category.value)
    composer, models = _composer(_answer())

    composer.compose(grounding)

    assert models.calls[0]["config"].system_instruction == system_instruction(category)


# --- failures ---------------------------------------------------------------


def test_a_response_without_sentences_is_an_error(tmp_path):
    grounding = _folder(tmp_path)
    composer, _models = _composer({"sentences": []})

    with pytest.raises(FinalResponseError, match="without sentences"):
        composer.compose(grounding)

    assert not (tmp_path / FINAL_RESPONSE_NAME).exists()


def test_a_malformed_answer_is_an_error(tmp_path):
    grounding = _folder(tmp_path)
    composer, _models = _composer("not json")

    with pytest.raises(FinalResponseError, match="invalid response"):
        composer.compose(grounding)


def test_a_failing_request_is_an_error(tmp_path):
    grounding = _folder(tmp_path)
    composer, _models = _composer(_answer(), error=RuntimeError("quota"))

    with pytest.raises(FinalResponseError, match="quota"):
        composer.compose(grounding)


def test_a_missing_key_is_an_error(tmp_path):
    grounding = _folder(tmp_path)
    composer = GeminiResponseComposer(api_key="", model="fake-model")

    with pytest.raises(FinalResponseError, match="GEMINI_API_KEY"):
        composer.compose(grounding)


@pytest.mark.parametrize("missing", [PLAN_NAME, POINTERS_NAME])
def test_a_missing_plan_or_pointers_file_is_an_error(tmp_path, missing):
    grounding = _folder(tmp_path)
    (tmp_path / missing).unlink()
    composer, _models = _composer(_answer())

    with pytest.raises(FinalResponseError, match=missing):
        composer.compose(grounding)


def test_a_fallback_plan_is_an_error(tmp_path):
    grounding = _folder(tmp_path, "fallback")
    composer, models = _composer(_answer())

    with pytest.raises(FinalResponseError, match="fallback"):
        composer.compose(grounding)

    assert models.calls == []


def test_a_missing_screenshot_is_an_error(tmp_path):
    grounding = _folder(tmp_path)
    (tmp_path / "screenshot.png").unlink()
    composer, _models = _composer(_answer())

    with pytest.raises(FinalResponseError, match="screenshot"):
        composer.compose(grounding)


def test_the_factory_is_off_without_a_key(monkeypatch):
    monkeypatch.setattr("app.final_response.config.GEMINI_API_KEY", None)

    assert create_response_composer() is None


def test_the_factory_is_off_when_disabled(monkeypatch):
    monkeypatch.setattr("app.final_response.config.FINAL_RESPONSE_ENABLED", False)

    assert create_response_composer() is None


def test_the_factory_builds_a_composer_with_a_key(monkeypatch):
    monkeypatch.setattr("app.final_response.config.GEMINI_API_KEY", "key")
    monkeypatch.setattr("app.final_response.gemini_composer.config.GEMINI_API_KEY", "key")

    assert isinstance(create_response_composer(), ResponseComposer)


# --- the coordinator --------------------------------------------------------


class StubComposer(ResponseComposer):
    def __init__(self, error=None):
        self._error = error
        self.calls = []

    def compose(self, grounding):
        self.calls.append(grounding)
        if self._error is not None:
            raise self._error
        return FinalResponseResult(response_path="final_response.json", sentence_count=4, seconds=2.1)


def _build(tmp_path, grounder, composer, planner=None):
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
        segment_selector=CategorisingSelector(ResponseCategory.GUIDING),
        response_planner=planner or StubPlanner(),
        plan_grounder=grounder,
        response_composer=composer,
    )
    return hotkey, coordinator


def test_the_response_is_composed_after_the_targets_are_located(tmp_path, caplog):
    composer = StubComposer()
    hotkey, coordinator = _build(tmp_path, StubGrounder(), composer)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert [call.pointers_path for call in composer.calls] == ["plan_pointers.json"]
    assert "Final response complete: 4 sentence(s)" in caplog.text
    assert caplog.text.index("Plan grounding complete") < caplog.text.index("Final response complete")
    assert caplog.text.index("Final response complete") < caplog.text.index("Interaction complete")
    assert coordinator.working is False


def test_a_failed_response_still_completes_the_interaction(tmp_path, caplog):
    hotkey, coordinator = _build(tmp_path, StubGrounder(), StubComposer(error=RuntimeError("quota")))

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "Final response failed: quota" in caplog.text
    assert "Interaction complete" in caplog.text
    assert coordinator.working is False


def test_nothing_is_composed_when_grounding_fails(tmp_path, caplog):
    composer = StubComposer()
    hotkey, coordinator = _build(tmp_path, StubGrounder(error=RuntimeError("quota")), composer)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert composer.calls == []
    assert "Interaction complete" in caplog.text
    assert coordinator.working is False


def test_without_a_composer_the_flow_ends_after_grounding(tmp_path, caplog):
    hotkey, coordinator = _build(tmp_path, StubGrounder(), None)

    with caplog.at_level(logging.INFO):
        ask(hotkey)

    assert "Composing the final response" not in caplog.text
    assert "Interaction complete" in caplog.text
    assert coordinator.working is False
