"""Preparing and executing a teaching plan, in that order.

The whole plan is prepared once - converted, then grounded - and only then does
execution start. These tests drive the real path the application takes once an
answer arrives (``response_to_teaching_plan`` -> ``ground_teaching_plan`` ->
``TeachingSequenceService``) and check the invariant that matters: every visual
target is prepared before the first step runs, and nothing is ever prepared
between steps.
"""

import logging

import pytest
from PySide6.QtWidgets import QApplication

from app.llm.response import (
    HeyArroResponse,
    ResponseContent,
    ResponseTone,
    TeachingMode,
)
from app.llm.response import TeachingPlan as ResponsePlan
from app.llm.response import TeachingStep as ResponseStep
from app.llm.response import VisualAction, VisualActionType, VisualTarget
from app.teaching import (
    TeachingSequenceService,
    response_to_teaching_plan,
)
from app.teaching.service import VisualTeachingService
from app.tts.provider import TTSProvider
from app.visual_grounding import (
    GroundingStatus,
    Rect,
    Refinement,
    VisualGrounder,
    ground_teaching_plan,
)

PNG = b"\x89PNG\r\n\x1a\n"
SIZE = (1000, 800)


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    yield QApplication.instance() or QApplication([])


# --- doubles ----------------------------------------------------------------


class SyncPool:
    """Runs workers inline so a whole teaching run can be asserted at once."""

    def start(self, runnable):
        runnable.run()

    def waitForDone(self, timeout=0):
        return True


class RecordingTTS(TTSProvider):
    """Records what was said, in the order it was said."""

    name = "recording"

    def __init__(self, events):
        self._events = events
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)
        self._events.append(("speak", text))

    def stop(self):
        return None


class RecordingOverlay(VisualTeachingService):
    """Records the steps the sequence asked it to show."""

    def __init__(self, events):
        self._events = events
        self.steps = []
        self.hide_calls = 0

    def prepare(self, capture):
        return None

    def show_step(self, step):
        self._events.append(("show", step.step_id))
        self.steps.append(step)

    def show_actions(self, actions):
        return None

    def hide(self):
        self._events.append(("hide", None))
        self.hide_calls += 1


class MovingGrounder(VisualGrounder):
    """Moves every target by a fixed offset, recording what it was asked for."""

    def __init__(self, events, shift=(50, 40), reject=()):
        self._events = events
        self._shift = shift
        self._reject = set(reject)
        self.calls = []

    def refine_target(self, request):
        self.calls.append(request)
        self._events.append(("ground", request.label))
        if request.label in self._reject:
            return Refinement(request.approximate, GroundingStatus.REFINEMENT_REJECTED)
        return Refinement(
            request.approximate.shifted(*self._shift), GroundingStatus.REFINED, 0.95
        )


class LandingGrounder(VisualGrounder):
    """Always answers with the same rectangle, whatever it was asked about."""

    def __init__(self, events, rect):
        self._events = events
        self._rect = rect
        self.calls = []

    def refine_target(self, request):
        self.calls.append(request)
        self._events.append(("ground", request.label))
        return Refinement(self._rect, GroundingStatus.REFINED, 0.95)


class ExplodingGrounder(VisualGrounder):
    """A grounder that cannot prepare anything at all."""

    def __init__(self, events=None):
        self._events = events

    def refine_target(self, request):
        raise RuntimeError("grounding exploded")


class TeachingRun:
    """The application's teaching path: convert, ground everything, then teach."""

    def __init__(self, make_grounder=None):
        self.events = []
        self.tts = RecordingTTS(self.events)
        self.overlay = RecordingOverlay(self.events)
        self.sequence = TeachingSequenceService(self.tts, self.overlay, SyncPool())
        self.grounder = make_grounder(self.events) if make_grounder else None
        self.executions = []

    def teach(self, answer):
        """Teach an answer, exactly as the coordinator does."""
        return self.teach_plan(response_to_teaching_plan(answer))

    def teach_plan(self, plan):
        prepared = ground_teaching_plan(self.grounder, plan, PNG, SIZE)
        self.executions.append(prepared)
        self.sequence.play(prepared)
        return prepared

    @property
    def kinds(self):
        return [event[0] for event in self.events]


# --- the answer under test --------------------------------------------------


def response(text="Here is the idea.", *steps):
    return HeyArroResponse(
        response=ResponseContent(text=text, tone=ResponseTone.NEUTRAL),
        teaching=ResponsePlan(mode=TeachingMode.GUIDED, steps=list(steps)),
    )


def rstep(instruction, *actions):
    return ResponseStep(instruction=instruction, visual_actions=list(actions))


def visual(kind, label, x, y, width=60, height=40):
    return VisualAction(
        type=VisualActionType(kind),
        target=VisualTarget(x=x, y=y, width=width, height=height, label=label),
    )


def three_step_response():
    """Three steps, five visual actions, four distinct targets."""
    return response(
        "Here is how matrix multiplication works.",
        rstep("Start with the rows of A.", visual("box", "Matrix A", 200, 150),
              visual("highlight", "Matrix A", 200, 150)),
        rstep("Then the columns of B.", visual("box", "Matrix B", 500, 350)),
        rstep("Multiply and place the result.",
              visual("point", "row one", 600, 400),
              visual("box", "result", 700, 500)),
    )


# --- preparation before execution -------------------------------------------


def test_every_target_is_grounded_before_the_first_step_runs():
    run = TeachingRun(make_grounder=MovingGrounder)

    run.teach(three_step_response())

    first_show = run.kinds.index("show")
    assert run.kinds.count("ground") == 4
    assert "ground" not in run.kinds[first_show:]


def test_nothing_is_grounded_during_execution():
    run = TeachingRun(make_grounder=MovingGrounder)

    run.teach(three_step_response())

    # Four distinct targets out of five actions: one request per distinct target,
    # all of them before the first step was shown.
    assert len(run.grounder.calls) == 4
    first_speak = run.kinds.index("speak")
    before_any_speech = [kind for kind in run.kinds[:first_speak] if kind != "hide"]
    assert before_any_speech == ["ground"] * 4
    assert "ground" not in run.kinds[first_speak:]


def test_every_target_is_collected_before_grounding_starts(caplog):
    run = TeachingRun(make_grounder=MovingGrounder)

    with caplog.at_level(logging.INFO):
        run.teach(three_step_response())

    assert "[TeachingGrounding] Total visual targets: 5" in caplog.text
    assert "[TeachingGrounding] Unique targets: 4" in caplog.text
    assert "[TeachingGrounding] Preparation complete" in caplog.text


def test_a_target_used_by_two_steps_is_grounded_once():
    run = TeachingRun(make_grounder=MovingGrounder)
    answer = response(
        "The same panel, twice.",
        rstep("Look at the panel.", visual("box", "Panel", 300, 200)),
        rstep("Now use it.", visual("highlight", "Panel", 300, 200)),
    )

    prepared = run.teach(answer)

    assert len(run.grounder.calls) == 1
    first = prepared.steps[0].visual_actions[0].target
    second = prepared.steps[1].visual_actions[0].target
    assert (first.x, first.y) == (second.x, second.y) == (350, 240)


def test_execution_receives_the_grounded_coordinates():
    run = TeachingRun(
        make_grounder=lambda events: LandingGrounder(
            events, Rect.from_center_size(250, 300, 60, 40)
        )
    )
    answer = response(
        "Look at the panel.",
        rstep("Look at the panel.", visual("box", "Panel", 100, 100)),
    )

    run.teach(answer)

    # The approximate target was (100,100); the overlay sees the refined one.
    target = run.overlay.steps[0].visual_actions[0].target
    assert (target.x, target.y) == (250, 300)
    assert (target.width, target.height) == (60, 40)


def test_a_target_that_cannot_be_refined_keeps_its_approximation():
    run = TeachingRun(make_grounder=lambda events: MovingGrounder(events, reject={"Unknown"}))
    answer = response(
        "Two panels.",
        rstep("The first one.", visual("box", "Known", 300, 200)),
        rstep("The other one.", visual("box", "Unknown", 700, 600)),
    )

    prepared = run.teach(answer)

    known = prepared.steps[0].visual_actions[0].target
    unknown = prepared.steps[1].visual_actions[0].target
    # The refined target moved...
    assert (known.x, known.y) == (350, 240)
    # ...and the rejected one is still teachable where the model thought it was.
    assert (unknown.x, unknown.y) == (700, 600)
    assert len(run.overlay.steps) == 2


def test_a_preparation_that_fails_never_reaches_execution():
    run = TeachingRun(make_grounder=ExplodingGrounder)

    with pytest.raises(RuntimeError):
        run.teach(three_step_response())

    # No partial lesson: nothing was shown, nothing was said.
    assert run.executions == []
    assert run.overlay.steps == []
    assert run.tts.spoken == []
    assert run.kinds == []


# --- execution --------------------------------------------------------------


def test_steps_run_in_plan_order():
    run = TeachingRun()

    run.teach(three_step_response())

    assert [event for event in run.events if event[0] == "show"] == [
        ("show", "step_1"),
        ("show", "step_2"),
        ("show", "step_3"),
    ]


def test_the_visual_of_a_step_is_up_before_its_sentence():
    run = TeachingRun(make_grounder=MovingGrounder)

    run.teach(three_step_response())

    # The introduction, then each step's visual before its own sentence.
    execution = run.kinds[run.kinds.index("speak"):]
    assert execution == [
        "speak",
        "show",
        "speak",
        "show",
        "speak",
        "show",
        "speak",
        "hide",
    ]


def test_the_plan_runs_introduction_steps_and_conclusion():
    run = TeachingRun()
    lesson = response_to_teaching_plan(
        response("Let's multiply two matrices.",
                 rstep("Start with a row.", visual("box", "Matrix A", 200, 150)),
                 rstep("Then a column.", visual("box", "Matrix B", 500, 350)))
    ).model_copy(update={"conclusion": "That is the whole product."})

    run.teach_plan(lesson)

    assert run.tts.spoken == [
        "Let's multiply two matrices.",
        "Start with a row.",
        "Then a column.",
        "That is the whole product.",
    ]
    assert [event for event in run.events if event[0] != "hide"] == [
        ("speak", "Let's multiply two matrices."),
        ("show", "step_1"),
        ("speak", "Start with a row."),
        ("show", "step_2"),
        ("speak", "Then a column."),
        ("speak", "That is the whole product."),
    ]


def test_an_answer_that_repeats_itself_is_spoken_once():
    run = TeachingRun()
    answer = response(
        "Open Settings.",
        rstep("Open Settings.", visual("box", "Settings", 200, 150)),
    )

    run.teach(answer)

    assert run.tts.spoken == ["Open Settings."]
    assert len(run.overlay.steps) == 1
