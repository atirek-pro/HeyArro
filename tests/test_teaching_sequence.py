"""The teaching sequence: visuals render first, then the matching sentence."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

import app.teaching.sequence as sequence_module
from app.capture.provider import ScreenCaptureResult
from app.llm.response import (
    VisualAction,
    VisualActionType,
    VisualTarget,
)
from app.teaching import (
    OverlayVisualTeachingService,
    ScreenGeometry,
    TeachingPlan,
    TeachingSequenceService,
    TeachingStep,
    VisualTeachingService,
    answer_only_teaching_plan,
    build_plan_utterances,
)
from app.tts.provider import TTSError, TTSProvider


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    yield QApplication.instance() or QApplication([])


# --- test doubles -----------------------------------------------------------


def make_capture(width, height):
    return ScreenCaptureResult(
        image=b"\x89PNG\r\n\x1a\n", monitor_index=1, width=width, height=height, timestamp=0.0
    )


def step(explanation, visuals=()):
    """One step for ``plan()``: what Arro says, and what is shown while he says it."""
    return (explanation, list(visuals))


def plan(*specs, introduction="A short opening.", conclusion=None):
    """Build the teaching plan a prepared run would hand to the sequence."""
    return TeachingPlan(
        objective="Understand this.",
        introduction=introduction,
        steps=[
            TeachingStep(
                step_id=f"step_{index}",
                order=index,
                objective=explanation,
                explanation=explanation,
                visual_actions=visuals,
            )
            for index, (explanation, visuals) in enumerate(specs, start=1)
        ],
        conclusion=conclusion,
    )


def show(kind, x, y, width=None, height=None, label=""):
    return VisualAction(
        type=VisualActionType(kind),
        target=VisualTarget(x=x, y=y, width=width, height=height, label=label),
    )


class EventTTS(TTSProvider):
    """Records speech into a shared event log and can fail on demand."""

    name = "event"

    def __init__(self, events, failures=()):
        self._events = events
        self._failures = set(failures)
        self.stop_calls = 0

    def speak(self, text):
        self._events.append(("speak", text))
        if text in self._failures:
            raise TTSError(f"could not speak: {text}")

    def stop(self):
        self.stop_calls += 1


class EventOverlay:
    """A fake overlay view that records what the service asked it to draw."""

    def __init__(self, events):
        self._events = events
        self.shown = []
        self.clear_calls = 0

    def show_actions(self, primitives):
        snapshot = [(p.kind, p.x, p.y, p.width, p.height, p.label) for p in primitives]
        self.shown.append(snapshot)
        for kind, _x, _y, _w, _h, label in snapshot:
            self._events.append(("show", label or kind))

    def clear(self):
        self.clear_calls += 1
        self._events.append(("clear", None))


class QueuedPool:
    """Queues workers so the test runs each speech explicitly, one at a time."""

    def __init__(self):
        self.queue = []

    def start(self, runnable):
        self.queue.append(runnable)

    def pending(self):
        return len(self.queue)

    def run_next(self):
        assert self.queue, "no worker is waiting to run"
        self.queue.pop(0).run()

    def waitForDone(self, timeout=0):
        return True


class ExplodingTeaching(VisualTeachingService):
    """A teaching service whose overlay always fails to render."""

    def __init__(self):
        self.hide_calls = 0

    def prepare(self, capture):
        return None

    def show_step(self, step):
        raise RuntimeError("overlay exploded")

    def show_actions(self, actions):
        raise RuntimeError("overlay exploded")

    def hide(self):
        self.hide_calls += 1


def make_sequence(events, tts=None, size=1000):
    tts = tts if tts is not None else EventTTS(events)
    overlay = EventOverlay(events)
    pool = QueuedPool()
    screen = ScreenGeometry(0, 0, size, size)
    teaching = OverlayVisualTeachingService(overlay, screen)
    teaching.prepare(make_capture(size, size))

    sequence = TeachingSequenceService(tts, teaching, pool)
    finished = []
    failed = []
    sequence.finished.connect(lambda: finished.append(True))
    sequence.failed.connect(failed.append)
    return sequence, tts, overlay, pool, finished, failed


def visual_and_speech(events):
    """Only the events that prove visual/sentence ordering."""
    return [event for event in events if event[0] in ("show", "speak")]


def shown_labels(events):
    return [event[1] for event in events if event[0] == "show"]


# --- the utterance plan -----------------------------------------------------


def test_utterances_run_from_the_introduction_to_the_conclusion():
    lesson = plan(
        step("One."),
        step("Two."),
        introduction="Let's begin.",
        conclusion="That is the idea.",
    )

    assert build_plan_utterances(lesson) == [
        ("Let's begin.", None),
        ("One.", lesson.steps[0]),
        ("Two.", lesson.steps[1]),
        ("That is the idea.", None),
    ]


def test_utterances_drop_a_blank_introduction_and_conclusion():
    # The plan model already rejects blank text, so this guards the defensive
    # path with a stubbed plan rather than an invalid one.
    lesson = plan(step("One."))
    stub = SimpleNamespace(introduction="   ", steps=lesson.steps, conclusion=None)

    assert build_plan_utterances(stub) == [("One.", lesson.steps[0])]


def test_every_step_is_yielded_with_its_own_step():
    lesson = plan(step("One."), step("Two."), introduction="Intro.")

    utterances = build_plan_utterances(lesson)

    assert utterances[1][1] is lesson.steps[0]
    assert utterances[2][1] is lesson.steps[1]


# --- immediate visual rendering ---------------------------------------------


def test_the_first_visual_is_rendered_before_its_own_sentence():
    events = []
    sequence, _tts, _overlay, pool, _finished, _failed = make_sequence(events)

    sequence.play(plan(step("Start here.", [show("highlight", 100, 100, 200, 60, "row")])))

    # The introduction is a sentence of its own, so nothing is shown for it.
    assert visual_and_speech(events) == []
    assert pool.pending() == 1

    pool.run_next()  # the introduction, then the step's visual

    # The visual is up, and the step's sentence has not been spoken yet.
    assert visual_and_speech(events) == [
        ("speak", "A short opening."),
        ("show", "row"),
    ]
    assert pool.pending() == 1


def test_each_visual_is_rendered_before_its_own_sentence():
    events = []
    steps = [
        step("First.", [show("highlight", 100, 100, 200, 60, "row")]),
        step("Second.", [show("box", 300, 300, 80, 80, "column")]),
    ]
    sequence, _tts, _overlay, pool, _finished, _failed = make_sequence(events)

    sequence.play(plan(*steps))
    assert visual_and_speech(events) == []

    pool.run_next()  # the introduction, then step one's visual
    assert visual_and_speech(events) == [
        ("speak", "A short opening."),
        ("show", "row"),
    ]

    pool.run_next()
    assert visual_and_speech(events) == [
        ("speak", "A short opening."),
        ("show", "row"),
        ("speak", "First."),
        ("show", "column"),
    ]

    pool.run_next()
    assert visual_and_speech(events) == [
        ("speak", "A short opening."),
        ("show", "row"),
        ("speak", "First."),
        ("show", "column"),
        ("speak", "Second."),
    ]


def test_a_visual_stays_up_for_the_whole_sentence():
    events = []
    sequence, _tts, _overlay, pool, _finished, _failed = make_sequence(events)

    sequence.play(plan(step("Look at this.", [show("circle", 10, 10, 40, 40, "Node")])))
    pool.run_next()  # the introduction, then the step's visual

    first_show = next(index for index, event in enumerate(events) if event[0] == "show")
    assert not any(event[0] == "clear" for event in events[first_show:])


# --- single step ------------------------------------------------------------


def test_single_step_shows_then_speaks_then_hides():
    events = []
    sequence, _tts, overlay, pool, finished, _failed = make_sequence(events)

    sequence.play(plan(step("Click Settings", [show("point", 100, 100, label="Settings")])))
    pool.run_next()  # the introduction
    pool.run_next()  # the step

    assert visual_and_speech(events) == [
        ("speak", "A short opening."),
        ("show", "Settings"),
        ("speak", "Click Settings"),
    ]
    assert finished == [True]
    assert sequence.active is False
    assert events[-1] == ("clear", None)
    assert overlay.clear_calls >= 1


# --- several visuals in one step --------------------------------------------


def test_one_step_can_show_several_visuals():
    events = []
    visuals = [
        show("circle", 200, 100, 40, 40, "1"),
        show("circle", 400, 300, 40, 40, "5"),
        show("point", 400, 300),
    ]
    sequence, _tts, overlay, pool, _finished, _failed = make_sequence(events)

    sequence.play(plan(step("Multiply these.", visuals)))
    pool.run_next()  # the introduction

    # All three are applied together, before the sentence.
    assert visual_and_speech(events) == [
        ("speak", "A short opening."),
        ("show", "1"),
        ("show", "5"),
        ("show", "point"),
    ]
    assert overlay.shown == [[("circle", 200, 100, 40, 40, "1"), ("circle", 400, 300, 40, 40, "5"), ("point", 400, 300, None, None, None)]]

    pool.run_next()
    assert visual_and_speech(events)[-1] == ("speak", "Multiply these.")


def test_the_pointer_is_only_used_when_a_step_asks_for_it():
    events = []
    sequence, _tts, overlay, pool, _finished, _failed = make_sequence(events)

    sequence.play(plan(step("Just a box.", [show("box", 10, 10, 40, 40, "Panel")])))
    pool.run_next()  # the introduction, then the step's visual

    assert [primitive[0] for primitive in overlay.shown[0]] == ["box"]


# --- steps without visuals --------------------------------------------------


def test_a_step_without_visuals_is_still_spoken():
    events = []
    sequence, _tts, overlay, pool, finished, _failed = make_sequence(events)

    sequence.play(plan(step("This concept is about state.")))
    pool.run_next()  # the introduction, then the step's (empty) visual state

    assert visual_and_speech(events) == [("speak", "A short opening.")]
    assert overlay.shown == [[]]  # an empty visual state, nothing invented

    pool.run_next()

    assert [event for event in events if event[0] == "speak"] == [
        ("speak", "A short opening."),
        ("speak", "This concept is about state."),
    ]
    assert finished == [True]
    assert overlay.clear_calls >= 1


def test_a_repeated_line_is_only_spoken_once():
    events = []
    sequence, _tts, _overlay, pool, finished, _failed = make_sequence(events)

    # A direct answer has nothing to teach step by step, so it becomes both the
    # introduction and the single step - and is heard once.
    sequence.play(answer_only_teaching_plan("Just an answer."))

    assert pool.pending() == 1
    pool.run_next()

    assert [event for event in events if event[0] == "speak"] == [("speak", "Just an answer.")]
    assert finished == [True]


# --- introduction and conclusion --------------------------------------------


def test_the_introduction_is_spoken_before_the_first_step():
    events = []
    sequence, _tts, _overlay, pool, _finished, _failed = make_sequence(events)

    sequence.play(
        plan(
            step("Start with row one.", [show("highlight", 1, 2, 10, 10, "row")]),
            introduction="Here is the whole matrix lesson.",
        )
    )

    # The introduction is a sentence of its own, so no visual is up for it.
    assert visual_and_speech(events) == []

    pool.run_next()

    assert visual_and_speech(events) == [
        ("speak", "Here is the whole matrix lesson."),
        ("show", "row"),
    ]


def test_the_conclusion_is_spoken_after_the_last_step():
    events = []
    sequence, _tts, _overlay, pool, finished, _failed = make_sequence(events)

    sequence.play(plan(step("The only step."), conclusion="That is the idea."))

    while pool.pending():
        pool.run_next()

    assert [event for event in events if event[0] == "speak"] == [
        ("speak", "A short opening."),
        ("speak", "The only step."),
        ("speak", "That is the idea."),
    ]
    assert finished == [True]


# --- educational sequence ---------------------------------------------------


def matrix_lesson():
    return [
        step("Start with the first row of the left matrix.",
             [show("highlight", 600, 300, 320, 70, "first row of A")]),
        step("Now take the first column of the right matrix.",
             [show("box", 1100, 500, 90, 260, "first column of B")]),
        step("Multiply the matching values: 1 times 5 and 2 times 7.",
             [show("circle", 560, 300, 70, 60, "1"),
              show("circle", 1060, 430, 70, 60, "5"),
              show("point", 1060, 430)]),
        step("Add the products: 5 plus 14 equals 19.",
             [show("underline", 900, 720, 400, 40, "5 + 14 = 19")]),
        step("That gives the top-left value of the result.",
             [show("highlight", 1400, 700, 120, 90, "19")]),
    ]


def test_a_matrix_lesson_renders_each_visual_with_its_sentence():
    events = []
    sequence, _tts, overlay, pool, finished, _failed = make_sequence(
        events, size=1920
    )

    sequence.play(plan(*matrix_lesson()))

    assert shown_labels(events) == []
    pool.run_next()  # the introduction, then step one's visual
    assert shown_labels(events) == ["first row of A"]
    pool.run_next()
    assert shown_labels(events) == ["first row of A", "first column of B"]
    pool.run_next()
    assert shown_labels(events) == ["first row of A", "first column of B", "1", "5", "point"]
    pool.run_next()
    assert shown_labels(events) == ["first row of A", "first column of B", "1", "5", "point", "5 + 14 = 19"]
    pool.run_next()
    assert shown_labels(events)[-1] == "19"

    pool.run_next()
    assert finished == [True]
    assert sequence.active is False
    # Every sentence is spoken after its own visuals and before the next ones.
    assert visual_and_speech(events) == [
        ("speak", "A short opening."),
        ("show", "first row of A"),
        ("speak", "Start with the first row of the left matrix."),
        ("show", "first column of B"),
        ("speak", "Now take the first column of the right matrix."),
        ("show", "1"),
        ("show", "5"),
        ("show", "point"),
        ("speak", "Multiply the matching values: 1 times 5 and 2 times 7."),
        ("show", "5 + 14 = 19"),
        ("speak", "Add the products: 5 plus 14 equals 19."),
        ("show", "19"),
        ("speak", "That gives the top-left value of the result."),
    ]
    assert overlay.clear_calls >= 1


# --- failures ---------------------------------------------------------------


def test_tts_failure_cleans_up_and_reports():
    events = []
    tts = EventTTS(events, failures={"Boom."})
    sequence, tts, overlay, pool, finished, failed = make_sequence(events, tts=tts)

    sequence.play(plan(step("Boom.", [show("box", 1, 2, 10, 10, "X")])))
    pool.run_next()  # the introduction
    pool.run_next()  # the step, whose sentence fails

    assert failed == ["could not speak: Boom."]
    assert finished == []
    assert sequence.active is False
    assert overlay.clear_calls >= 1
    assert tts.stop_calls >= 1


def test_overlay_failure_does_not_stop_the_speech():
    events = []
    tts = EventTTS(events)
    teaching = ExplodingTeaching()
    pool = QueuedPool()
    sequence = TeachingSequenceService(tts, teaching, pool)
    finished = []
    sequence.finished.connect(lambda: finished.append(True))

    sequence.play(plan(step("Keep talking.", [show("box", 1, 2, 10, 10, "X")])))
    pool.run_next()
    pool.run_next()

    assert [event for event in events if event[0] == "speak"] == [
        ("speak", "A short opening."),
        ("speak", "Keep talking."),
    ]
    assert finished == [True]
    assert teaching.hide_calls >= 1


# --- cancellation -----------------------------------------------------------


def test_cancel_stops_speech_hides_the_overlay_and_terminates():
    events = []
    steps = [
        step("One.", [show("box", 1, 2, 10, 10, "A")]),
        step("Two.", [show("box", 3, 4, 10, 10, "B")]),
    ]
    sequence, tts, overlay, _pool, finished, _failed = make_sequence(events)

    sequence.play(plan(*steps))
    sequence.cancel()

    assert sequence.active is False
    assert tts.stop_calls >= 1
    assert overlay.clear_calls >= 1
    assert finished == []


def test_a_late_speech_completion_after_cancel_is_ignored():
    events = []
    steps = [
        step("One.", [show("box", 1, 2, 10, 10, "A")]),
        step("Two.", [show("box", 3, 4, 10, 10, "B")]),
    ]
    sequence, _tts, overlay, pool, finished, _failed = make_sequence(events)

    sequence.play(plan(*steps))
    pool.run_next()  # the introduction, then step one's visual

    assert shown_labels(events) == ["A"]

    sequence.cancel()
    pool.run_next()  # step one's sentence finishes after the cancel

    assert ("show", "B") not in events
    assert shown_labels(events) == ["A"]
    assert finished == []


def test_play_after_cancel_starts_a_fresh_sequence():
    events = []
    sequence, _tts, _overlay, pool, finished, _failed = make_sequence(events)

    sequence.play(plan(step("First.", [show("box", 1, 2, 10, 10, "A")])))
    pool.run_next()  # the first lesson's introduction, then its visual
    sequence.cancel()
    sequence.play(plan(step("Second.", [show("box", 3, 4, 10, 10, "B")])))

    while pool.pending():
        pool.run_next()

    assert shown_labels(events) == ["A", "B"]
    assert [event for event in events if event[0] == "speak"][-1] == ("speak", "Second.")
    assert finished == [True]


# --- grounding is preparation, not execution --------------------------------


def test_execution_has_no_grounding_hook():
    """Grounding prepares targets; the sequence can only draw and speak."""
    source = Path(sequence_module.__file__).read_text(encoding="utf-8").lower()

    for forbidden in ("grounder", "groundingworker", "visual_grounding"):
        assert forbidden not in source

    signature = inspect.signature(TeachingSequenceService.__init__).parameters
    assert "grounder" not in signature


# --- architecture -----------------------------------------------------------


def test_sequence_has_no_drawing_or_engine_code():
    source = Path(sequence_module.__file__).read_text(encoding="utf-8").lower()

    for forbidden in ("qpainter", "qwidget", "qcolor", "pyttsx3"):
        assert forbidden not in source
    assert "app.teaching.overlay" not in source
