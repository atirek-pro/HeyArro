import pytest

from app.state import (
    VALID_TRANSITIONS,
    ApplicationState,
    ApplicationStateMachine,
)


def test_starts_in_idle():
    assert ApplicationStateMachine().state is ApplicationState.IDLE


def test_idle_to_listening_succeeds():
    machine = ApplicationStateMachine()
    assert machine.transition_to(ApplicationState.LISTENING) is True
    assert machine.state is ApplicationState.LISTENING


def test_listening_to_processing_succeeds():
    machine = ApplicationStateMachine()
    machine.transition_to(ApplicationState.LISTENING)
    assert machine.transition_to(ApplicationState.PROCESSING) is True
    assert machine.state is ApplicationState.PROCESSING


def test_processing_to_responding_succeeds():
    machine = ApplicationStateMachine()
    machine.transition_to(ApplicationState.LISTENING)
    machine.transition_to(ApplicationState.PROCESSING)
    assert machine.transition_to(ApplicationState.RESPONDING) is True
    assert machine.state is ApplicationState.RESPONDING


def test_responding_to_idle_succeeds():
    machine = ApplicationStateMachine()
    machine.transition_to(ApplicationState.LISTENING)
    machine.transition_to(ApplicationState.PROCESSING)
    machine.transition_to(ApplicationState.RESPONDING)
    assert machine.transition_to(ApplicationState.IDLE) is True
    assert machine.state is ApplicationState.IDLE


def test_error_to_idle_succeeds():
    machine = ApplicationStateMachine(ApplicationState.ERROR)
    assert machine.transition_to(ApplicationState.IDLE) is True
    assert machine.state is ApplicationState.IDLE


@pytest.mark.parametrize(
    "start",
    [
        ApplicationState.IDLE,
        ApplicationState.LISTENING,
        ApplicationState.PROCESSING,
        ApplicationState.RESPONDING,
    ],
)
def test_active_states_can_enter_error(start):
    machine = ApplicationStateMachine(start)
    assert machine.transition_to(ApplicationState.ERROR) is True
    assert machine.state is ApplicationState.ERROR


def test_idle_to_responding_is_rejected():
    machine = ApplicationStateMachine()
    assert machine.transition_to(ApplicationState.RESPONDING) is False
    assert machine.state is ApplicationState.IDLE


def test_idle_to_processing_is_rejected():
    machine = ApplicationStateMachine()
    assert machine.transition_to(ApplicationState.PROCESSING) is False
    assert machine.state is ApplicationState.IDLE


def test_listening_to_responding_is_rejected():
    machine = ApplicationStateMachine()
    machine.transition_to(ApplicationState.LISTENING)
    assert machine.transition_to(ApplicationState.RESPONDING) is False
    assert machine.state is ApplicationState.LISTENING


def test_listening_cannot_skip_processing_back_to_idle():
    machine = ApplicationStateMachine()
    machine.transition_to(ApplicationState.LISTENING)
    assert machine.transition_to(ApplicationState.IDLE) is False
    assert machine.state is ApplicationState.LISTENING


def test_rejected_transition_preserves_current_state():
    machine = ApplicationStateMachine(ApplicationState.PROCESSING)
    for target in (ApplicationState.IDLE, ApplicationState.LISTENING, ApplicationState.PROCESSING):
        assert machine.transition_to(target) is False
        assert machine.state is ApplicationState.PROCESSING


def test_error_cannot_go_straight_to_listening():
    machine = ApplicationStateMachine(ApplicationState.ERROR)
    assert machine.transition_to(ApplicationState.LISTENING) is False
    assert machine.state is ApplicationState.ERROR


def test_unknown_state_value_is_rejected():
    machine = ApplicationStateMachine()
    assert machine.transition_to("not-a-state") is False
    assert machine.state is ApplicationState.IDLE


def test_can_transition_to_reports_allowed_targets():
    machine = ApplicationStateMachine()
    assert machine.can_transition_to(ApplicationState.LISTENING) is True
    assert machine.can_transition_to(ApplicationState.RESPONDING) is False
    assert machine.can_transition_to(ApplicationState.IDLE) is False


def test_observers_receive_state_change_events():
    machine = ApplicationStateMachine()
    events = []
    machine.state_changed.connect(lambda old, new: events.append((old, new)))

    machine.transition_to(ApplicationState.LISTENING)
    machine.transition_to(ApplicationState.PROCESSING)
    machine.transition_to(ApplicationState.RESPONDING)
    machine.transition_to(ApplicationState.IDLE)

    assert events == [
        (ApplicationState.IDLE, ApplicationState.LISTENING),
        (ApplicationState.LISTENING, ApplicationState.PROCESSING),
        (ApplicationState.PROCESSING, ApplicationState.RESPONDING),
        (ApplicationState.RESPONDING, ApplicationState.IDLE),
    ]


def test_rejected_transition_emits_nothing():
    machine = ApplicationStateMachine()
    events = []
    machine.state_changed.connect(lambda old, new: events.append((old, new)))

    machine.transition_to(ApplicationState.RESPONDING)

    assert events == []


def test_every_state_has_a_documented_transition_set():
    assert set(VALID_TRANSITIONS) == set(ApplicationState)


def test_companion_has_an_appearance_for_every_state():
    from app.ui.companion import APPEARANCE

    assert set(APPEARANCE) == set(ApplicationState)
