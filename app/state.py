"""The single authoritative high-level application state and its transitions."""

import logging
from enum import Enum

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class ApplicationState(Enum):
    """High-level lifecycle of the assistant."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"


# Every transition the application is allowed to make. Anything not listed here
# is rejected by ApplicationStateMachine.transition_to().
VALID_TRANSITIONS = {
    ApplicationState.IDLE: frozenset(
        {
            ApplicationState.LISTENING,
            ApplicationState.ERROR,
        }
    ),
    ApplicationState.LISTENING: frozenset(
        {
            ApplicationState.PROCESSING,
            ApplicationState.ERROR,
        }
    ),
    ApplicationState.PROCESSING: frozenset(
        {
            ApplicationState.RESPONDING,
            ApplicationState.ERROR,
        }
    ),
    ApplicationState.RESPONDING: frozenset(
        {
            ApplicationState.IDLE,
            ApplicationState.ERROR,
        }
    ),
    ApplicationState.ERROR: frozenset({ApplicationState.IDLE}),
}


class ApplicationStateMachine(QObject):
    """Owns the application state and validates every transition.

    The machine is only driven from the Qt main thread, so no locking is needed.
    """

    state_changed = Signal(object, object)

    def __init__(self, initial=ApplicationState.IDLE, parent=None):
        super().__init__(parent)
        self._state = ApplicationState(initial)

    @property
    def state(self):
        return self._state

    def can_transition_to(self, new_state):
        """True when ``new_state`` is reachable from the current state."""
        try:
            target = ApplicationState(new_state)
        except ValueError:
            return False
        return target is not self._state and target in VALID_TRANSITIONS[self._state]

    def transition_to(self, new_state):
        """Validate and apply a transition, returning True when it was applied.

        An invalid transition is rejected, the current state is preserved and a
        useful error is logged.
        """
        try:
            target = ApplicationState(new_state)
        except ValueError:
            logger.error("Rejected transition: %r is not an application state", new_state)
            return False

        if target is self._state:
            logger.error("Rejected transition %s -> %s: already in that state", self._state.value, target.value)
            return False

        if target not in VALID_TRANSITIONS[self._state]:
            logger.error("Rejected invalid transition %s -> %s", self._state.value, target.value)
            return False

        previous = self._state
        self._state = target
        logger.info("Application state: %s -> %s", previous.value, target.value)
        self.state_changed.emit(previous, target)
        return True
