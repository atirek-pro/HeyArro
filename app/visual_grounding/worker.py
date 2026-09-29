"""Grounds a whole teaching plan off the UI thread.

Refinement makes a network request per distinct target, so it must never run on
the UI thread. The worker grounds the entire plan in one go and hands back the
same plan with refined targets, so the teaching sequence can then draw and speak
without ever making a request of its own.

Preparation can outlive the window it was started for (a slow request while the
user quits), so emitting is guarded: once Qt has torn the signals object down
there is nobody left to tell. A preparation that blows up is reported as a
failure, because executing a half-prepared plan is worse than not executing it.
"""

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

from app.visual_grounding.grounder import ground_teaching_plan

logger = logging.getLogger(__name__)


class GroundingSignals(QObject):
    """Signals emitted by a GroundingWorker."""

    finished = Signal(object)
    failed = Signal(str)


class GroundingWorker(QRunnable):
    """Grounds every visual target in a teaching plan off the UI thread."""

    def __init__(self, grounder, plan, screenshot, screenshot_size):
        super().__init__()
        self.signals = GroundingSignals()
        self._grounder = grounder
        self._plan = plan
        self._screenshot = screenshot
        self._screenshot_size = screenshot_size

    def run(self):
        try:
            grounded = ground_teaching_plan(
                self._grounder, self._plan, self._screenshot, self._screenshot_size
            )
        except Exception as exc:
            self._safe_emit(lambda: self.signals.failed.emit(str(exc)))
            return
        self._safe_emit(lambda: self.signals.finished.emit(grounded))

    def _safe_emit(self, emit):
        """Emit unless the application is shutting down and the signals are gone."""
        try:
            emit()
        except RuntimeError:
            logger.debug("Dropping a grounding result: the application is shutting down")
