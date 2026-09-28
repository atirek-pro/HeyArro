"""Runs an LLM request on a thread pool so the Qt UI thread stays responsive."""

from PySide6.QtCore import QObject, QRunnable, Signal


class LLMSignals(QObject):
    """Signals emitted by an LLMWorker."""

    finished = Signal(object)
    failed = Signal(str)


class LLMWorker(QRunnable):
    """Executes ``provider.process()`` off the UI thread."""

    def __init__(self, provider, request):
        super().__init__()
        self.signals = LLMSignals()
        self._provider = provider
        self._request = request

    def run(self):
        try:
            response = self._provider.process(self._request)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(response)


class TeachingPlanWorker(QRunnable):
    """Executes ``provider.generate_teaching_plan()`` off the UI thread.

    Generating a follow-up lesson is a model call like any other, so it gets the
    same treatment as answering a question: it runs on the pool and reports back
    by signal, and the UI thread never waits for it.
    """

    def __init__(
        self,
        provider,
        user_query,
        screenshot=None,
        screenshot_size=None,
        context=None,
        guidance=None,
    ):
        super().__init__()
        self.signals = LLMSignals()
        self._provider = provider
        self._user_query = user_query
        self._screenshot = screenshot
        self._screenshot_size = screenshot_size
        self._context = context
        self._guidance = guidance

    def run(self):
        try:
            plan = self._provider.generate_teaching_plan(
                self._user_query,
                screenshot=self._screenshot,
                screenshot_size=self._screenshot_size,
                context=self._context,
                guidance=self._guidance,
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(plan)
