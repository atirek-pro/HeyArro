"""Manual visual-teaching demo: ``python -m app.teaching``.

Runs one scripted interaction through the real coordinator, the real voice and
the real overlay, without a microphone, Whisper or Gemini: the "screen capture"
is a real screenshot of the primary screen and the "model" returns a fixed
matrix-multiplication lesson. Use it to watch several visual primitives
(highlight, box, circle, underline, pointer) appear before each sentence, evolve
from step to step, and clean themselves up.

Set TEACHING_ENABLED=false to run it voice-only, or TTS_PROVIDER=mock for a
silent run.
"""

import logging
import sys

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from app.capture import create_screen_capture_provider
from app.capture.provider import ScreenCaptureProvider
from app.coordinator import ApplicationCoordinator
from app.llm.provider import VisionLLMProvider
from app.llm.response import (
    HeyArroResponse,
    ResponseContent,
    ResponseTone,
    VisualAction,
    VisualActionType,
    VisualTarget,
    TeachingMode,
    TeachingPlan,
    TeachingStep,
)
from app.state import ApplicationState
from app.teaching import create_visual_teaching_service
from app.transcription.provider import TranscriptionProvider, TranscriptionResult
from app.tts import get_tts_provider
from app.visual_grounding import create_visual_grounder

PRESS_DELAY_MS = 300
RELEASE_DELAY_MS = 1200
QUIT_DELAY_MS = 800
# Safety net only. With visual grounding on, every target costs one refinement
# request, so the demo can legitimately take a few minutes.
SAFETY_TIMEOUT_MS = 600000


def target(x, y, width=None, height=None, label=""):
    """A screenshot-space target; width/height make it a region."""
    return VisualTarget(x=x, y=y, width=width, height=height, label=label)


def action(kind, x, y, width=None, height=None, label=""):
    return VisualAction(
        type=VisualActionType(kind), target=target(x, y, width, height, label)
    )


DEMO_RESPONSE = HeyArroResponse(
    response=ResponseContent(
        text="Multiplying a row by a column.",
        tone=ResponseTone.INSTRUCTIONAL,
    ),
    teaching=TeachingPlan(
        mode=TeachingMode.GUIDED,
        steps=[
            TeachingStep(
                instruction="Start with the first row of the left matrix.",
                visual_actions=[
                    action("highlight", 600, 300, 320, 70, "first row of A"),
                    action("point", 600, 300),
                ],
            ),
            TeachingStep(
                instruction="Now take the first column of the right matrix.",
                visual_actions=[action("box", 1100, 500, 90, 260, "first column of B")],
            ),
            TeachingStep(
                instruction="Multiply the matching values: 1 times 5, and 2 times 7.",
                visual_actions=[
                    action("circle", 560, 300, 70, 60, "1"),
                    action("circle", 1060, 430, 70, 60, "5"),
                    action("point", 1060, 430),
                ],
            ),
            TeachingStep(
                instruction="Add the two products: 5 plus 14 equals 19.",
                visual_actions=[action("underline", 900, 720, 400, 40, "5 + 14 = 19")],
            ),
            TeachingStep(
                instruction="That gives the top-left value of the result matrix.",
                visual_actions=[
                    action("highlight", 1400, 700, 120, 90, "19"),
                    action("circle", 1400, 700, 150, 112),
                ],
            ),
        ],
    ),
)


class DemoHotkey(QObject):
    pressed = Signal()
    released = Signal()

    def start(self):
        return None

    def stop(self):
        return None


class DemoRecorder(QObject):
    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._recording = False

    @property
    def recording(self):
        return self._recording

    def start(self):
        self._recording = True

    def stop(self):
        self._recording = False
        self.finished.emit("demo-recording.wav", {})

    def abort(self):
        self._recording = False


class DemoTranscription(TranscriptionProvider):
    def transcribe(self, audio_file):
        return TranscriptionResult(
            text="Explain this matrix multiplication.", duration=1.0, language="en"
        )


class DemoCapture(ScreenCaptureProvider):
    """Returns one captured screenshot, standing in for the live screen."""

    def __init__(self, result):
        self._result = result

    def list_monitors(self):
        return [self._result.monitor] if self._result.monitor else []

    def capture_screen(self, monitor_index=None):
        return self._result


class DemoLLM(VisionLLMProvider):
    name = "demo"

    def process(self, request):
        return DEMO_RESPONSE


def _capture_reference_screen():
    """Take one real screenshot so the demo's coordinates map like production."""
    try:
        return create_screen_capture_provider().capture_screen()
    except Exception as exc:  # pragma: no cover - depends on the host
        from PySide6.QtGui import QGuiApplication

        from app.capture.provider import MonitorInfo, ScreenCaptureResult

        logger.warning("Live capture failed (%s); using a synthetic screenshot", exc)
        rect = QGuiApplication.primaryScreen().geometry()
        monitor = MonitorInfo(1, rect.left(), rect.top(), rect.width(), rect.height(), True)
        return ScreenCaptureResult(
            image=b"\x89PNG\r\n\x1a\n",
            monitor_index=1,
            width=rect.width(),
            height=rect.height(),
            timestamp=0.0,
            monitor=monitor,
        )


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    coordinator = ApplicationCoordinator(
        hotkey=DemoHotkey(),
        recorder=DemoRecorder(),
        transcription=DemoTranscription(),
        capture=DemoCapture(_capture_reference_screen()),
        llm=DemoLLM(),
        tts=get_tts_provider(),
        teaching=create_visual_teaching_service(),
        grounder=create_visual_grounder(),
        responding_hold_ms=QUIT_DELAY_MS,
    )

    def on_state_changed(previous, current):
        if current is ApplicationState.IDLE and coordinator.last_response is not None:
            QTimer.singleShot(QUIT_DELAY_MS, app.quit)

    coordinator.state_changed.connect(on_state_changed)

    # Drive one interaction: press the (virtual) hotkey, then release it.
    QTimer.singleShot(PRESS_DELAY_MS, coordinator.on_hotkey_pressed)
    QTimer.singleShot(RELEASE_DELAY_MS, coordinator.on_hotkey_released)
    QTimer.singleShot(SAFETY_TIMEOUT_MS, app.quit)

    app.aboutToQuit.connect(coordinator.shutdown)
    print(
        "Demo running: each sentence is preceded by its own visual state "
        "(row, column, cells, expression, result)."
    )
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
