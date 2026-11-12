"""Hey Arro entry point.

Composition root only: it builds the services and hands them to
ApplicationCoordinator, which drives one interaction at a time. A glass status
pill follows each interaction, and the answer is spoken while its marks are
drawn on the screen; every step is also written to the log.
"""

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from app.audio.recorder import RECORDINGS_DIR, Recorder
from app.capture import create_screen_capture_provider
from app.config import (
    OVERLAY_ENABLED,
    REFINEMENT_DIR,
    REFINEMENT_ENABLED,
    SCREENSHOT_DIR,
    STATUS_INDICATOR_ENABLED,
)
from app.coordinator import ApplicationCoordinator
from app.final_response import create_response_composer
from app.final_response.overlay import ScreenOverlay
from app.final_response.presenter import ResponsePresenter
from app.final_response.speech import create_speech_provider
from app.hotkey.global_hotkey import GlobalHotkey
from app.logging_setup import configure_logging
from app.grounding import create_plan_grounder
from app.planner import create_response_planner
from app.refinement import create_refinement_provider, create_segment_selector
from app.transcription import create_transcription_provider
from app.ui.status_indicator import StatusIndicator
from app.ui.tray import Tray, app_icon

logger = logging.getLogger(__name__)


def _status_text(transcription, capture):
    return (
        "Hey Arro is running in the background.\n\n"
        "Global hotkey: Ctrl + Alt (hold it while you ask)\n"
        f"Speech to text: {type(transcription).__name__}\n"
        f"Screen capture: {type(capture).__name__}\n"
        f"Recordings: {RECORDINGS_DIR}\n"
        f"Screenshots: {SCREENSHOT_DIR}\n"
        f"Screen segments: {REFINEMENT_DIR if REFINEMENT_ENABLED else 'disabled'}\n\n"
        "Hold Ctrl + Alt, ask your question out loud, then release. "
        "The answer is spoken while its marks are drawn on the screen, and every "
        "step is written to the log."
    )


def main():
    configure_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("Hey Arro")
    app.setWindowIcon(app_icon())
    app.setQuitOnLastWindowClosed(False)

    hotkey = GlobalHotkey(keys=("ctrl", "alt"))
    recorder = Recorder()
    transcription = create_transcription_provider()
    capture = create_screen_capture_provider()
    refinement = create_refinement_provider() if REFINEMENT_ENABLED else None
    segment_selector = create_segment_selector() if refinement else None
    response_planner = create_response_planner() if segment_selector else None
    plan_grounder = create_plan_grounder() if response_planner else None
    response_composer = create_response_composer() if plan_grounder else None
    response_presenter = (
        ResponsePresenter(
            overlay=ScreenOverlay() if OVERLAY_ENABLED else None,
            speech=create_speech_provider(),
        )
        if response_composer
        else None
    )

    coordinator = ApplicationCoordinator(
        hotkey=hotkey,
        recorder=recorder,
        transcription=transcription,
        capture=capture,
        refinement=refinement,
        segment_selector=segment_selector,
        response_planner=response_planner,
        plan_grounder=plan_grounder,
        response_composer=response_composer,
        response_presenter=response_presenter,
    )

    if STATUS_INDICATOR_ENABLED:
        indicator = StatusIndicator()
        coordinator.status_changed.connect(indicator.set_status)

    tray = Tray(
        on_status=lambda: QMessageBox.information(
            None, "Hey Arro", _status_text(transcription, capture)
        ),
        on_quit=app.quit,
    )

    def on_error(message):
        # The only thing that ever surfaces outside the log.
        tray.showMessage("Hey Arro", message, QSystemTrayIcon.Warning, 5000)

    coordinator.error_occurred.connect(on_error)

    tray.show()

    try:
        hotkey.start()
    except Exception as exc:  # pragma: no cover - depends on the host
        QMessageBox.critical(
            None, "Hey Arro", f"Could not register the global hotkey:\n{exc}"
        )

    logger.info(
        "Hey Arro ready - hold Ctrl + Alt to ask (recordings: %s, screenshots: %s)",
        RECORDINGS_DIR,
        SCREENSHOT_DIR,
    )

    app.aboutToQuit.connect(coordinator.shutdown)
    app.aboutToQuit.connect(recorder.abort)
    app.aboutToQuit.connect(hotkey.stop)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
