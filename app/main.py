"""Clicky Assistant prototype entry point.

Composition root only: it builds the services, hands them to
ApplicationCoordinator, and connects the coordinator's state to the UI.
The lifecycle itself lives in app/coordinator.py and app/state.py.
"""

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from app.audio.recorder import RECORDINGS_DIR, Recorder
from app.capture import create_screen_capture_provider
from app.config import LLM_PROVIDER, SCREENSHOT_DIR, TRANSCRIPT_DISPLAY_MS
from app.coordinator import ApplicationCoordinator
from app.hotkey.global_hotkey import GlobalHotkey
from app.llm import UnsupportedProviderError, get_llm_provider
from app.state import ApplicationState
from app.transcription import create_transcription_provider
from app.ui.caption import Caption
from app.ui.companion import Companion
from app.ui.indicator import Indicator
from app.ui.tray import Tray

logger = logging.getLogger(__name__)


def _status_text(coordinator, transcription, capture, llm):
    return (
        "Clicky Assistant is running in the background.\n\n"
        "Global hotkey: Ctrl + Alt\n"
        f"State: {coordinator.state.value}\n"
        f"Transcription: {type(transcription).__name__}\n"
        f"Screen capture: {type(capture).__name__}\n"
        f"Response: {type(llm).__name__}\n"
        f"Provider: {LLM_PROVIDER} (model: {getattr(llm, 'model', 'n/a')})\n\n"
        "Hold Ctrl + Alt to record, transcribe, capture and respond."
    )


def _select_llm_provider():
    """Build the configured provider, keeping the app alive if it is misconfigured."""
    try:
        return get_llm_provider()
    except UnsupportedProviderError as exc:
        logger.error("%s", exc)
        logger.error("Falling back to the mock provider so the application keeps running")
        return get_llm_provider("mock")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Clicky Assistant")
    app.setQuitOnLastWindowClosed(False)

    indicator = Indicator()
    companion = Companion()
    caption = Caption(TRANSCRIPT_DISPLAY_MS)
    hotkey = GlobalHotkey(keys=("ctrl", "alt"))
    recorder = Recorder()
    transcription = create_transcription_provider()
    capture = create_screen_capture_provider()
    llm = _select_llm_provider()

    coordinator = ApplicationCoordinator(
        hotkey=hotkey,
        recorder=recorder,
        transcription=transcription,
        capture=capture,
        llm=llm,
    )

    tray = Tray(
        on_status=lambda: QMessageBox.information(
            None,
            "Clicky Assistant",
            _status_text(coordinator, transcription, capture, llm),
        ),
        on_quit=app.quit,
    )

    # --- view wiring: the coordinator drives, the views only render ---------

    def on_state_changed(previous, current):
        companion.render_state(current)

        if current is ApplicationState.LISTENING:
            caption.hide()
            indicator.show_listening()
        elif current is ApplicationState.PROCESSING:
            indicator.show_released()

    def on_error(message):
        tray.showMessage("Clicky Assistant", message, QSystemTrayIcon.Warning, 5000)

    def on_response(response):
        caption.show_text(response.text)

    coordinator.state_changed.connect(on_state_changed)
    coordinator.error_occurred.connect(on_error)
    coordinator.response_ready.connect(on_response)

    tray.show()
    companion.start()

    try:
        hotkey.start()
    except Exception as exc:  # pragma: no cover - depends on the host
        QMessageBox.critical(
            None, "Clicky Assistant", f"Could not register the global hotkey:\n{exc}"
        )

    logger.info(
        "Vision LLM provider: %s (configured=%s, model=%s)",
        type(llm).__name__,
        LLM_PROVIDER,
        getattr(llm, "model", "n/a"),
    )
    logger.info(
        "Clicky Assistant ready - hold Ctrl + Alt to record, transcribe, capture and respond "
        "(recordings: %s, screenshots: %s)",
        RECORDINGS_DIR,
        SCREENSHOT_DIR,
    )

    app.aboutToQuit.connect(coordinator.shutdown)
    app.aboutToQuit.connect(caption.hide)
    app.aboutToQuit.connect(recorder.abort)
    app.aboutToQuit.connect(companion.stop)
    app.aboutToQuit.connect(hotkey.stop)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
