"""Hey Arro entry point.

Composition root only: it builds the services and hands them to
ApplicationCoordinator, which drives one interaction at a time. The application
displays nothing: the question, the screen it was asked about and the answer are
written to the log.
"""

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from app.audio.recorder import RECORDINGS_DIR, Recorder
from app.capture import create_screen_capture_provider
from app.config import LLM_PROVIDER, SCREENSHOT_DIR
from app.coordinator import ApplicationCoordinator
from app.hotkey.global_hotkey import GlobalHotkey
from app.llm import UnsupportedProviderError, get_llm_provider
from app.transcription import create_transcription_provider
from app.ui.tray import Tray

logger = logging.getLogger(__name__)


def _status_text(transcription, capture, llm):
    return (
        "Hey Arro is running in the background.\n\n"
        "Global hotkey: Ctrl + Alt (hold it while you ask)\n"
        f"Speech to text: {type(transcription).__name__}\n"
        f"Screen capture: {type(capture).__name__}\n"
        f"Provider: {LLM_PROVIDER} ({type(llm).__name__}, model: {getattr(llm, 'model', 'n/a')})\n"
        f"Recordings: {RECORDINGS_DIR}\n"
        f"Screenshots: {SCREENSHOT_DIR}\n\n"
        "Hold Ctrl + Alt, ask your question out loud, then release. "
        "What was asked, what was on screen and what the model answered are "
        "written to the log."
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
    app.setApplicationName("Hey Arro")
    app.setQuitOnLastWindowClosed(False)

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
            None, "Hey Arro", _status_text(transcription, capture, llm)
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
        "Model provider: %s (configured=%s, model=%s)",
        type(llm).__name__,
        LLM_PROVIDER,
        getattr(llm, "model", "n/a"),
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
