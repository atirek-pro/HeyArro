"""In-memory TTS provider for tests and offline runs.

It makes no sound and needs no dependency: it records what it was asked to say
so tests and the coordinator can be exercised without a real speech engine.
"""

import logging
import time

from app.tts.provider import TTSProvider, normalize_text

logger = logging.getLogger(__name__)


class MockTTSProvider(TTSProvider):
    """Records spoken text instead of producing audio."""

    name = "mock"

    def __init__(self, delay_seconds=0.0):
        self._delay = delay_seconds
        self.spoken_texts = []
        self.stop_calls = 0

    @property
    def spoken(self):
        """True once at least one utterance has been requested."""
        return bool(self.spoken_texts)

    def speak(self, text):
        speakable = normalize_text(text)
        if not speakable:
            return

        if self._delay:
            # Stand in for the duration of real speech.
            time.sleep(self._delay)

        logger.info("Mock TTS speaking: %s", speakable)
        self.spoken_texts.append(speakable)

    def stop(self):
        self.stop_calls += 1
