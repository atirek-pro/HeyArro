"""Minimal stand-in provider used for tests and for running without API keys.

It performs no network calls and holds no model weights.
"""

import logging
import time

from app.config import MOCK_LLM_DELAY_SECONDS
from app.llm.provider import VisionLLMProvider, VisionLLMRequest
from app.llm.response import HeyArroResponse, ResponseContent, ResponseTone

logger = logging.getLogger(__name__)


class MockVisionLLMProvider(VisionLLMProvider):
    """Echoes the request back as a structured response so the whole pipeline can run offline."""

    name = "mock"

    def __init__(self, delay_seconds=MOCK_LLM_DELAY_SECONDS):
        self._delay = delay_seconds

    def process(self, request: VisionLLMRequest) -> HeyArroResponse:
        if self._delay:
            # Stand in for the latency of a real request.
            time.sleep(self._delay)

        transcript = (request.transcript or "").strip()
        attachment = (
            f"{len(request.screenshot)} byte screenshot"
            if request.screenshot
            else "no screenshot"
        )
        logger.info("Mock provider responding to %r (%s)", transcript, attachment)

        return HeyArroResponse(
            response=ResponseContent(
                text=f'[mock response] You said: "{transcript}" ({attachment}).',
                tone=ResponseTone.NEUTRAL,
            )
        )

