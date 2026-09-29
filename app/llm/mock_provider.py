"""Minimal stand-in provider used for tests and for running without API keys.

It performs no network calls and holds no model weights.
"""

import logging
import time

from app.config import MOCK_LLM_DELAY_SECONDS
from app.llm.provider import VisionLLMProvider, VisionLLMRequest
from app.llm.response import (
    HeyArroResponse,
    ResponseContent,
    ResponseTone,
    TeachingMode,
    TeachingPlan,
)

logger = logging.getLogger(__name__)


class MockVisionLLMProvider(VisionLLMProvider):
    """Echoes the request back as a structured response so the whole pipeline can run offline."""

    name = "mock"

    def __init__(self, delay_seconds=MOCK_LLM_DELAY_SECONDS):
        self._delay = delay_seconds
        # What callers asked for, so tests can check the request that was made
        # without this provider having to understand it.
        self.plan_requests = []

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
            ),
            teaching=TeachingPlan(mode=TeachingMode.EXPLANATORY, steps=[]),
        )

    def generate_teaching_plan(
        self, user_query, screenshot=None, screenshot_size=None, context=None, guidance=None
    ) -> "TeachingPlan":
        """Return a deterministic teaching plan so plan-driven flows run offline.

        The plan carries no visual actions: this provider cannot see the screen,
        and inventing targets would contradict what the real prompt asks for.
        What it was asked for is recorded, so a caller's request can be checked
        without this provider interpreting it.
        """
        # Imported here so this module does not pull the teaching package in at
        # import time.
        from app.teaching.plan import TeachingPlan, TeachingStep

        self.plan_requests.append(
            {
                "user_query": user_query,
                "screenshot": screenshot,
                "screenshot_size": screenshot_size,
                "context": context,
                "guidance": guidance,
            }
        )

        if self._delay:
            time.sleep(self._delay)

        question = (user_query or "").strip() or "(no question)"
        logger.info("Mock provider planning how to teach %r", question)

        return TeachingPlan(
            objective=f"Understand this: {question}",
            introduction=f"Let's work through this together: {question}",
            steps=[
                TeachingStep(
                    step_id="step_1",
                    order=1,
                    objective="See what we are looking at.",
                    explanation="First, let's look at what is on your screen.",
                    transition="Now let's look at the part that answers your question.",
                ),
                TeachingStep(
                    step_id="step_2",
                    order=2,
                    objective="Understand the answer.",
                    explanation="This is the part that answers your question.",
                ),
            ],
            conclusion="That is the main idea.",
        )
