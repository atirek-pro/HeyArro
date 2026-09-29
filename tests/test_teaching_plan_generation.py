"""Teaching-plan generation: Gemini returns a validated TeachingPlan."""

import pytest

from app.llm.gemini_provider import GeminiVisionProvider
from app.llm.mock_provider import MockVisionLLMProvider
from app.llm.openai_provider import OpenAIVisionProvider
from app.llm.provider import (
    LLMError,
    ProviderConfigurationError,
    ProviderNotImplementedError,
    VisionLLMRequest,
)
from app.llm.response import VisualAction, VisualActionType, VisualTarget
from app.teaching.plan import TeachingPlan
from app.teaching.plan_prompt import (
    TEACHING_PLAN_SYSTEM_INSTRUCTION,
    build_teaching_plan_prompt,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def png_bytes(width, height):
    """A minimal PNG whose IHDR carries a real pixel size."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
        + b"\x00" * 8
    )


# --- test doubles -----------------------------------------------------------


class FakeResponse:
    def __init__(self, text=None, parsed=None):
        self.text = text
        self.parsed = parsed


class FakeModels:
    def __init__(self, response=None, error=None):
        self.calls = []
        self._response = response
        self._error = error

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response, error)


def generator(response=None, error=None, **kwargs):
    client = FakeClient(response, error)
    return GeminiVisionProvider(api_key="test-key", client=client, **kwargs), client


def prompt_text(client):
    texts = []
    for part in client.models.calls[0]["contents"].parts:
        piece = getattr(part, "text", None)
        if piece:
            texts.append(piece)
    return " ".join(texts)


# --- plan builders ----------------------------------------------------------


def action(x=10, y=20, label="Matrix A"):
    return VisualAction(
        type=VisualActionType.HIGHLIGHT,
        target=VisualTarget(x=x, y=y, width=120, height=80, label=label),
    )


def step_payload(step_id="step_1", order=1, actions=None, transition=None):
    return {
        "step_id": step_id,
        "order": order,
        "objective": f"Objective {order}.",
        "explanation": f"Spoken explanation {order}.",
        "visual_actions": actions or [],
        "transition": transition,
    }


def plan_payload(steps=None, **overrides):
    payload = {
        "objective": "Understand how this works.",
        "introduction": "Let's break this down together.",
        "steps": [step_payload(step_id="step_1", order=1), step_payload(step_id="step_2", order=2)]
        if steps is None
        else steps,
        "conclusion": "The key idea is that each result comes from one row and one column.",
    }
    payload.update(overrides)
    return payload


# --- generation -------------------------------------------------------------


def test_gemini_generates_a_plan_from_structured_output():
    steps = [step_payload(), step_payload("step_2", 2)]
    provider, _client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload(steps))))

    plan = provider.generate_teaching_plan("Explain matrix multiplication.", PNG)

    assert isinstance(plan, TeachingPlan)
    assert plan.objective == "Understand how this works."
    assert [step.step_id for step in plan.steps] == ["step_1", "step_2"]


def test_gemini_generates_a_plan_from_structured_text():
    payload = plan_payload()
    provider, _client = generator(FakeResponse(text=TeachingPlan.model_validate(payload).model_dump_json()))

    plan = provider.generate_teaching_plan("Explain this.", PNG)

    assert isinstance(plan, TeachingPlan)
    assert len(plan.steps) == 2


def test_gemini_asks_for_the_teaching_plan_schema():
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    provider.generate_teaching_plan("Explain this.", PNG)

    request_config = client.models.calls[0]["config"]
    assert request_config.response_schema is TeachingPlan
    assert request_config.response_mime_type == "application/json"
    assert request_config.system_instruction == TEACHING_PLAN_SYSTEM_INSTRUCTION


def test_gemini_sends_the_query_and_the_screenshot():
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    provider.generate_teaching_plan("Explain what this graph is showing.", png_bytes(1280, 720))

    parts = client.models.calls[0]["contents"].parts
    assert any(getattr(part, "inline_data", None) for part in parts)
    assert "Explain what this graph is showing." in prompt_text(client)


def test_gemini_tells_the_model_the_screenshot_grid():
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    provider.generate_teaching_plan("Explain this.", png_bytes(1280, 720))

    assert "1280 by 720" in prompt_text(client)


def test_without_a_screenshot_the_model_is_told_not_to_use_visual_actions():
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    provider.generate_teaching_plan("What is a monad?")

    parts = client.models.calls[0]["contents"].parts
    assert not any(getattr(part, "inline_data", None) for part in parts)
    assert "do not use any visual actions" in prompt_text(client)


def test_gemini_passes_earlier_context_through():
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    provider.generate_teaching_plan(
        "And the second one?",
        PNG,
        None,
        "The learner just asked about the first matrix.",
    )

    text = prompt_text(client)
    assert "Context from earlier in this session:" in text
    assert "The learner just asked about the first matrix." in text


def test_gemini_passes_the_teaching_guidance_through():
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    provider.generate_teaching_plan(
        "Explain this.", PNG, None, None, "Teach this for someone new to the topic."
    )

    text = prompt_text(client)
    assert "How to teach this request:" in text
    assert "Teach this for someone new to the topic." in text


def test_gemini_says_nothing_about_depth_without_guidance():
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    provider.generate_teaching_plan("Explain this.", PNG)

    assert "How to teach this request:" not in prompt_text(client)


def test_gemini_passes_the_answering_guidance_to_the_model_too():
    """The depth reaches the prompt that answers the question, not just planning."""
    payload = {
        "response": {"text": "Here you go.", "tone": "neutral"},
        "teaching": {"mode": "direct", "steps": []},
    }
    provider, client = generator(FakeResponse(parsed=payload))

    provider.process(
        VisionLLMRequest(transcript="What is this?", guidance="Teach it plainly.")
    )

    text = prompt_text(client)
    assert "How to teach this:" in text
    assert "Teach it plainly." in text


def test_a_provider_that_cannot_plan_reports_it_clearly():
    with pytest.raises(ProviderNotImplementedError):
        OpenAIVisionProvider().generate_teaching_plan("Explain this.")


def test_the_configured_model_is_used():
    provider, client = generator(
        FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())), model="gemini-test-model"
    )

    provider.generate_teaching_plan("Explain this.", PNG)

    assert client.models.calls[0]["model"] == "gemini-test-model"


def test_plan_generation_reuses_the_provider_client():
    """No second Gemini client is built for planning."""
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    provider.generate_teaching_plan("First question.", PNG)
    provider.generate_teaching_plan("Second question.", PNG)

    assert provider._client is client
    assert len(client.models.calls) == 2


# --- the shape of the generated plan ----------------------------------------


def test_a_generated_plan_keeps_the_teaching_order():
    steps = [step_payload("step_1", 1), step_payload("step_2", 2), step_payload("step_3", 3)]
    provider, _client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload(steps))))

    plan = provider.generate_teaching_plan("Teach me this from scratch.", PNG)

    assert [step.order for step in plan.steps] == [1, 2, 3]
    assert plan.steps[0].order < plan.steps[1].order < plan.steps[2].order
    assert [step.step_id for step in plan.steps] == ["step_1", "step_2", "step_3"]


def test_a_generated_plan_carries_the_existing_visual_actions():
    steps = [
        step_payload(actions=[action(100, 200, "first row").model_dump()]),
        step_payload("step_2", 2, actions=[action(300, 400, "first column").model_dump()]),
    ]
    provider, _client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload(steps))))

    plan = provider.generate_teaching_plan("Explain this matrix.", PNG)

    highlighted = plan.steps[0].visual_actions[0]
    assert isinstance(highlighted, VisualAction)
    assert highlighted.target.label == "first row"
    assert (highlighted.target.width, highlighted.target.height) == (120, 80)
    assert plan.steps[1].visual_actions[0].target.label == "first column"


def test_a_generated_plan_may_have_no_visual_actions():
    provider, _client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    plan = provider.generate_teaching_plan("What does this concept mean?", PNG)

    assert all(step.visual_actions == [] for step in plan.steps)


def test_a_transition_is_optional_per_step():
    steps = [
        step_payload(transition="Now let's look at the second matrix."),
        step_payload("step_2", 2),
    ]
    provider, _client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload(steps))))

    plan = provider.generate_teaching_plan("Explain this.", PNG)

    assert plan.steps[0].transition.startswith("Now let's")
    assert plan.steps[1].transition is None


# --- failures ---------------------------------------------------------------


def test_an_invalid_plan_is_rejected():
    broken = plan_payload(objective="")
    provider, _client = generator(FakeResponse(parsed=broken))

    with pytest.raises(LLMError):
        provider.generate_teaching_plan("Explain this.", PNG)


def test_a_plan_with_duplicate_orders_is_rejected():
    broken = plan_payload(steps=[step_payload("step_1", 1), step_payload("step_2", 1)])
    provider, _client = generator(FakeResponse(parsed=broken))

    with pytest.raises(LLMError):
        provider.generate_teaching_plan("Explain this.", PNG)


def test_malformed_structured_text_is_rejected():
    provider, _client = generator(FakeResponse(text="not a plan at all"))

    with pytest.raises(LLMError):
        provider.generate_teaching_plan("Explain this.", PNG)


def test_an_empty_response_is_rejected():
    provider, _client = generator(FakeResponse(text=""))

    with pytest.raises(LLMError):
        provider.generate_teaching_plan("Explain this.", PNG)


def test_an_api_failure_is_reported():
    provider, _client = generator(error=RuntimeError("503 Service Unavailable"))

    with pytest.raises(LLMError) as info:
        provider.generate_teaching_plan("Explain this.", PNG)

    assert "503" in str(info.value)


def test_a_missing_key_is_reported():
    provider = GeminiVisionProvider(api_key="", client=None)

    with pytest.raises(ProviderConfigurationError):
        provider.generate_teaching_plan("Explain this.", PNG)


def test_an_invalid_screenshot_is_rejected_before_the_api_is_called():
    provider, client = generator(FakeResponse(parsed=TeachingPlan.model_validate(plan_payload())))

    with pytest.raises(LLMError):
        provider.generate_teaching_plan("Explain this.", b"this is not an image")

    assert client.models.calls == []


# --- the mock provider ------------------------------------------------------


def test_the_mock_provider_returns_a_deterministic_plan():
    provider = MockVisionLLMProvider(delay_seconds=0)

    first = provider.generate_teaching_plan("Explain this graph.")
    second = provider.generate_teaching_plan("Explain this graph.")

    assert isinstance(first, TeachingPlan)
    assert first == second
    assert "Explain this graph." in first.objective
    assert [step.order for step in first.steps] == [1, 2]
    assert [step.step_id for step in first.steps] == ["step_1", "step_2"]
    assert first.conclusion


def test_the_mock_records_what_it_was_asked_to_plan():
    """A caller's request can be checked without the mock interpreting it."""
    provider = MockVisionLLMProvider(delay_seconds=0)

    provider.generate_teaching_plan(
        "make it simpler",
        screenshot=PNG,
        screenshot_size=(10, 20),
        context="The lesson that just finished.",
        guidance="Teach this plainly.",
    )

    assert provider.plan_requests == [
        {
            "user_query": "make it simpler",
            "screenshot": PNG,
            "screenshot_size": (10, 20),
            "context": "The lesson that just finished.",
            "guidance": "Teach this plainly.",
        }
    ]


def test_the_mock_plan_invents_no_visual_targets():
    """It cannot see the screen, so it must not pretend to."""
    plan = MockVisionLLMProvider(delay_seconds=0).generate_teaching_plan("What is this?")

    assert all(step.visual_actions == [] for step in plan.steps)


def test_the_mock_plan_handles_a_missing_query():
    plan = MockVisionLLMProvider(delay_seconds=0).generate_teaching_plan("")

    assert "no question" in plan.objective


# --- the prompt -------------------------------------------------------------


def test_the_teaching_prompt_states_the_core_principles():
    lowered = TEACHING_PLAN_SYSTEM_INSTRUCTION.lower()

    assert "not an image captioner" in lowered
    assert "one conceptual unit" in lowered
    assert "approximate" in lowered
    assert "the screenshot is the only source of visual truth" in lowered
    assert "never invent" in lowered
    assert "no markdown" in lowered
    assert "spoken aloud" in lowered
    assert "size the plan to the request" in lowered
    assert "structured teaching plan only" in lowered


def test_the_teaching_prompt_is_provider_neutral():
    lowered = TEACHING_PLAN_SYSTEM_INSTRUCTION.lower()

    for vendor in ("gemini", "openai", "claude", "anthropic"):
        assert vendor not in lowered


def test_the_prompt_builder_states_the_pixel_grid():
    prompt = build_teaching_plan_prompt("explain this", True, (1280, 720))

    assert "explain this" in prompt
    assert "1280 by 720" in prompt
    assert "1279" in prompt and "719" in prompt
    assert "0-1000" in prompt


def test_the_prompt_builder_handles_a_missing_query():
    assert "(no question was recognised)" in build_teaching_plan_prompt("   ", True)


def test_the_prompt_builder_forbids_visual_actions_without_a_screenshot():
    assert "do not use any visual actions" in build_teaching_plan_prompt("explain", False)
