import subprocess
import sys
from pathlib import Path

import pytest

import app.coordinator
import app.llm
from app import config
from app.llm import available_providers, get_llm_provider
from app.llm.claude_provider import ClaudeVisionProvider
from app.llm.gemini_provider import GeminiVisionProvider
from app.llm.mock_provider import MockVisionLLMProvider
from app.llm.openai_provider import OpenAIVisionProvider
from app.llm.provider import (
    LLMError,
    ProviderConfigurationError,
    ProviderNotImplementedError,
    SYSTEM_INSTRUCTION,
    UnsupportedProviderError,
    VisionLLMProvider,
    VisionLLMRequest,
    build_user_prompt,
    image_mime_type,
    image_size,
    validate_screenshot,
)
from app.llm.response import (
    HeyArroResponse,
    ResponseContent,
    ResponseTone,
    TargetType,
    TeachingMode,
    TeachingPlan,
)
from app.llm.worker import LLMWorker

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32

ALL_PROVIDERS = [
    GeminiVisionProvider,
    MockVisionLLMProvider,
    OpenAIVisionProvider,
    ClaudeVisionProvider,
]


# --- test doubles -----------------------------------------------------------


def make_response(text="answer", tone="neutral", mode="direct", steps=None):
    """Build a valid structured response the way Gemini would return one."""
    return HeyArroResponse(
        response=ResponseContent(text=text, tone=ResponseTone(tone)),
        teaching=TeachingPlan(mode=TeachingMode(mode), steps=steps or []),
    )


class FakeResponse:
    """A Gemini response carrying a structured payload."""

    def __init__(self, text=None, parsed=None):
        self.text = text
        self.parsed = parsed


class MalformedResponse:
    """No text, no parsed payload and no candidates, like a truncated response."""


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


def gemini(response=None, error=None, **kwargs):
    """A Gemini provider wired to a fake SDK client."""
    client = FakeClient(response, error)
    provider = GeminiVisionProvider(api_key="test-key", client=client, **kwargs)
    return provider, client


def gemini_parts(client):
    return client.models.calls[0]["contents"].parts


def structured(text="ok", **kwargs):
    """A fake Gemini response whose parsed payload is already structured."""
    return FakeResponse(parsed=make_response(text=text, **kwargs))


# --- abstraction ------------------------------------------------------------


@pytest.mark.parametrize("provider_class", ALL_PROVIDERS)
def test_every_provider_satisfies_the_abstraction(provider_class):
    assert issubclass(provider_class, VisionLLMProvider)
    assert isinstance(provider_class(), VisionLLMProvider)


@pytest.mark.parametrize("provider_class", ALL_PROVIDERS)
def test_every_provider_implements_process(provider_class):
    provider = provider_class()
    assert callable(provider.process)
    assert provider.name


def test_request_is_plain_data():
    request = VisionLLMRequest(transcript="hello", screenshot=PNG)
    assert request.transcript == "hello"
    assert request.screenshot == PNG


def test_request_screenshot_is_optional():
    request = VisionLLMRequest("hello")

    assert request.screenshot is None
    assert request.screenshot_size is None


# --- provider factory -------------------------------------------------------


def test_factory_returns_each_registered_provider():
    expected = {
        "gemini": GeminiVisionProvider,
        "openai": OpenAIVisionProvider,
        "claude": ClaudeVisionProvider,
        "mock": MockVisionLLMProvider,
    }

    for name, provider_class in expected.items():
        assert isinstance(get_llm_provider(name), provider_class)


def test_factory_lists_every_provider():
    assert available_providers() == ["claude", "gemini", "mock", "openai"]


def test_factory_uses_the_configured_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    assert isinstance(get_llm_provider(), OpenAIVisionProvider)


def test_factory_returns_gemini_when_configured_for_gemini(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    assert isinstance(get_llm_provider(), GeminiVisionProvider)


def test_factory_is_case_and_whitespace_insensitive():
    assert isinstance(get_llm_provider("  GeMiNi "), GeminiVisionProvider)


def test_unsupported_provider_name_produces_a_clear_error():
    with pytest.raises(UnsupportedProviderError) as info:
        get_llm_provider("hal9000")

    message = str(info.value)
    assert "hal9000" in message
    assert "gemini" in message


def test_missing_provider_name_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "")
    with pytest.raises(UnsupportedProviderError):
        get_llm_provider()


# --- gemini: request --------------------------------------------------------


def test_gemini_request_contains_the_transcript():
    provider, client = gemini(structured())

    provider.process(VisionLLMRequest(transcript="What is this error?", screenshot=PNG))

    texts = [part.text for part in gemini_parts(client) if getattr(part, "text", None)]
    assert any("What is this error?" in text for text in texts)


def test_gemini_request_contains_the_screenshot():
    provider, client = gemini(structured())

    provider.process(VisionLLMRequest(transcript="hello", screenshot=PNG))

    images = [part for part in gemini_parts(client) if getattr(part, "inline_data", None)]
    assert len(images) == 1
    assert images[0].inline_data.data == PNG
    assert images[0].inline_data.mime_type == "image/png"


def test_gemini_sends_the_neutral_system_instruction():
    provider, client = gemini(structured())

    provider.process(VisionLLMRequest(transcript="hello", screenshot=PNG))

    assert client.models.calls[0]["config"].system_instruction == SYSTEM_INSTRUCTION


def test_gemini_requests_structured_json_output():
    provider, client = gemini(structured())

    provider.process(VisionLLMRequest(transcript="hello", screenshot=PNG))

    request_config = client.models.calls[0]["config"]
    assert request_config.response_mime_type == "application/json"
    assert request_config.response_schema is HeyArroResponse


def test_gemini_uses_the_configured_model():
    provider, client = gemini(structured(), model="gemini-test-model")

    provider.process(VisionLLMRequest("hello", PNG))

    assert client.models.calls[0]["model"] == "gemini-test-model"


def test_gemini_missing_screenshot_sends_text_only():
    provider, client = gemini(structured())

    provider.process(VisionLLMRequest(transcript="hello", screenshot=None))

    assert not [part for part in gemini_parts(client) if getattr(part, "inline_data", None)]
    texts = " ".join(part.text for part in gemini_parts(client) if getattr(part, "text", None))
    assert "hello" in texts


def test_gemini_tells_the_model_the_screenshot_pixel_grid():
    provider, client = gemini(structured())

    provider.process(VisionLLMRequest("hi", png_bytes(1280, 720)))

    texts = " ".join(part.text for part in gemini_parts(client) if getattr(part, "text", None))
    assert "1280 by 720" in texts


def test_gemini_prefers_the_size_carried_by_the_request():
    provider, client = gemini(structured())

    provider.process(
        VisionLLMRequest("hi", png_bytes(1280, 720), screenshot_size=(800, 600))
    )

    texts = " ".join(part.text for part in gemini_parts(client) if getattr(part, "text", None))
    assert "800 by 600" in texts
    assert "1280 by 720" not in texts


# --- gemini: structured response --------------------------------------------


def test_gemini_converts_structured_output_into_hey_arro_response():
    expected = make_response(text="The screen shows an error dialog.", mode="explanatory")
    provider, _client = gemini(FakeResponse(parsed=expected))

    result = provider.process(VisionLLMRequest("what is this?", PNG))

    assert isinstance(result, HeyArroResponse)
    assert result.response.text == "The screen shows an error dialog."
    assert result.teaching.mode is TeachingMode.EXPLANATORY


def test_gemini_validates_a_parsed_mapping():
    provider, _client = gemini(
        FakeResponse(
            parsed={
                "response": {"text": "Open Settings to fix this.", "tone": "instructional"},
                "teaching": {
                    "mode": "guided",
                    "steps": [
                        {
                            "instruction": "Open Settings.",
                            "visual_actions": [
                                {
                                    "type": "box",
                                    "target": {
                                        "x": 842,
                                        "y": 316,
                                        "width": 140,
                                        "height": 48,
                                        "label": "Settings",
                                    },
                                }
                            ],
                        }
                    ],
                },
            }
        )
    )

    result = provider.process(VisionLLMRequest("where?", PNG))

    assert result.response.tone is ResponseTone.INSTRUCTIONAL
    assert result.teaching.mode is TeachingMode.GUIDED
    action = result.teaching.steps[0].visual_actions[0]
    assert action.target.x == 842
    assert (action.target.width, action.target.height) == (140, 48)
    assert action.target.type is TargetType.REGION


def test_gemini_validates_structured_json_text():
    provider, _client = gemini(FakeResponse(text=make_response(text="a direct answer").model_dump_json()))

    result = provider.process(VisionLLMRequest("hi", PNG))

    assert result.response.text == "a direct answer"


def test_gemini_reads_structured_text_from_candidates_when_parsed_is_absent():
    payload = make_response(text="from candidates").model_dump_json()

    class Part:
        text = payload

    class Content:
        parts = [Part()]

    class Candidate:
        content = Content()

    class Response:
        parsed = None
        candidates = [Candidate()]

    provider, _client = gemini(Response())
    assert provider.process(VisionLLMRequest("hi", PNG)).response.text == "from candidates"


def test_gemini_invalid_structured_output_is_reported():
    provider, _client = gemini(
        FakeResponse(parsed={"response": {"text": "hi", "tone": "furious"}, "teaching": {"mode": "direct"}})
    )

    with pytest.raises(LLMError):
        provider.process(VisionLLMRequest("hi", PNG))


def test_gemini_malformed_json_text_is_reported():
    provider, _client = gemini(FakeResponse(text="not json at all"))

    with pytest.raises(LLMError):
        provider.process(VisionLLMRequest("hi", PNG))


def test_gemini_empty_response_is_reported():
    provider, _client = gemini(FakeResponse(text=""))
    with pytest.raises(LLMError):
        provider.process(VisionLLMRequest("hi", PNG))


def test_gemini_malformed_response_is_reported():
    provider, _client = gemini(MalformedResponse())
    with pytest.raises(LLMError):
        provider.process(VisionLLMRequest("hi", PNG))


# --- gemini: failures -------------------------------------------------------


def test_gemini_api_failure_is_reported_as_an_llm_error():
    provider, _client = gemini(error=RuntimeError("503 Service Unavailable"))

    with pytest.raises(LLMError) as info:
        provider.process(VisionLLMRequest("hi", PNG))

    assert "503" in str(info.value)


def test_gemini_network_timeout_is_reported_as_an_llm_error():
    provider, _client = gemini(error=TimeoutError("request timed out"))

    with pytest.raises(LLMError):
        provider.process(VisionLLMRequest("hi", PNG))


def test_gemini_missing_api_key_is_reported():
    provider = GeminiVisionProvider(api_key="", client=None)

    with pytest.raises(ProviderConfigurationError):
        provider.process(VisionLLMRequest("hi", PNG))


def test_gemini_rejects_an_invalid_screenshot_before_calling_the_api():
    provider, client = gemini(structured())

    with pytest.raises(LLMError):
        provider.process(VisionLLMRequest("hi", b"this is not an image"))

    assert client.models.calls == []


def test_gemini_rejects_an_oversized_screenshot_before_calling_the_api():
    provider, client = gemini(structured())

    with pytest.raises(LLMError):
        provider.process(VisionLLMRequest("hi", PNG + b"\x00" * config.MAX_SCREENSHOT_BYTES))

    assert client.models.calls == []


# --- screenshot handling ----------------------------------------------------


def test_supported_image_formats_are_detected():
    assert image_mime_type(PNG) == "image/png"
    assert image_mime_type(JPEG) == "image/jpeg"
    assert image_mime_type(b"GIF89a") is None
    assert image_mime_type(None) is None
    assert image_mime_type(b"") is None


def test_validate_screenshot_returns_none_without_an_image():
    assert validate_screenshot(None) is None
    assert validate_screenshot(b"") is None


def test_validate_screenshot_accepts_png_and_jpeg():
    assert validate_screenshot(PNG) == "image/png"
    assert validate_screenshot(JPEG) == "image/jpeg"


def test_validate_screenshot_rejects_unsupported_formats():
    with pytest.raises(LLMError):
        validate_screenshot(b"GIF89a not supported")


def test_validate_screenshot_rejects_oversized_images():
    with pytest.raises(LLMError):
        validate_screenshot(PNG, max_bytes=8)


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


def test_image_size_reads_png_dimensions():
    assert image_size(png_bytes(1280, 720)) == (1280, 720)


def test_image_size_is_none_when_unknown():
    assert image_size(None) is None
    assert image_size(b"") is None
    assert image_size(b"not an image") is None
    assert image_size(JPEG) is None
    assert image_size(PNG) is None  # zeroed header: no usable size


# --- prompt -----------------------------------------------------------------


def test_system_instruction_is_provider_neutral_and_guards_against_invention():
    assert "screenshot" in SYSTEM_INSTRUCTION.lower()
    assert "invent" in SYSTEM_INSTRUCTION.lower()
    assert not any(vendor in SYSTEM_INSTRUCTION.lower() for vendor in ("gemini", "openai", "claude"))


def test_system_instruction_describes_the_teaching_modes():
    lowered = SYSTEM_INSTRUCTION.lower()
    assert "direct" in lowered
    assert "guided" in lowered
    assert "explanatory" in lowered


def test_user_prompt_carries_the_question():
    assert "What is this error?" in build_user_prompt("What is this error?", True)


def test_user_prompt_asks_for_a_transcript_only_answer_without_a_screenshot():
    assert "No screenshot" in build_user_prompt("hello", False)


def test_user_prompt_states_the_screenshot_pixel_size():
    prompt = build_user_prompt("explain this", True, (1280, 720))

    assert "1280 by 720" in prompt
    assert "1279" in prompt and "719" in prompt
    assert "0-1000" in prompt  # normalised coordinates are explicitly ruled out


def test_user_prompt_omits_the_size_when_it_is_unknown():
    assert "pixels" not in build_user_prompt("explain this", True)


# --- placeholders -----------------------------------------------------------


@pytest.mark.parametrize(
    "provider_class, label",
    [(OpenAIVisionProvider, "openai"), (ClaudeVisionProvider, "claude")],
)
def test_placeholders_report_that_they_are_not_implemented(provider_class, label):
    provider = provider_class()

    with pytest.raises(ProviderNotImplementedError) as info:
        provider.process(VisionLLMRequest("hello", PNG))

    assert label in str(info.value).lower()


@pytest.mark.parametrize("provider_class", [OpenAIVisionProvider, ClaudeVisionProvider])
def test_placeholders_also_raise_plain_not_implemented_error(provider_class):
    """Callers can catch either LLMError or NotImplementedError."""
    with pytest.raises(NotImplementedError):
        provider_class().process(VisionLLMRequest("hello", PNG))


# --- mock -------------------------------------------------------------------


def test_mock_returns_a_structured_response():
    result = MockVisionLLMProvider(delay_seconds=0).process(VisionLLMRequest("hello", PNG))

    assert isinstance(result, HeyArroResponse)
    assert "hello" in result.response.text
    assert str(len(PNG)) in result.response.text
    assert result.response.tone is ResponseTone.NEUTRAL
    assert result.teaching.steps == []


def test_mock_works_without_a_screenshot():
    result = MockVisionLLMProvider(delay_seconds=0).process(VisionLLMRequest("hello"))

    assert "no screenshot" in result.response.text


# --- worker -----------------------------------------------------------------


def test_worker_forwards_the_request_and_emits_the_response():
    class RecordingProvider(VisionLLMProvider):
        name = "recording"

        def __init__(self):
            self.seen = []

        def process(self, request):
            self.seen.append(request)
            return make_response(text="done")

    provider = RecordingProvider()
    worker = LLMWorker(provider, VisionLLMRequest("hi", PNG))
    finished = []
    worker.signals.finished.connect(finished.append)

    worker.run()

    assert provider.seen[0].transcript == "hi"
    assert provider.seen[0].screenshot == PNG
    assert [response.response.text for response in finished] == ["done"]


def test_worker_reports_provider_failures_without_raising():
    worker = LLMWorker(OpenAIVisionProvider(), VisionLLMRequest("hi", PNG))
    failed = []
    worker.signals.failed.connect(failed.append)

    worker.run()

    assert len(failed) == 1
    assert "not implemented" in failed[0].lower()


# --- architecture -----------------------------------------------------------


def test_coordinator_does_not_reference_vendor_specific_code():
    source = Path(app.coordinator.__file__).read_text(encoding="utf-8").lower()

    for vendor in ("gemini", "openai", "anthropic", "claude", "google"):
        assert vendor not in source

    # ...and it does not pick a provider either, so it cannot know one.
    assert "get_llm_provider" not in source


def test_shared_contract_has_no_vendor_specific_types():
    source = Path(app.llm.provider.__file__).read_text(encoding="utf-8").lower()

    for vendor in ("gemini", "openai", "anthropic", "claude", "google"):
        assert vendor not in source


def test_importing_the_llm_package_does_not_load_a_vendor_sdk():
    root = Path(app.llm.__file__).resolve().parents[2]
    code = (
        "import sys, app.llm; "
        "print([m for m in sys.modules if m.split('.')[0] in ('google', 'openai', 'anthropic')])"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
