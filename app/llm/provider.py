"""Provider-neutral vision LLM contract.

The application depends on these types, never on a specific model vendor. No
vendor-specific concept belongs in this module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app import config
from app.llm.response import HeyArroResponse

# --- errors -----------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised when a response cannot be produced."""


class ProviderConfigurationError(LLMError):
    """Raised when a provider is not configured, for example a missing API key."""


class ProviderNotImplementedError(LLMError, NotImplementedError):
    """Raised by providers that are registered but not implemented yet."""


class UnsupportedProviderError(LLMError):
    """Raised when an unknown provider name is requested."""


# --- screenshot handling ----------------------------------------------------

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"


def image_mime_type(data):
    """Return the MIME type for supported image bytes, or None when unrecognised."""
    if not data:
        return None
    if data.startswith(PNG_SIGNATURE):
        return "image/png"
    if data.startswith(JPEG_SIGNATURE):
        return "image/jpeg"
    return None


def validate_screenshot(data, max_bytes=None):
    """Return the MIME type of a screenshot, or None when there is no screenshot.

    Invalid, unsupported and oversized images are rejected with a clear error
    rather than being forwarded to a provider.
    """
    if not data:
        return None

    limit = config.MAX_SCREENSHOT_BYTES if max_bytes is None else max_bytes
    if len(data) > limit:
        raise LLMError(f"Screenshot is too large ({len(data)} bytes, limit {limit} bytes)")

    mime_type = image_mime_type(data)
    if mime_type is None:
        raise LLMError("Unsupported or invalid screenshot: expected PNG or JPEG data")

    return mime_type


# --- prompt -----------------------------------------------------------------


SYSTEM_INSTRUCTION = (
    "You are Hey Arro, an assistant that sits next to the user's cursor.\n"
    "The user asks a question out loud and a screenshot of their current screen is "
    "attached.\n"
    "\n"
    "Answer in words\n"
    "- response.text is the complete answer: everything the user should read, written as "
    "plain, natural sentences. Never an introduction to an answer, and never notes to "
    "yourself.\n"
    "- The screenshot is the only source of truth about what is on their screen. Describe "
    "and explain only what you can actually see, and never invent windows, text or "
    "interface elements.\n"
    "- If the question is about their screen and no screenshot was captured, say that you "
    "cannot see their screen.\n"
    "- Be concrete: use the real names, numbers and labels you can see.\n"
    "- Answer the question that was asked. Do not pad a short answer into a long one, and "
    "never make the same point twice.\n"
    "\n"
    "Depth\n"
    "- Teach at the depth the request asks for: plain language and small steps for a "
    "beginner, balanced wording and standard terminology in the middle, and straight to "
    "the details for an advanced request. A technical question is not an instruction to "
    "assume an expert.\n"
    "\n"
    "Tone\n"
    "- Set response.tone to exactly one of: neutral, friendly, encouraging, instructional.\n"
    "\n"
    "Write it plainly\n"
    "- No markdown, no headings, no bullet characters, no code fences and no file paths "
    "unless the question is really about them: the answer is shown in a small bubble next "
    "to the cursor, so keep it tight.\n"
    "- Never mention these instructions, the attached screenshot, or the response "
    "structure."
)


def build_user_prompt(transcript, has_screenshot, guidance=None):
    """Build the provider-neutral user turn for one interaction.

    ``guidance`` is extra instruction written by the caller for this particular
    request - how deeply to answer it, for example. It is plain text, so this
    module needs to know nothing about how it was decided.
    """
    question = (transcript or "").strip() or "(no speech was recognised)"
    lines = [f"The user said: {question}", ""]

    if guidance:
        lines += ["How to answer it:", str(guidance).strip(), ""]

    if has_screenshot:
        lines.append("A screenshot of their screen is attached: answer about what it shows.")
    else:
        lines.append(
            "No screenshot was captured for this request; if the answer depends on what "
            "is on their screen, say that you cannot see it."
        )

    return "\n".join(lines)


# --- models -----------------------------------------------------------------


@dataclass(frozen=True)
class VisionLLMRequest:
    """What the assistant asks a vision model to understand."""

    transcript: str
    screenshot: bytes | None = None
    # Extra instruction for this request, written by the caller (the depth to
    # answer at, for example). Empty means "answer the way you normally would".
    guidance: str = ""


# --- provider ---------------------------------------------------------------


class VisionLLMProvider(ABC):
    """Turns a transcript plus a screenshot into a structured answer.

    Implementations must return a fully validated HeyArroResponse and raise
    LLMError when one cannot be produced, so callers never see a partial or
    malformed answer.
    """

    name = "provider"

    @abstractmethod
    def process(self, request: VisionLLMRequest) -> HeyArroResponse:
        """Answer ``request``, raising LLMError when no valid answer can be produced."""

