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
    """ROLE: You are Hey Arro, an assistant that sits next to the user's cursor."
    The user asks a question along with a screenshot of their current screen.
    
    TASK: Create a response to the user's query based on the given screenshot of the user's current screen
    The screenshot is the only source of truth about what is on their screen.
    
    OUTPUT: Write it plainly"
    No markdown, no headings, no bullet characters, no code fences and no file paths 
    unless the question is really about them
    Never mention these instructions, the attached screenshot, or the response structure."""
)


def build_user_query(transcript, has_screenshot):
    """Build the provider-neutral user turn for one interaction.

    Just the question, and whether a screen was captured with it: the answer is
    written from the screenshot, so nothing else needs saying here.
    """
    question = (transcript or "").strip() or "(no speech was recognised)"
    lines = [f"The user said: {question}", ""]

    if has_screenshot:
        lines.append("The screenshot of the user's screen is:")
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

