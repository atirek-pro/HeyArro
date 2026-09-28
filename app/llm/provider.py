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


def image_size(data):
    """Return ``(width, height)`` for a PNG screenshot, or None when unknown.

    Visual targets are meaningless without the frame they were measured in, so
    the model is always told the screenshot's exact pixel size. Callers normally
    pass the capture's own size; this reads it from the PNG header when they do
    not.
    """
    if not data or not data.startswith(PNG_SIGNATURE) or len(data) < 24:
        return None

    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


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
    "You are Hey Arro, a visual AI teaching companion that sits next to the user's "
    "mouse cursor.\n"
    "The user speaks a question out loud and a screenshot of their current screen is "
    "attached.\n"
    "\n"
    "Answer the question and, at the same time, decide what the user should see while "
    "you say each sentence. The screenshot is the only source of visual truth: every "
    "visual action must refer to something you can actually see in it.\n"
    "\n"
    "Choose teaching.mode from exactly one of:\n"
    "- direct: the user mainly needs a short, straightforward answer.\n"
    "- guided: walk the user through something using one or more visual steps.\n"
    "- explanatory: explain a concept or situation, using the screen where it helps.\n"
    "\n"
    "Each teaching step is ONE unit of explanation:\n"
    "- instruction: one concise sentence, short enough to be spoken aloud. Say what the "
    "shown thing means or why it matters - not merely its name.\n"
    "- visual_actions: what to show while that sentence is spoken. The visual appears "
    "before the sentence starts and stays until it ends, so it must match the sentence "
    "exactly.\n"
    "\n"
    "visual_actions is a list: a step may need none, one, or several. Each action has a "
    "type and a target:\n"
    "- 'point': move Arro's pointer to the spot;\n"
    "- 'highlight': soft filled region (or a radial glow around a point);\n"
    "- 'box': outlined rectangle around the region;\n"
    "- 'circle': outlined ellipse/circle around the region;\n"
    "- 'underline': emphasise a line of text, an expression or a row.\n"
    "\n"
    "Target geometry is given in screenshot pixels, in the exact pixel grid the "
    "request tells you the screenshot uses (x from 0 to width-1, y from 0 to "
    "height-1). Never use normalised, percentage or 0-1000 coordinates:\n"
    "- x and y are the CENTRE of the region; x is horizontal from the left edge and y "
    "is vertical from the top edge. They are always required.\n"
    "- width and height are the region's size. Give them whenever you can see how big "
    "the thing is (a row, a column, a cell, a button, a diagram, a code block). If you "
    "can only locate a point, leave them out.\n"
    "- label is a short name shown beside the visual; leave it empty when it adds "
    "nothing. confidence is how sure you are, from 0 to 1.\n"
    "\n"
    "Use visual actions like a teacher, not for decoration:\n"
    "- when you mention something visible, show it;\n"
    "- use 'highlight', 'box' or 'circle' for a region, and 'underline' for a line of "
    "text or an expression;\n"
    "- use 'point' only when pointing genuinely helps, and add it alongside a "
    "highlight/box when the user should look at one exact spot;\n"
    "- when a sentence concerns several things at once, give several actions in the "
    "same step (for example both operands being multiplied together).\n"
    "\n"
    "Teach, do not caption:\n"
    "- progress logically, one concept per step;\n"
    "- explain the relationship between the things you highlight and why they matter;\n"
    "- use the concrete values you can see (numbers, labels, code) when they are there;\n"
    "- never repeat the same idea, and never stop at 'here is X';\n"
    "- keep every instruction short - split a long explanation into more steps;\n"
    "- teach at the depth the request asks for: plain and in small steps for a beginner, "
    "balanced in the middle, and straight to the detail for an advanced request. A technical "
    "question is not an instruction to assume an expert.\n"
    "\n"
    "Only create a visual action when you can identify its target confidently. Never "
    "invent coordinates, regions, text or interface elements, and never guess a width "
    "or height you cannot see. If nothing on screen helps, leave visual_actions empty: "
    "a step with no visual is perfectly fine.\n"
    "\n"
    "response.text is one short spoken summary. When you give teaching steps, put what "
    "the user should hear into the steps' instructions instead of repeating it there.\n"
    "\n"
    "Never claim to see something that is not visible, never perform actions on the "
    "computer, and never return anything outside the requested structure."
)


def build_user_prompt(transcript, has_screenshot, screenshot_size=None, guidance=None):
    """Build the provider-neutral user turn for one interaction.

    The screenshot's pixel size is included whenever it is known: without it the
    model tends to answer in a normalised 0-1000 grid instead of the screenshot's
    own pixels, which would put every visual target in the wrong place.

    ``guidance`` is extra instruction written by the caller for this particular
    request - how deeply to teach it, for example. It is plain text, so this
    module needs to know nothing about how it was decided.
    """
    question = (transcript or "").strip() or "(no speech was recognised)"
    lines = [f"The user said: {question}", ""]

    if guidance:
        lines += ["How to teach this:", str(guidance).strip(), ""]

    if has_screenshot:
        lines.append("Answer using the attached screenshot of their screen.")
        if screenshot_size:
            width, height = screenshot_size
            lines.append(
                f"The screenshot is exactly {width} by {height} pixels. Give every visual "
                f"target's x, y, width and height as plain pixel counts in that grid - never "
                f"normalised, never percentages, never a 0-1000 scale."
            )
            lines.append(
                f"x runs from 0 at the left edge to {width - 1} at the right edge (centre "
                f"{width // 2}); y runs from 0 at the top edge to {height - 1} at the bottom "
                f"edge (centre {height // 2})."
            )
    else:
        lines.append(
            "No screenshot was captured for this request; if the answer depends on "
            "what is on their screen, say that you cannot see it."
        )

    return "\n".join(lines)


# --- models -----------------------------------------------------------------


@dataclass(frozen=True)
class VisionLLMRequest:
    """What the assistant asks a vision model to understand."""

    transcript: str
    screenshot: bytes | None = None
    # ``(width, height)`` of the screenshot in pixels, when known. Visual
    # targets are only meaningful in this frame, so it travels with the request.
    screenshot_size: tuple[int, int] | None = None
    # Extra instruction for this request, written by the caller (the depth to
    # teach at, for example). Empty means "answer the way you normally would".
    guidance: str = ""


# --- provider ---------------------------------------------------------------


class VisionLLMProvider(ABC):
    """Turns a transcript plus a screenshot into a structured teaching response.

    Implementations must return a fully validated HeyArroResponse and raise
    LLMError when one cannot be produced, so callers never see a partial or
    malformed answer.
    """

    name = "provider"

    @abstractmethod
    def process(self, request: VisionLLMRequest) -> HeyArroResponse:
        """Answer ``request``, raising LLMError when no valid answer can be produced."""

    def generate_teaching_plan(
        self,
        user_query,
        screenshot=None,
        screenshot_size=None,
        context=None,
        guidance=None,
    ) -> "TeachingPlan":
        """Plan how to teach ``user_query``, returning a validated plan.

        This is the generator behind follow-up lessons: the caller passes what
        the learner said, the current screen when showing something may help,
        the lesson being continued (``context``) and the depth to teach at
        (``guidance``), and gets back an ``app.teaching.plan.TeachingPlan`` -
        imported lazily, so the teaching package stays out of this module. The
        targets in it are approximate on purpose; grounding refines them later.

        Providers that cannot plan yet inherit this and raise, so a caller
        always gets a clear error instead of a silent no-op.
        """
        raise ProviderNotImplementedError(
            f"The {self.name} provider cannot generate teaching plans yet."
        )
