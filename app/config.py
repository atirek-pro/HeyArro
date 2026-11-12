"""Application configuration.

Every setting is read from the environment, which is populated from the ``.env``
file in the project root when one exists. Real environment variables always win
over the file, so a one-off ``LLM_PROVIDER=mock python -m app.main`` still
overrides what ``.env`` says.
"""

import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env without overriding variables already set in the real environment.
load_dotenv(ENV_FILE, override=False)


def _read(name, default=None, cast=str):
    """Read ``name`` from the environment, falling back to its CLICKY_ alias."""
    raw = os.environ.get(name) or os.environ.get(f"CLICKY_{name}")

    if raw is None or raw == "":
        return default

    if cast is str:
        return raw

    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a valid %s; using %r", name, raw, cast.__name__, default)
        return default


# --- speech to text (local faster-whisper) ----------------------------------

# tiny | base | small | medium | large-v3 - bigger is slower but more accurate.
WHISPER_MODEL_SIZE = _read("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = _read("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _read("WHISPER_COMPUTE_TYPE", "int8")
# None means "detect the spoken language automatically".
WHISPER_LANGUAGE = _read("WHISPER_LANGUAGE")

# --- screen capture (mss) ---------------------------------------------------

# 0 = all monitors combined, 1 = primary monitor, 2 and above = the others.
SCREENSHOT_MONITOR = _read("SCREENSHOT_MONITOR", 1, int)

_screenshot_dir = _read("SCREENSHOT_DIR")
SCREENSHOT_DIR = (
    Path(_screenshot_dir)
    if _screenshot_dir
    else Path(tempfile.gettempdir()) / "clicky-screenshots"
)

# --- screenshot refinement (unstructured) -----------------------------------

# Off switch: "false" skips refinement and saves the screenshot to SCREENSHOT_DIR.
REFINEMENT_ENABLED = (_read("REFINEMENT_ENABLED", "true") or "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Each screenshot gets its own sub-folder here, holding the image and its segments.
# Created in the directory the app is started from unless overridden.
_refinement_dir = _read("REFINEMENT_DIR")
REFINEMENT_DIR = Path(_refinement_dir) if _refinement_dir else Path.cwd() / "screen-segments"

# hi_res | ocr_only | auto - hi_res runs layout detection (a model downloaded on first
# use) and finds panels, images and tables; ocr_only finds text blocks only.
REFINEMENT_STRATEGY = (_read("REFINEMENT_STRATEGY", "hi_res") or "hi_res").strip().lower()
# Tesseract language codes, comma separated (eng, deu, eng,fra...). Empty lets
# unstructured use its own default.
REFINEMENT_LANGUAGES = tuple(
    code.strip() for code in (_read("REFINEMENT_LANGUAGES", "") or "").split(",") if code.strip()
)
# Segments covering fewer pixels than this are dropped as noise.
REFINEMENT_MIN_SEGMENT_PIXELS = _read("REFINEMENT_MIN_SEGMENT_PIXELS", 16, int)

# --- relevant segment selection (Gemini) --------------------------------------

# After the transcript arrives, Gemini scores which segments matter for the question and
# the result is saved as relevant_ss_manifest.json. Needs refinement and GEMINI_API_KEY.
SEGMENT_SELECTION_ENABLED = (_read("SEGMENT_SELECTION_ENABLED", "true") or "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Empty = use GEMINI_MODEL. A smaller, faster model is usually enough for this step.
SEGMENT_SELECTION_MODEL = _read("SEGMENT_SELECTION_MODEL")
# Tokens the model may spend thinking before it answers: 0 = none (fastest), -1 = the
# model's own default.
SEGMENT_SELECTION_THINKING_BUDGET = _read("SEGMENT_SELECTION_THINKING_BUDGET", 0, int)

# After the segments are selected, Gemini plans how to respond (not the answer itself) and
# the result is saved as response_plan.json next to the manifests. Needs segment selection.
RESPONSE_PLANNING_ENABLED = (_read("RESPONSE_PLANNING_ENABLED", "true") or "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Empty = use GEMINI_MODEL.
RESPONSE_PLANNING_MODEL = _read("RESPONSE_PLANNING_MODEL")
# Planning benefits from a little thinking: 0 = none (fastest), -1 = the model's own default.
RESPONSE_PLANNING_THINKING_BUDGET = _read("RESPONSE_PLANNING_THINKING_BUDGET", 1024, int)

# After the plan is written, Gemini finds where each step's target is on the screenshot and
# the result is saved as plan_pointers.json next to the plan. Needs response planning.
PLAN_GROUNDING_ENABLED = (_read("PLAN_GROUNDING_ENABLED", "true") or "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Empty = use GEMINI_MODEL.
PLAN_GROUNDING_MODEL = _read("PLAN_GROUNDING_MODEL")
# Locating an element needs no reasoning: 0 = none (fastest), -1 = the model's own default.
PLAN_GROUNDING_THINKING_BUDGET = _read("PLAN_GROUNDING_THINKING_BUDGET", 0, int)

# After the plan is grounded, Gemini writes the final response from the plan: the sentences
# with the pointer of the step each belongs to, saved as final_response.json. Needs grounding.
FINAL_RESPONSE_ENABLED = (_read("FINAL_RESPONSE_ENABLED", "true") or "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Empty = use GEMINI_MODEL.
FINAL_RESPONSE_MODEL = _read("FINAL_RESPONSE_MODEL")
# The plan already did the reasoning: 0 = none (fastest), -1 = the model's own default.
FINAL_RESPONSE_THINKING_BUDGET = _read("FINAL_RESPONSE_THINKING_BUDGET", 0, int)

# --- presenting the final response ------------------------------------------

# How long the indicator says "Responding" before the first sentence is spoken.
RESPONDING_PAUSE_SECONDS = _read("RESPONDING_PAUSE_SECONDS", 2.0, float)
# Speak each sentence aloud. Off = the marks are still drawn.
TTS_ENABLED = (_read("TTS_ENABLED", "true") or "true").strip().lower() not in ("0", "false", "no", "off")
# "kokoro" (natural local voice, the default) or "windows" (the pyttsx3 SAPI voice).
TTS_PROVIDER = (_read("TTS_PROVIDER", "kokoro") or "kokoro").strip().lower()

# Kokoro: female voices - American af_heart / af_bella, British bf_emma / bf_isabella,
# Indian hf_alpha / hf_beta.
KOKORO_VOICE = (_read("KOKORO_VOICE", "af_heart") or "af_heart").strip()
# 1.0 = natural speed; 0.8 slower, 1.2 faster.
KOKORO_SPEED = _read("KOKORO_SPEED", 1.0, float)
# Pronunciation: en-us or en-gb. Empty = from the voice (American and Indian en-us, British en-gb).
KOKORO_LANG = _read("KOKORO_LANG")
# Where the Kokoro model and voices live; downloaded here when missing.
KOKORO_MODEL_DIR = Path(_read("KOKORO_MODEL_DIR") or PROJECT_ROOT / "tts_model")

# Windows voice: words per minute; None keeps the voice's own speed.
TTS_RATE_WPM = _read("TTS_RATE_WPM", None, int)
# Windows voice: 0.0 - 1.0; None keeps the system volume.
TTS_VOLUME = _read("TTS_VOLUME", None, float)
# Windows voice: a SAPI voice id; empty uses the default Windows voice.
TTS_VOICE_ID = _read("TTS_VOICE_ID")
# Draw each sentence's mark (point, circle, highlight...) over the screen.
OVERLAY_ENABLED = (_read("OVERLAY_ENABLED", "true") or "true").strip().lower() not in ("0", "false", "no", "off")
# The glass pill at the top of the screen showing Listening, Transcribing, Refining...
STATUS_INDICATOR_ENABLED = (_read("STATUS_INDICATOR_ENABLED", "true") or "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# --- vision LLM provider ----------------------------------------------------

# "gemini" (real), "mock" (offline echo), "openai" / "claude" (placeholders).
LLM_PROVIDER = (_read("LLM_PROVIDER", "gemini") or "gemini").strip().lower()

# Never hard-code a key here and never log it; it belongs in .env only.
GEMINI_API_KEY = _read("GEMINI_API_KEY")
GEMINI_MODEL = _read("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_TIMEOUT_SECONDS = _read("GEMINI_TIMEOUT_SECONDS", 60.0, float)

# Screenshots larger than this are rejected instead of being sent to a provider.
MAX_SCREENSHOT_BYTES = _read("MAX_SCREENSHOT_BYTES", 8 * 1024 * 1024, int)

# --- mock provider ----------------------------------------------------------

# Artificial latency, so an offline run still takes a realistic amount of time.
MOCK_LLM_DELAY_SECONDS = _read("MOCK_LLM_DELAY_SECONDS", 0.4, float)
