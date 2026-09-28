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


def _read_flag(name, default=False):
    """Read a boolean flag, treating "1/true/yes/on" (any case) as True."""
    raw = os.environ.get(name) or os.environ.get(f"CLICKY_{name}")
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- assistant behaviour ----------------------------------------------------

# How long the transcript bubble stays on screen next to the cursor.
TRANSCRIPT_DISPLAY_MS = _read("TRANSCRIPT_DISPLAY_MS", 4000, int)

# How long the RESPONDING state lasts before returning to IDLE.
RESPONDING_HOLD_MS = _read("RESPONDING_HOLD_MS", 2500, int)

# How long the ERROR state stays visible before returning to IDLE.
ERROR_HOLD_MS = _read("ERROR_HOLD_MS", 1200, int)

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

# --- text to speech (voice output) ------------------------------------------

# "windows" (local SAPI5 voice via pyttsx3), "mock" (records but stays silent),
# "elevenlabs" (placeholder; reports "not implemented yet").
TTS_PROVIDER = (_read("TTS_PROVIDER", "windows") or "windows").strip().lower()

# Optional tuning for the Windows voice; None keeps the engine's own defaults.
TTS_RATE_WPM = _read("TTS_RATE_WPM", None, int)
TTS_VOLUME = _read("TTS_VOLUME", None, float)
TTS_VOICE_ID = _read("TTS_VOICE_ID")

# --- visual teaching overlay ------------------------------------------------

# Draw Arro's pointer and target highlights on a transparent, click-through
# overlay. Turn off to run headless or to fall back to voice-only answers.
TEACHING_ENABLED = _read_flag("TEACHING_ENABLED", True)

# Development aid only: show the raw response text in the cursor caption again.
# The normal flow shows the teaching overlay instead.
DEBUG_RESPONSE_CAPTION = _read_flag("DEBUG_RESPONSE_CAPTION", False)

# --- visual grounding (target refinement) -----------------------------------

# Refine each approximate target with a second, high-resolution vision pass.
# Off means the first pass's coordinates are used exactly as they always were.
VISUAL_GROUNDING_ENABLED = _read_flag("VISUAL_GROUNDING_ENABLED", True)

# Context (in screenshot pixels) added around an approximate target before the
# refinement crop is taken. The first pass may already be inaccurate, so the
# crop deliberately includes surroundings.
GROUNDING_CROP_PADDING = _read("GROUNDING_CROP_PADDING", 240, int)

# A refinement is only accepted when the model is at least this confident;
# otherwise the approximate target is kept.
GROUNDING_CONFIDENCE_THRESHOLD = _read("GROUNDING_CONFIDENCE_THRESHOLD", 0.70, float)

# Development aid: save the crop and annotated images for each refinement so the
# first pass's guess can be compared with what refinement found.
VISUAL_GROUNDING_DEBUG = _read_flag("VISUAL_GROUNDING_DEBUG", False)

_grounding_dir = _read("GROUNDING_DEBUG_DIR")
GROUNDING_DEBUG_DIR = (
    Path(_grounding_dir)
    if _grounding_dir
    else Path(tempfile.gettempdir()) / "clicky-grounding"
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

# Artificial latency, so the RESPONDING state stays visible while testing.
MOCK_LLM_DELAY_SECONDS = _read("MOCK_LLM_DELAY_SECONDS", 0.4, float)
