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
