"""Manual voice check: ``python -m app.tts "some text to speak"``.

Speaks a phrase through the configured TTS provider without the microphone,
transcription or vision pipeline, so the local voice can be validated on its
own. Defaults to the configured provider; override with e.g.
``TTS_PROVIDER=mock python -m app.tts``.
"""

import logging
import sys

from app.tts import TTSError, get_tts_provider

DEFAULT_PHRASE = "Hello, I am Arro. How can I help you?"


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = list(sys.argv[1:] if argv is None else argv)
    text = " ".join(args).strip() or DEFAULT_PHRASE

    provider = get_tts_provider()
    print(f"Speaking with {type(provider).__name__}: {text}")

    try:
        provider.speak(text)
    except TTSError as exc:
        print(f"Voice output failed: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
