"""Provider-neutral text-to-speech contract.

The application only depends on these types; no engine or vendor detail belongs
in this module. Speech is deliberately limited to plain text - a TTS provider
never learns about the model that produced the answer, the screenshot or the
teaching plan.
"""

from abc import ABC, abstractmethod


class TTSError(RuntimeError):
    """Raised when speech cannot be produced."""


class UnsupportedTTSProviderError(TTSError):
    """Raised when an unknown TTS provider name is requested."""


class TTSNotImplementedError(TTSError, NotImplementedError):
    """Raised by providers that are registered but not implemented yet."""


def normalize_text(text):
    """Return the speakable form of ``text``, or "" when there is nothing to say.

    Collapsing whitespace keeps the engines from reading out runs of spaces or
    blank lines; a blank result means "nothing to speak" rather than an error.
    """
    return " ".join(str(text or "").split())


class TTSProvider(ABC):
    """Speaks text out loud."""

    name = "tts"

    @abstractmethod
    def speak(self, text):
        """Speak ``text``.

        Blank text is a safe no-op. Failures are raised as TTSError so the
        caller can surface them through the normal error path.
        """

    @abstractmethod
    def stop(self):
        """Stop current speech when the engine supports it."""
