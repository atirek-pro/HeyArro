"""Placeholder ElevenLabs cloud TTS provider.

It satisfies the TTSProvider abstraction without calling the ElevenLabs API, so
a cloud voice can be added later without changing the coordinator or the rest of
the application. No key handling, streaming or cloud configuration is
implemented yet.
"""

from app.tts.provider import TTSNotImplementedError, TTSProvider


class ElevenLabsTTSProvider(TTSProvider):
    """Registered so the provider can be selected, but not implemented yet."""

    name = "elevenlabs"

    def speak(self, text):
        raise TTSNotImplementedError(
            "The ElevenLabs TTS provider is not implemented yet; "
            "set TTS_PROVIDER to windows or mock."
        )

    def stop(self):
        # speak() never starts any speech, so there is never anything to stop.
        return None
