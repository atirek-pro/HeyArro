"""What Hey Arro is doing right now, as shown by the status indicator.

The coordinator and the response presenter report these; the indicator only
displays them. ``IDLE`` hides the indicator.
"""

from enum import Enum


class AssistantStatus(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    REFINING = "refining"
    PLANNING = "planning"
    RESPONDING = "responding"
    SPEAKING = "speaking"

    @property
    def label(self):
        return _LABELS[self]


_LABELS = {
    AssistantStatus.IDLE: "",
    AssistantStatus.LISTENING: "Listening",
    AssistantStatus.TRANSCRIBING: "Transcribing",
    AssistantStatus.REFINING: "Refining",
    AssistantStatus.PLANNING: "Planning",
    AssistantStatus.RESPONDING: "Responding",
    AssistantStatus.SPEAKING: "Speaking",
}
