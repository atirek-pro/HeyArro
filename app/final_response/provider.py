"""Provider-neutral contract for composing the final response.

Composing is the step after grounding: it turns the response plan into the
sentences the user will get, each tied to the plan step it belongs to and that
step's pointer on the screen. It draws nothing and speaks nothing. The
application depends on these types, never on a specific model vendor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.grounding.provider import GroundingResult


class FinalResponseError(RuntimeError):
    """Raised when the final response cannot be composed."""


@dataclass(frozen=True)
class FinalResponseResult:
    """The final response written for one question.

    ``response_path`` is the ``final_response.json`` that was written next to the
    plan and its pointers.
    """

    response_path: str
    sentence_count: int = 0
    seconds: float = 0.0


class ResponseComposer(ABC):
    """Writes the final response sentences from a grounded plan."""

    @abstractmethod
    def compose(self, grounding: GroundingResult) -> FinalResponseResult:
        """Read the plan and its pointers next to ``grounding.pointers_path``, then write ``final_response.json``."""
