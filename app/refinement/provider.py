"""Provider-neutral screenshot refinement contract.

Refinement is the first step after a screenshot is received: it stores the
screenshot in its own folder and splits it into segments (text blocks, titles,
lists, tables, images...) saved next to it. The application depends on these
types, never on a specific document-parsing library.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ScreenRefinementError(RuntimeError):
    """Raised when a screenshot cannot be refined."""


@dataclass(frozen=True)
class ScreenSegment:
    """One region of the screen identified inside the screenshot.

    ``left``/``top`` are pixels from the top-left corner of the screenshot.
    ``path`` is the cropped image of the region, when it could be written.
    """

    index: int
    kind: str
    text: str
    left: int
    top: int
    width: int
    height: int
    path: str | None = None

    @property
    def box(self):
        return (self.left, self.top, self.left + self.width, self.top + self.height)


@dataclass(frozen=True)
class RefinementResult:
    """The folder that holds one screenshot and everything derived from it."""

    folder: str
    screenshot_path: str
    segments: list[ScreenSegment] = field(default_factory=list)
    strategy: str = "n/a"

    @property
    def segment_count(self):
        return len(self.segments)


class ScreenRefinementProvider(ABC):
    """Splits a captured screenshot into segments."""

    @abstractmethod
    def refine(self, capture, output_dir) -> RefinementResult:
        """Save ``capture`` under a new folder in ``output_dir`` and segment it."""


class SegmentSelectionError(RuntimeError):
    """Raised when the relevant segments cannot be selected."""


class ResponseCategory(str, Enum):
    """What kind of help a question asks for, given what is on screen."""

    TEACHING = "teaching"
    INSTRUCTING = "instructing"
    GUIDING = "guiding"
    INFORMATIVE = "informative"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class SelectionResult:
    """The segments that matter for one question, and what kind of help it asks for.

    ``scores`` maps a segment id to its score (1-10, higher is more relevant).
    ``confidence`` (0-1) and ``category_reason`` describe the ``response_category``.
    ``manifest_path`` is the ``relevant_ss_manifest.json`` that was written.
    """

    manifest_path: str
    scores: dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0
    response_category: ResponseCategory = ResponseCategory.FALLBACK
    confidence: float = 0.0
    category_reason: str = ""

    @property
    def selected_count(self):
        return len(self.scores)


class SegmentSelector(ABC):
    """Picks the segments of a refined screenshot that matter for a question."""

    @abstractmethod
    def select(self, refinement: RefinementResult, transcript: str) -> SelectionResult:
        """Categorise the question, score the relevant segments and write ``relevant_ss_manifest.json``."""
