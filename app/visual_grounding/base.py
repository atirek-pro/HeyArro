"""The visual grounding contract.

The coordinator and teaching sequence depend on this abstraction, never on the
Gemini client or the image toolkit. A grounder turns an approximate target into
a precise one, and always degrades to the approximate target when it cannot -
grounding must never break a lesson.
"""

from abc import ABC, abstractmethod

from app.visual_grounding.models import GroundingRequest, Refinement


class GroundingError(RuntimeError):
    """Raised when a target cannot be refined."""


def refinement_key(request: GroundingRequest):
    """The identity of a target: what it is called and roughly where it is.

    The name is preferred over the request's full description, because the
    description folds in the sentence being taught - the same target mentioned
    in two different sentences must still only be refined once. Targets that
    have no name fall back to their description.
    """
    name = (request.label or request.description or "").strip().lower()
    return (name, request.approximate.as_tuple())


class VisualGrounder(ABC):
    """Refines approximate visual targets into precise ones."""

    @abstractmethod
    def refine_target(self, request: GroundingRequest) -> Refinement:
        """Refine one approximate target.

        Unusable input is not an error: implementations return the approximate
        rectangle with an ``APPROXIMATE`` status so callers can carry on.
        """

    def refine_targets(self, requests):
        """Refine several targets, refining each distinct one only once.

        Two requests that name the same thing in the same place are the same
        target, so the second one reuses the first one's answer instead of
        costing another request. The result lines up with the input.
        """
        requests = list(requests or ())
        known = {}
        refinements = []

        for request in requests:
            key = refinement_key(request)
            if key in known:
                refinements.append(known[key])
                continue
            refinement = self.refine_target(request)
            known[key] = refinement
            refinements.append(refinement)

        return refinements

    def clear_cache(self):
        """Forget anything cached for a previous interaction (optional)."""
        return None
