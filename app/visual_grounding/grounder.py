"""The visual grounding pipeline.

The first vision pass understands *what* should be shown; this pipeline works
out *where* exactly it is:

    approximate target
        -> grow by the configured padding and crop that context region
        -> ask the refinement model to locate the target inside the crop
        -> map the crop-relative answer back into screenshot coordinates
        -> accept it only when the model is confident enough

Anything that goes wrong - bad crop, request failure, invalid structured output,
low confidence, a box outside the screenshot - falls back to the approximate
target, because a refinement must never make a reasonable target worse.

Refinements are cached for the current teaching interaction only, so the same
target is not refined twice while one lesson plays.
"""

import hashlib
import logging
import time

from app import config
from app.visual_grounding.base import GroundingError, VisualGrounder, refinement_key
from app.visual_grounding.coordinate_mapper import GroundingCoordinateMapper
from app.visual_grounding.cropper import (
    APPROXIMATE_COLOUR,
    REFINED_COLOUR,
    ScreenshotCropper,
    annotate,
    crop_region,
    save_debug_image,
)
from app.visual_grounding.models import (
    GroundingRequest,
    GroundingStatus,
    Refinement,
    rect_from_target,
    target_description,
    target_from_rect,
)

logger = logging.getLogger(__name__)

_CROP_COLOUR = (70, 140, 255)

# A box may round a few pixels past an edge; going past that means the model
# ignored the crop, and such an answer cannot be trusted to point anywhere exact.
_BOUNDS_TOLERANCE = 4


def _within(rect, bounds, tolerance=_BOUNDS_TOLERANCE):
    """True when ``rect`` lies inside ``bounds``, allowing a little rounding."""
    return (
        rect.left >= bounds.left - tolerance
        and rect.top >= bounds.top - tolerance
        and rect.right <= bounds.right + tolerance
        and rect.bottom <= bounds.bottom + tolerance
    )


class GroundingCache:
    """Remembers refinements for one teaching interaction."""

    def __init__(self):
        self._entries = {}

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, value):
        self._entries[key] = value

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)


def cache_key(screenshot, description, approximate):
    """Key a refinement by the screenshot, the target and its approximate box."""
    digest = hashlib.sha1(screenshot or b"").hexdigest()[:12]
    return (digest, (description or "").strip().lower(), approximate.as_tuple())


def ground_teaching_plan(grounder, plan, screenshot, screenshot_size):
    """Refine every visual target in ``plan`` before teaching starts.

    This is the *preparation* phase. All grounding happens here, once, on the
    screenshot the plan was produced from, so the teaching sequence can then draw
    and speak without ever making a request of its own.

    Targets are collected from every step, duplicates are refined once, and the
    plan comes back as a copy with refined targets written into the existing
    ``visual_action.target`` fields. A target that could not be refined keeps the
    approximation the first pass produced, so the lesson still runs.

    A failure of the grounding *mechanism* is deliberately not swallowed: it
    propagates to the caller so a run is abandoned rather than executed half
    prepared.
    """
    if grounder is None or not screenshot or not screenshot_size:
        return plan

    steps = list(plan.steps)
    logger.info("[TeachingGrounding] Preparation started")
    logger.info("[TeachingGrounding] Steps: %s", len(steps))

    requests = []
    locations = []
    for step_index, step in enumerate(steps):
        for action_index, action in enumerate(step.visual_actions):
            target = getattr(action, "target", None)
            approximate = rect_from_target(target)
            if approximate is None:
                continue
            requests.append(
                GroundingRequest(
                    screenshot=screenshot,
                    screenshot_size=tuple(screenshot_size),
                    # The step's own words describe what it is pointing at.
                    description=target_description(
                        target, getattr(step, "explanation", "")
                    ),
                    approximate=approximate,
                    label=getattr(target, "label", "") or "",
                )
            )
            locations.append((step_index, action_index, target))

    logger.info("[TeachingGrounding] Total visual targets: %s", len(requests))
    if not requests:
        logger.info("[TeachingGrounding] Preparation complete (nothing to refine)")
        return plan

    logger.info(
        "[TeachingGrounding] Unique targets: %s",
        len({refinement_key(item) for item in requests}),
    )
    logger.info("[TeachingGrounding] Refinement started")

    started = time.perf_counter()
    refinements = grounder.refine_targets(requests)
    logger.info(
        "[TeachingGrounding] Refinement completed in %.1fs", time.perf_counter() - started
    )

    actions = [list(step.visual_actions) for step in steps]
    refined_count = 0
    for (step_index, action_index, target), refinement in zip(locations, refinements):
        if not refinement.accepted:
            continue
        actions[step_index][action_index] = actions[step_index][action_index].model_copy(
            update={
                "target": target_from_rect(refinement.rect, target, refinement.confidence)
            }
        )
        refined_count += 1

    if not refined_count:
        logger.info(
            "[TeachingGrounding] Preparation complete (0/%s targets refined)", len(requests)
        )
        return plan

    grounded = plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"visual_actions": actions[index]})
                for index, step in enumerate(steps)
            ]
        }
    )
    logger.info(
        "[TeachingGrounding] Preparation complete (%s/%s targets refined)",
        refined_count,
        len(requests),
    )
    return grounded


class VisualGroundingService(VisualGrounder):
    """Refines approximate targets by cropping context and asking again."""

    def __init__(
        self,
        locator,
        cropper=None,
        cache=None,
        padding=None,
        threshold=None,
        debug=None,
        debug_dir=None,
    ):
        self._locator = locator
        self._cropper = cropper if cropper is not None else ScreenshotCropper()
        self._cache = cache if cache is not None else GroundingCache()
        self._padding = config.GROUNDING_CROP_PADDING if padding is None else padding
        self._threshold = (
            config.GROUNDING_CONFIDENCE_THRESHOLD if threshold is None else threshold
        )
        self._debug = config.VISUAL_GROUNDING_DEBUG if debug is None else debug
        self._debug_dir = config.GROUNDING_DEBUG_DIR if debug_dir is None else debug_dir

    @property
    def padding(self):
        return self._padding

    @property
    def threshold(self):
        return self._threshold

    @property
    def cache(self):
        return self._cache

    def clear_cache(self):
        """Forget refinements from previous interactions."""
        self._cache.clear()

    # --- the pipeline -------------------------------------------------------

    def refine_target(self, request: GroundingRequest) -> Refinement:
        approximate = request.approximate
        self._log(f"[Grounding] Target: {request.description}")
        self._log(f"[Grounding] Approximate: {approximate.describe()}")

        region = crop_region(approximate, self._padding, request.screenshot_size)
        if region is None:
            self._log("[Grounding] Crop: skipped (nothing usable to crop)")
            self._log("[Grounding] Result: APPROXIMATE")
            return Refinement(approximate, GroundingStatus.APPROXIMATE)
        self._log(f"[Grounding] Crop: {region.describe()}")

        key = cache_key(request.screenshot, request.description, approximate)
        cached = self._cache.get(key)
        if cached is not None:
            self._log(f"[Grounding] Refined: {cached.rect.describe()} (from cache)")
            self._log("[Grounding] Result: REFINED")
            return cached

        try:
            crop = self._cropper.crop(request.screenshot, region)
            grounded = self._locator.locate(crop.image, request.description, crop.size)
        except GroundingError as exc:
            self._log(f"[Grounding] Refinement failed: {exc}")
            self._log("[Grounding] Result: APPROXIMATE")
            return Refinement(approximate, GroundingStatus.APPROXIMATE)
        except Exception as exc:  # never let grounding break a lesson
            logger.warning("Visual grounding failed unexpectedly: %s", exc, exc_info=True)
            self._log("[Grounding] Result: APPROXIMATE")
            return Refinement(approximate, GroundingStatus.APPROXIMATE)

        mapper = GroundingCoordinateMapper.for_crop(crop)
        refined_in_crop = grounded.rect
        self._log(f"[Grounding] Refined: {refined_in_crop.describe()} (crop pixels)")

        screenshot_rect = mapper.crop_to_screenshot_rect(refined_in_crop)
        self._log(
            f"[Grounding] Crop origin {mapper.origin} -> screenshot {screenshot_rect.describe()}"
        )
        self._log(
            f"[Grounding] Confidence: {grounded.confidence:.2f} "
            f"(threshold {self._threshold:.2f})"
        )

        if self._debug:
            self._write_debug(request, crop, region, refined_in_crop, screenshot_rect)

        if refined_in_crop.is_empty:
            return self._reject(
                approximate, grounded.confidence, "the box has no area"
            )

        if not _within(refined_in_crop, mapper.crop_bounds()):
            # The model was told to stay inside the crop; a box that ignores
            # that cannot be trusted to point anywhere exact.
            return self._reject(
                approximate, grounded.confidence, "the box lies outside the crop"
            )

        if grounded.confidence < self._threshold:
            return self._reject(
                approximate, grounded.confidence, "confidence below threshold"
            )

        refinement = Refinement(screenshot_rect, GroundingStatus.REFINED, grounded.confidence)
        self._cache.put(key, refinement)
        self._log(f"[Grounding] Result: REFINED ({screenshot_rect.describe()})")
        return refinement

    # --- helpers ------------------------------------------------------------

    def _reject(self, approximate, confidence, reason):
        self._log(f"[Grounding] Result: REFINEMENT_REJECTED ({reason})")
        return Refinement(approximate, GroundingStatus.REFINEMENT_REJECTED, confidence)

    def _log(self, message):
        logger.info(message)

    def _write_debug(self, request, crop, region, refined_in_crop, screenshot_rect):
        """Save what the first pass thought beside what refinement found."""
        try:
            mapper = GroundingCoordinateMapper.for_crop(crop)
            approximate_in_crop = mapper.screenshot_to_crop_rect(request.approximate)

            save_debug_image(self._debug_dir, "crop", crop.image)
            annotated_crop = annotate(
                crop.image,
                [
                    (approximate_in_crop, APPROXIMATE_COLOUR, "first pass"),
                    (refined_in_crop, REFINED_COLOUR, "refined"),
                ],
            )
            path = save_debug_image(self._debug_dir, "crop-annotated", annotated_crop)

            annotated_screen = annotate(
                request.screenshot,
                [
                    (region, _CROP_COLOUR, "crop"),
                    (request.approximate, APPROXIMATE_COLOUR, "first pass"),
                    (screenshot_rect, REFINED_COLOUR, "refined"),
                ],
            )
            save_debug_image(self._debug_dir, "screenshot", annotated_screen)

            logger.info("[Grounding] Debug images written under %s", path.parent)
        except Exception:
            logger.debug("Could not write grounding debug images", exc_info=True)
