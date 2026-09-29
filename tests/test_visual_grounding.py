"""Visual grounding: cropping context, refining, mapping back, and falling back."""

import pytest
from pydantic import ValidationError
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from app import config
from app.llm.response import (
    HeyArroResponse,
    ResponseContent,
    ResponseTone,
    TeachingMode,
    TeachingPlan,
    TeachingStep,
    VisualAction,
    VisualActionType,
    VisualTarget,
)
from app.visual_grounding import (
    CropResult,
    GroundedTarget,
    GroundingCache,
    GroundingCoordinateMapper,
    GroundingError,
    GroundingRequest,
    GroundingStatus,
    Rect,
    Refinement,
    ScreenshotCropper,
    VisualGrounder,
    VisualGroundingService,
    cache_key,
    create_visual_grounder,
    crop_region,
    ground_teaching_plan,
    rect_from_target,
    target_description,
    target_from_rect,
)
from app.teaching.plan import TeachingPlan as LessonPlan
from app.teaching.plan import TeachingStep as LessonStep
from app.visual_grounding.cropper import decode_png, encode_png
from app.visual_grounding.gemini_grounder import GeminiTargetLocator, build_grounding_prompt
from app.visual_grounding.worker import GroundingWorker


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    yield QApplication.instance() or QApplication([])


# --- helpers ----------------------------------------------------------------


def png_bytes(width=100, height=80, colour=(10, 20, 30)):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor(*colour))
    return encode_png(image)


def target(x, y, width=None, height=None, label=""):
    return VisualTarget(x=x, y=y, width=width, height=height, label=label)


def action(kind, x, y, width=None, height=None, label=""):
    return VisualAction(
        type=VisualActionType(kind), target=target(x, y, width, height, label)
    )


class Bare:
    """A duck-typed stand-in for a malformed target."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def request_for(approximate, description="the thing", size=(1000, 800)):
    return GroundingRequest(
        screenshot=png_bytes(8, 8),
        screenshot_size=size,
        description=description,
        approximate=approximate,
    )


class FakeCropper:
    """Returns a fixed crop, remembering the regions it was asked for."""

    def __init__(self, image=None, error=None):
        self._image = image
        self._error = error
        self.calls = []

    def crop(self, screenshot, region):
        self.calls.append(region)
        if self._error is not None:
            raise self._error
        return CropResult(
            image=self._image if self._image is not None else png_bytes(20, 20),
            origin_x=region.left,
            origin_y=region.top,
            width=region.width,
            height=region.height,
        )


class FakeLocator:
    """Returns a fixed crop-relative box, or raises."""

    def __init__(self, box=None, confidence=0.95, error=None):
        self._box = box or GroundedTarget(x=100, y=80, width=120, height=60, confidence=confidence)
        self._error = error
        self.calls = []

    def locate(self, crop_image, description, crop_size=None):
        self.calls.append((description, crop_size))
        if self._error is not None:
            raise self._error
        return self._box


def build_service(locator, cropper=None, **kwargs):
    return VisualGroundingService(locator=locator, cropper=cropper or FakeCropper(), **kwargs)


# --- Rect -------------------------------------------------------------------


def test_rect_from_centre_and_size():
    rect = Rect.from_center_size(500, 400, 100, 60)

    assert (rect.left, rect.top, rect.right, rect.bottom) == (450, 370, 550, 430)
    assert rect.width == 100 and rect.height == 60
    assert rect.centre == (500, 400)
    assert rect.is_empty is False


def test_a_target_without_a_size_is_an_empty_rect_at_its_point():
    rect = Rect.from_center_size(10, 20)

    assert (rect.left, rect.top, rect.right, rect.bottom) == (10, 20, 10, 20)
    assert rect.is_empty is True


def test_rect_expand_and_shift_and_clamp():
    rect = Rect(100, 100, 200, 150)

    assert rect.expand(50).as_tuple() == (50, 50, 200, 150)
    assert rect.shifted(10, -20).as_tuple() == (110, 80, 100, 50)
    assert rect.clamp(1000, 800).as_tuple() == (100, 100, 100, 50)

    outside = Rect(950, 750, 1200, 900).clamp(1000, 800)
    assert outside.as_tuple() == (950, 750, 50, 50)

    beyond = Rect(1200, 900, 1400, 1000).clamp(1000, 800)
    assert beyond.is_empty is True


# --- crop region ------------------------------------------------------------


def test_crop_region_adds_padding_around_the_target():
    approximate = Rect.from_center_size(500, 400, 100, 60)

    assert crop_region(approximate, 100, (1000, 800)) == Rect(350, 270, 650, 530)


def test_crop_region_clamps_at_the_left_and_top_edges():
    approximate = Rect.from_center_size(20, 10, 20, 20)

    assert crop_region(approximate, 200, (1000, 800)) == Rect(0, 0, 230, 220)


def test_crop_region_clamps_at_the_right_and_bottom_edges():
    approximate = Rect.from_center_size(980, 790, 20, 20)

    assert crop_region(approximate, 200, (1000, 800)) == Rect(770, 580, 1000, 800)


def test_crop_region_handles_a_target_partly_outside_the_screenshot():
    approximate = Rect.from_center_size(990, 400, 100, 60)

    assert crop_region(approximate, 100, (1000, 800)) == Rect(840, 270, 1000, 530)


def test_crop_region_gives_up_when_the_target_is_off_screen():
    approximate = Rect.from_center_size(1500, 900, 40, 40)

    assert crop_region(approximate, 100, (1000, 800)) is None


def test_crop_region_gives_up_on_a_bad_screenshot_size():
    approximate = Rect.from_center_size(10, 10, 40, 40)

    assert crop_region(approximate, 100, (0, 800)) is None
    assert crop_region(approximate, 100, (1000, -1)) is None


def test_crop_region_gives_up_on_a_target_too_small_to_crop():
    point = Rect.from_center_size(500, 400)

    assert crop_region(point, 0, (1000, 800)) is None
    assert crop_region(point, 5, (1000, 800)) is None  # still under the minimum crop size


def test_crop_region_expands_a_bare_point_by_the_padding():
    point = Rect.from_center_size(500, 400)

    assert crop_region(point, 100, (1000, 800)) == Rect(400, 300, 600, 500)


# --- coordinate mapping -----------------------------------------------------


def test_crop_coordinates_map_back_to_screenshot_coordinates():
    mapper = GroundingCoordinateMapper.from_origin(500, 300, 600, 500)

    assert mapper.crop_to_screenshot_point(220, 180) == (720, 480)
    assert mapper.crop_to_screenshot_rect(Rect(100, 50, 140, 90)) == Rect(600, 350, 640, 390)


def test_crop_mapper_round_trips_and_reports_its_bounds():
    mapper = GroundingCoordinateMapper.from_origin(120, 60, 200, 100)

    assert mapper.screenshot_to_crop_rect(Rect(150, 80, 190, 120)) == Rect(30, 20, 70, 60)
    assert mapper.crop_bounds() == Rect(0, 0, 200, 100)
    assert mapper.origin == (120, 60)
    assert mapper.size == (200, 100)


def test_mapper_can_be_built_from_a_crop_result():
    crop = CropResult(image=b"x", origin_x=10, origin_y=20, width=30, height=40)

    mapper = GroundingCoordinateMapper.for_crop(crop)

    assert mapper.origin == (10, 20)
    assert crop.rect == Rect(10, 20, 40, 60)


# --- bridges to the response model ------------------------------------------


def test_rect_from_target_uses_centre_and_size():
    assert rect_from_target(target(500, 400, 100, 60)) == Rect(450, 370, 550, 430)
    assert rect_from_target(target(500, 400)) == Rect(500, 400, 500, 400)
    assert rect_from_target(Bare(x="left", y=400)) is None
    assert rect_from_target(Bare(y=400)) is None
    assert rect_from_target(None) is None


def test_target_from_rect_keeps_label_and_reports_confidence():
    original = target(500, 400, 10, 10, label="Settings")

    refined = target_from_rect(Rect(450, 350, 570, 410), original, 0.91)

    assert (refined.x, refined.y) == (510, 380)
    assert (refined.width, refined.height) == (120, 60)
    assert refined.label == "Settings"
    assert refined.confidence == 0.91
    assert refined.type.value == "region"


def test_target_description_prefers_the_label_but_keeps_context():
    assert target_description(target(1, 1, label="Play button")) == "Play button"
    assert target_description(target(1, 1), "Click the play button.") == "Click the play button."
    assert target_description(target(1, 1, label="Play"), "Click play.") == "Play (Click play.)"
    assert target_description(None) == "the highlighted element"


def test_grounded_target_validates_its_box_and_confidence():
    grounded = GroundedTarget.model_validate(
        {"x": 10, "y": 20, "width": 30, "height": 40, "confidence": 0.8}
    )

    assert grounded.rect == Rect(10, 20, 40, 60)
    assert grounded.confidence == 0.8


def test_grounded_target_rejects_an_impossible_confidence():
    with pytest.raises(ValidationError):
        GroundedTarget.model_validate(
            {"x": 1, "y": 2, "width": 3, "height": 4, "confidence": 1.5}
        )


# --- the cropper ------------------------------------------------------------


def test_cropper_extracts_the_region_and_remembers_its_origin():
    image = QImage(60, 40, QImage.Format_RGB32)
    image.fill(QColor(0, 0, 0))
    image.setPixelColor(20, 10, QColor(255, 0, 0))
    screenshot = encode_png(image)

    crop = ScreenshotCropper().crop(screenshot, Rect(10, 5, 40, 25))

    assert (crop.origin_x, crop.origin_y) == (10, 5)
    assert (crop.width, crop.height) == (30, 20)
    cropped = decode_png(crop.image)
    assert (cropped.width(), cropped.height()) == (30, 20)
    assert cropped.pixelColor(10, 5).name() == "#ff0000"


def test_cropper_rejects_a_region_outside_the_screenshot():
    with pytest.raises(GroundingError):
        ScreenshotCropper().crop(png_bytes(40, 30), Rect(30, 20, 80, 60))


def test_cropper_rejects_an_empty_region():
    with pytest.raises(GroundingError):
        ScreenshotCropper().crop(png_bytes(40, 30), Rect(10, 10, 10, 10))


def test_cropper_rejects_bytes_that_are_not_an_image():
    with pytest.raises(GroundingError):
        ScreenshotCropper().crop(b"not an image", Rect(0, 0, 10, 10))


# --- the pipeline -----------------------------------------------------------


def test_a_confident_refinement_replaces_the_approximate_target():
    locator = FakeLocator(GroundedTarget(x=100, y=80, width=120, height=60, confidence=0.95))
    service = build_service(locator, padding=100)

    result = service.refine_target(request_for(Rect.from_center_size(500, 400, 100, 60)))

    assert result.status is GroundingStatus.REFINED
    assert result.accepted is True
    assert result.rect == Rect(450, 350, 570, 410)
    assert result.confidence == 0.95


def test_a_low_confidence_refinement_is_rejected_and_the_original_kept():
    locator = FakeLocator(GroundedTarget(x=10, y=10, width=10, height=10, confidence=0.4))
    service = build_service(locator, padding=100, threshold=0.7)
    approximate = Rect.from_center_size(500, 400, 100, 60)

    result = service.refine_target(request_for(approximate))

    assert result.status is GroundingStatus.REFINEMENT_REJECTED
    assert result.accepted is False
    assert result.rect == approximate
    assert result.confidence == 0.4


def test_confidence_exactly_at_the_threshold_is_accepted():
    locator = FakeLocator(GroundedTarget(x=100, y=80, width=120, height=60, confidence=0.7))
    service = build_service(locator, padding=100, threshold=0.7)

    result = service.refine_target(request_for(Rect.from_center_size(500, 400, 100, 60)))

    assert result.status is GroundingStatus.REFINED


def test_a_failing_locator_falls_back_to_the_approximate_target():
    locator = FakeLocator(error=GroundingError("api exploded"))
    service = build_service(locator, padding=100)
    approximate = Rect.from_center_size(500, 400, 100, 60)

    result = service.refine_target(request_for(approximate))

    assert result.status is GroundingStatus.APPROXIMATE
    assert result.rect == approximate


def test_an_unexpected_locator_error_also_falls_back():
    locator = FakeLocator(error=RuntimeError("something odd"))
    service = build_service(locator, padding=100)

    result = service.refine_target(request_for(Rect.from_center_size(500, 400, 100, 60)))

    assert result.status is GroundingStatus.APPROXIMATE


def test_a_failing_cropper_falls_back():
    cropper = FakeCropper(error=GroundingError("crop exploded"))
    service = build_service(FakeLocator(), cropper=cropper, padding=100)

    result = service.refine_target(request_for(Rect.from_center_size(500, 400, 100, 60)))

    assert result.status is GroundingStatus.APPROXIMATE


def test_a_target_that_cannot_be_cropped_is_not_refined():
    locator = FakeLocator()
    service = build_service(locator, padding=100)

    result = service.refine_target(request_for(Rect.from_center_size(5000, 5000, 40, 40)))

    assert result.status is GroundingStatus.APPROXIMATE
    assert locator.calls == []


def test_a_box_that_ignores_the_crop_bounds_is_rejected():
    locator = FakeLocator(GroundedTarget(x=0, y=0, width=900, height=900, confidence=0.99))
    service = build_service(locator, padding=100, threshold=0.5)
    approximate = Rect.from_center_size(990, 790, 20, 20)

    result = service.refine_target(request_for(approximate))

    # The model was told to stay inside the crop; this box covers everything.
    assert result.status is GroundingStatus.REFINEMENT_REJECTED
    assert result.rect == approximate


def test_an_empty_refined_box_is_rejected():
    locator = FakeLocator(GroundedTarget(x=20, y=20, width=0, height=0, confidence=0.99))
    service = build_service(locator, padding=100, threshold=0.5)
    approximate = Rect.from_center_size(500, 400, 100, 60)

    result = service.refine_target(request_for(approximate))

    assert result.status is GroundingStatus.REFINEMENT_REJECTED
    assert result.rect == approximate


def test_the_same_target_is_only_refined_once():
    locator = FakeLocator()
    service = build_service(locator, padding=100)
    request = request_for(Rect.from_center_size(500, 400, 100, 60))

    first = service.refine_target(request)
    second = service.refine_target(request)

    assert len(locator.calls) == 1
    assert first == second
    assert len(service.cache) == 1


def test_clearing_the_cache_refines_again():
    locator = FakeLocator()
    service = build_service(locator, padding=100)
    request = request_for(Rect.from_center_size(500, 400, 100, 60))

    service.refine_target(request)
    service.clear_cache()
    service.refine_target(request)

    assert len(locator.calls) == 2


def test_cache_keys_separate_different_targets_and_screenshots():
    first = cache_key(b"screen-a", "Play button", Rect(0, 0, 10, 10))
    second = cache_key(b"screen-a", "Play button", Rect(10, 0, 20, 10))
    third = cache_key(b"screen-b", "Play button", Rect(0, 0, 10, 10))
    same = cache_key(b"screen-a", "  play BUTTON ", Rect(0, 0, 10, 10))

    assert first != second
    assert first != third
    assert first == same


def test_debug_mode_writes_comparison_images(tmp_path):
    locator = FakeLocator(GroundedTarget(x=100, y=80, width=120, height=60, confidence=0.95))
    service = build_service(locator, padding=100, debug=True, debug_dir=tmp_path)

    result = service.refine_target(request_for(Rect.from_center_size(500, 400, 100, 60)))

    assert result.accepted is True
    names = sorted(path.name for path in tmp_path.glob("*.png"))
    assert len(names) == 3
    assert any("crop-annotated" in name for name in names)
    assert any("screenshot" in name for name in names)


# --- preparing a whole response ---------------------------------------------


def step_with(instruction, *actions):
    return TeachingStep(instruction=instruction, visual_actions=list(actions))


def make_response(*steps):
    return HeyArroResponse(
        response=ResponseContent(text="An answer.", tone=ResponseTone.NEUTRAL),
        teaching=TeachingPlan(mode=TeachingMode.GUIDED, steps=list(steps)),
    )


def pstep(explanation, *actions):
    """One step for ``plan_with()``: what is said, and what is shown for it."""
    return (explanation, list(actions))


def plan_with(*specs):
    return LessonPlan(
        objective="Understand this.",
        introduction="An opener.",
        steps=[
            LessonStep(
                step_id=f"step_{index}",
                order=index,
                objective=explanation,
                explanation=explanation,
                visual_actions=actions,
            )
            for index, (explanation, actions) in enumerate(specs, start=1)
        ],
    )


def test_refine_targets_refines_each_distinct_target_only_once():
    locator = FakeLocator()
    service = build_service(locator, padding=100)
    same = request_for(Rect.from_center_size(500, 400, 100, 60), description="Row one")
    also_same = request_for(Rect.from_center_size(500, 400, 100, 60), description="  row ONE ")
    other = request_for(Rect.from_center_size(200, 200, 40, 40), description="Row two")

    refinements = service.refine_targets([same, same, also_same, other])

    assert len(refinements) == 4  # one per request, in order
    assert len(locator.calls) == 2  # but only two distinct targets were refined
    assert refinements[0] == refinements[1] == refinements[2]
    assert refinements[3] != refinements[0]


def test_ground_teaching_plan_updates_every_target_before_teaching():
    locator = FakeLocator(GroundedTarget(x=100, y=80, width=120, height=60, confidence=0.95))
    service = build_service(locator, padding=100)
    lesson = plan_with(
        pstep("First.", action("box", 500, 400, 100, 60, "A")),
        pstep("Second.", action("highlight", 500, 400, 100, 60, "A")),
    )

    grounded = ground_teaching_plan(service, lesson, png_bytes(8, 8), (1000, 800))

    assert grounded is not lesson
    assert len(locator.calls) == 1  # both steps describe the same target
    first = grounded.steps[0].visual_actions[0]
    second = grounded.steps[1].visual_actions[0]
    assert (first.target.x, first.target.y) == (510, 380)
    assert (first.target.width, first.target.height) == (120, 60)
    assert second.target == first.target
    assert first.type is VisualActionType.BOX  # the primitive is untouched
    assert first.target.label == "A"
    # the plan the model produced is left alone
    assert lesson.steps[0].visual_actions[0].target.x == 500


def test_ground_teaching_plan_keeps_approximate_targets_when_refinement_is_rejected():
    locator = FakeLocator(GroundedTarget(x=1, y=1, width=2, height=2, confidence=0.1))
    service = build_service(locator, padding=100)
    lesson = plan_with(pstep("First.", action("box", 500, 400, 100, 60, "A")))

    grounded = ground_teaching_plan(service, lesson, png_bytes(8, 8), (1000, 800))

    assert grounded is lesson  # nothing changed, so nothing was rebuilt
    assert grounded.steps[0].visual_actions[0].target.x == 500


def test_ground_teaching_plan_without_a_grounder_or_screenshot_is_a_no_op():
    service = build_service(FakeLocator())
    lesson = plan_with(pstep("First.", action("box", 500, 400, 100, 60, "A")))

    assert ground_teaching_plan(None, lesson, b"png", (10, 10)) is lesson
    assert ground_teaching_plan(service, lesson, None, None) is lesson


def test_ground_teaching_plan_ignores_steps_without_targets():
    locator = FakeLocator()
    service = build_service(locator, padding=100)
    lesson = plan_with(pstep("Nothing on screen helps here."))

    assert ground_teaching_plan(service, lesson, png_bytes(8, 8), (1000, 800)) is lesson
    assert locator.calls == []


def test_a_grounder_that_blows_up_stops_the_preparation():
    class Exploding(VisualGrounder):
        def refine_target(self, request):
            raise RuntimeError("grounding exploded")

    lesson = plan_with(pstep("First.", action("box", 500, 400, 100, 60, "A")))

    # A failed preparation must reach the caller, so a half-prepared plan is
    # never executed.
    with pytest.raises(RuntimeError):
        ground_teaching_plan(Exploding(), lesson, png_bytes(8, 8), (1000, 800))


# --- service selection ------------------------------------------------------


def test_factory_is_off_when_grounding_is_disabled(monkeypatch):
    monkeypatch.setattr(config, "VISUAL_GROUNDING_ENABLED", False)

    assert create_visual_grounder() is None


def test_factory_is_off_when_teaching_is_disabled(monkeypatch):
    monkeypatch.setattr(config, "TEACHING_ENABLED", False)

    assert create_visual_grounder() is None


def test_factory_is_off_without_a_gemini_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")

    assert create_visual_grounder() is None


def test_factory_builds_the_refining_service(monkeypatch):
    monkeypatch.setattr(config, "VISUAL_GROUNDING_ENABLED", True)
    monkeypatch.setattr(config, "TEACHING_ENABLED", True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")

    assert isinstance(create_visual_grounder(), VisualGroundingService)


# --- the Gemini refinement call ---------------------------------------------


class FakeModels:
    def __init__(self, response=None, error=None):
        self.calls = []
        self._response = response
        self._error = error

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response, error)


class FakeResponse:
    def __init__(self, text=None, parsed=None):
        self.text = text
        self.parsed = parsed


def locator_with(response=None, error=None):
    client = FakeClient(response, error)
    return GeminiTargetLocator(api_key="test-key", client=client), client


def test_locator_decodes_the_structured_box():
    expected = GroundedTarget(x=12, y=34, width=56, height=78, confidence=0.88)
    locator, _client = locator_with(FakeResponse(parsed=expected))

    grounded = locator.locate(png_bytes(20, 20), "Play button", (20, 20))

    assert grounded == expected


def test_locator_sends_the_crop_and_the_target_description():
    locator, client = locator_with(
        FakeResponse(parsed=GroundedTarget(x=1, y=2, width=3, height=4, confidence=0.9))
    )

    locator.locate(png_bytes(20, 20), "the Save button", (640, 480))

    call = client.models.calls[0]
    parts = call["contents"].parts
    assert any(getattr(part, "inline_data", None) for part in parts)
    texts = " ".join(part.text for part in parts if getattr(part, "text", None))
    assert "the Save button" in texts
    assert "640" in texts and "480" in texts
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is GroundedTarget


def test_locator_falls_back_to_text_when_there_is_no_parsed_payload():
    payload = GroundedTarget(
        x=5, y=6, width=7, height=8, confidence=0.75
    ).model_dump_json()
    locator, _client = locator_with(FakeResponse(text=payload))

    grounded = locator.locate(png_bytes(20, 20), "thing", (20, 20))

    assert (grounded.x, grounded.y, grounded.width, grounded.height) == (5, 6, 7, 8)


def test_locator_reports_an_api_failure():
    locator, _client = locator_with(error=RuntimeError("503 unavailable"))

    with pytest.raises(GroundingError):
        locator.locate(png_bytes(20, 20), "thing", (20, 20))


def test_locator_reports_invalid_structured_output():
    locator, _client = locator_with(
        FakeResponse(parsed={"x": 1, "y": 2, "width": 3, "height": 4, "confidence": 9.9})
    )

    with pytest.raises(GroundingError):
        locator.locate(png_bytes(20, 20), "thing", (20, 20))


def test_locator_reports_an_empty_answer():
    locator, _client = locator_with(FakeResponse(text=""))

    with pytest.raises(GroundingError):
        locator.locate(png_bytes(20, 20), "thing", (20, 20))


def test_locator_reports_a_missing_key():
    locator = GeminiTargetLocator(api_key="", client=None)

    with pytest.raises(GroundingError):
        locator.locate(png_bytes(20, 20), "thing", (20, 20))


def test_locator_rejects_a_crop_that_is_not_an_image():
    locator, _client = locator_with(FakeResponse())

    with pytest.raises(GroundingError):
        locator.locate(b"not an image", "thing", (20, 20))


def test_the_grounding_prompt_states_the_crop_grid():
    prompt = build_grounding_prompt("the Files folder", (640, 480))

    assert "the Files folder" in prompt
    assert "640" in prompt and "480" in prompt


# --- the worker -------------------------------------------------------------


class ShiftingGrounder(VisualGrounder):
    """A grounder that moves every target, to prove the worker plumbs results."""

    def __init__(self, shift=(10, 20)):
        self._shift = shift

    def refine_target(self, request):
        return Refinement(
            request.approximate.shifted(*self._shift), GroundingStatus.REFINED, 0.9
        )


def test_worker_returns_the_plan_with_refined_targets():
    lesson = plan_with(pstep("First.", action("box", 100, 100, 40, 40, "Settings")))
    worker = GroundingWorker(ShiftingGrounder(), lesson, png_bytes(8, 8), (100, 100))
    finished = []
    worker.signals.finished.connect(finished.append)

    worker.run()

    assert len(finished) == 1
    refined = finished[0].steps[0].visual_actions[0].target
    # centre (100,100) shifted by (10,20)
    assert (refined.x, refined.y) == (110, 120)
    assert (refined.width, refined.height) == (40, 40)


def test_worker_reports_a_preparation_that_blows_up():
    class Exploding(VisualGrounder):
        def refine_target(self, request):
            raise RuntimeError("grounding exploded")

    lesson = plan_with(pstep("First.", action("box", 500, 400, 100, 60, "A")))
    worker = GroundingWorker(Exploding(), lesson, png_bytes(8, 8), (1000, 800))
    finished = []
    failed = []
    worker.signals.finished.connect(finished.append)
    worker.signals.failed.connect(failed.append)

    worker.run()

    # The caller must never receive a half-prepared plan to execute.
    assert finished == []
    assert failed == ["grounding exploded"]


def test_worker_survives_an_emit_after_the_signals_were_torn_down():
    """A slow refinement can outlive the window; emitting then must not raise."""
    lesson = plan_with(pstep("First.", action("box", 1, 1, 4, 4)))
    worker = GroundingWorker(ShiftingGrounder(), lesson, b"png", (10, 10))

    def torn_down():
        raise RuntimeError("Signal source has been deleted")

    worker._safe_emit(torn_down)


def test_cache_can_be_used_standalone():
    cache = GroundingCache()
    cache.put("k", "v")

    assert cache.get("k") == "v"
    assert len(cache) == 1

    cache.clear()

    assert cache.get("k") is None
    assert len(cache) == 0
