import subprocess
import sys
from pathlib import Path

import pytest

import app.coordinator
import app.teaching
import app.teaching.service as service_module
from app import config
from app.capture.provider import ScreenCaptureResult
from app.llm.response import (
    TeachingStep,
    VisualAction,
    VisualActionType,
    VisualTarget,
)
from app.teaching import (
    CoordinateMapper,
    CoordinateMappingError,
    NullVisualTeachingService,
    OverlayVisualTeachingService,
    ScreenGeometry,
    create_visual_teaching_service,
    has_bounds,
    iter_actions,
    iter_point_actions,
    normalize_label,
    resolve_action,
    resolve_actions,
    step_actions,
)


# --- helpers ----------------------------------------------------------------


def make_capture(width, height, monitor_index=1):
    return ScreenCaptureResult(
        image=b"\x89PNG\r\n\x1a\n",
        monitor_index=monitor_index,
        width=width,
        height=height,
        timestamp=0.0,
    )


def target(x=None, y=None, width=None, height=None, label=""):
    return VisualTarget(x=x, y=y, width=width, height=height, label=label)


def action(kind, x=None, y=None, width=None, height=None, label=""):
    return VisualAction(
        type=VisualActionType(kind), target=target(x, y, width, height, label)
    )


def step(instruction="Do it.", actions=None):
    return TeachingStep(instruction=instruction, visual_actions=actions or [])


class FakeOverlay:
    """Records the primitive lists a teaching service asks it to draw."""

    def __init__(self):
        self.calls = []
        self.clear_calls = 0

    def show_actions(self, primitives):
        self.calls.append(
            [(p.kind, p.x, p.y, p.width, p.height, p.label) for p in primitives]
        )

    def clear(self):
        self.clear_calls += 1


class Bare:
    """A duck-typed stand-in for a malformed action/target."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


# --- coordinate mapping -----------------------------------------------------


def test_unscaled_capture_maps_one_to_one():
    mapper = CoordinateMapper(ScreenGeometry(0, 0, 1920, 1080), 1920, 1080)

    assert mapper.to_screen(200, 150) == (200, 150)
    assert mapper.scale == (1.0, 1.0)
    assert mapper.scale_size(200, 100) == (200, 100)


def test_scaled_display_maps_screenshot_pixels_to_logical_pixels():
    # A 200% display: the screenshot is 3840x2160 but Qt reports 1920x1080.
    mapper = CoordinateMapper(ScreenGeometry(0, 0, 1920, 1080), 3840, 2160)

    assert mapper.to_screen(200, 150) == (100, 75)
    assert mapper.scale_size(200, 100) == (100, 50)
    # centre (100, 75) with a 100x50 screen-space size -> left/top (50, 50)
    assert mapper.to_screen_rect(200, 150, 200, 100) == (50, 50, 100, 50)


def test_screen_origin_is_added():
    mapper = CoordinateMapper(ScreenGeometry(1920, 0, 1920, 1080), 1920, 1080)

    assert mapper.to_screen(200, 150) == (2120, 150)


def test_mapper_can_be_built_from_a_capture():
    mapper = CoordinateMapper.from_capture(
        make_capture(3840, 2160), ScreenGeometry(0, 0, 1920, 1080)
    )

    assert mapper.to_screen(400, 300) == (200, 150)
    assert mapper.screen.width == 1920


def test_to_screen_rect_centres_the_region():
    mapper = CoordinateMapper(ScreenGeometry(0, 0, 1920, 1080), 1920, 1080)

    assert mapper.to_screen_rect(100, 50, 200, 80) == (0, 10, 200, 80)


def test_defensive_screenshot_size_is_rejected():
    with pytest.raises(CoordinateMappingError):
        CoordinateMapper(ScreenGeometry(0, 0, 100, 100), 0, 100)


# --- step helpers -----------------------------------------------------------


def test_normalize_label_trims_and_drops_blanks():
    assert normalize_label("  Settings  ") == "Settings"
    assert normalize_label("   ") is None
    assert normalize_label("") is None
    assert normalize_label(None) is None


def test_step_actions_is_always_a_list():
    built = step(actions=[action("box", 1, 2)])

    assert step_actions(built) == built.visual_actions
    assert step_actions(Bare()) == []


def test_iter_actions_walks_every_step():
    first = action("highlight", 1, 2, width=10, height=10, label="A")
    second = action("box", 3, 4, width=10, height=10, label="B")
    steps = [step("one", [first]), step("two"), step("three", [second])]

    assert list(iter_actions(steps)) == [first, second]
    assert list(iter_actions(None)) == []


def test_iter_point_actions_only_yields_pointers():
    point = action("point", 5, 6)
    box = action("box", 7, 8, width=10, height=10)
    steps = [step("one", [box, point])]

    assert [(s, a) for s, a in iter_point_actions(steps)] == [(steps[0], point)]


def test_has_bounds():
    assert has_bounds(target(1, 2, 10, 20)) is True
    assert has_bounds(target(1, 2)) is False
    assert has_bounds(target(1, 2, 10, None)) is False
    assert has_bounds(None) is False


# --- primitives -------------------------------------------------------------


def mapper_1000():
    return CoordinateMapper(ScreenGeometry(0, 0, 1000, 1000), 1000, 1000)


def test_resolve_action_maps_a_point_target():
    primitive = resolve_action(mapper_1000(), action("point", 200, 100, label="Settings"))

    assert (primitive.kind, primitive.x, primitive.y) == ("point", 200, 100)
    assert primitive.width is None and primitive.height is None
    assert primitive.label == "Settings"
    assert primitive.is_point is True


def test_resolve_action_maps_a_region_target():
    primitive = resolve_action(
        mapper_1000(), action("box", 200, 100, width=300, height=80, label="Panel")
    )

    assert (primitive.x, primitive.y, primitive.width, primitive.height) == (200, 100, 300, 80)
    assert primitive.has_bounds is True
    assert primitive.is_point is False


def test_resolve_action_scales_centre_and_size():
    mapper = CoordinateMapper(ScreenGeometry(0, 0, 1000, 500), 2000, 1000)
    primitive = resolve_action(mapper, action("circle", 200, 100, width=300, height=80))

    assert (primitive.x, primitive.y) == (100, 50)
    assert (primitive.width, primitive.height) == (150, 40)


def test_resolve_action_drops_a_blank_label():
    primitive = resolve_action(mapper_1000(), action("highlight", 10, 10, label="   "))

    assert primitive.label is None


def test_resolve_action_never_invents_geometry():
    assert resolve_action(mapper_1000(), Bare(type=VisualActionType.POINT, target=None)) is None
    assert (
        resolve_action(mapper_1000(), Bare(type=VisualActionType.POINT, target=Bare(x=None, y=5)))
        is None
    )
    assert resolve_actions(mapper_1000(), [None, Bare(type="box")]) == []


def test_resolve_actions_keeps_the_order():
    actions = [
        action("circle", 1, 2, width=10, height=10, label="1"),
        action("circle", 3, 4, width=10, height=10, label="5"),
    ]

    primitives = resolve_actions(mapper_1000(), actions)

    assert [p.label for p in primitives] == ["1", "5"]


# --- teaching service -------------------------------------------------------


def overlay_service(screen=None, capture=None):
    overlay = FakeOverlay()
    screen = screen or ScreenGeometry(0, 0, 1000, 1000)
    service = OverlayVisualTeachingService(overlay, screen)
    if capture is not None:
        service.prepare(capture)
    return service, overlay


def test_show_step_applies_every_action_at_once():
    service, overlay = overlay_service(capture=make_capture(1000, 1000))
    actions = [
        action("circle", 200, 100, width=40, height=40, label="1"),
        action("circle", 400, 300, width=40, height=40, label="5"),
        action("point", 400, 300),
    ]

    service.show_step(step("Multiply.", actions))

    assert overlay.calls == [
        [
            ("circle", 200, 100, 40, 40, "1"),
            ("circle", 400, 300, 40, 40, "5"),
            ("point", 400, 300, None, None, None),
        ]
    ]


def test_show_step_scales_regions_onto_the_screen():
    # 2000x1000 screenshot on a 1000x1000 screen: x halves, y is unchanged.
    service, overlay = overlay_service(capture=make_capture(2000, 1000))

    service.show_step(step("Look.", [action("box", 400, 200, width=200, height=100, label="A")]))

    assert overlay.calls == [[("box", 200, 200, 100, 100, "A")]]


def test_a_step_without_actions_fades_everything_out():
    service, overlay = overlay_service(capture=make_capture(1000, 1000))

    service.show_step(step("Nothing on screen helps here."))

    assert overlay.calls == [[]]  # an empty visual state, not a hard clear
    assert overlay.clear_calls == 0


def test_show_actions_without_a_prepared_capture_clears():
    service, overlay = overlay_service(capture=None)

    service.show_actions([action("box", 10, 10, width=10, height=10)])

    assert overlay.calls == []
    assert overlay.clear_calls == 1


def test_prepare_without_a_capture_means_nothing_is_shown():
    service, overlay = overlay_service(capture=make_capture(1000, 1000))
    assert service.mapper is not None

    service.prepare(None)
    service.show_step(step("Look.", [action("box", 10, 10, width=10, height=10)]))

    assert service.mapper is None
    assert overlay.calls == []
    assert overlay.clear_calls == 1


def test_hide_clears_the_overlay():
    service, overlay = overlay_service(capture=make_capture(1000, 1000))

    service.hide()

    assert overlay.clear_calls == 1


# --- null service -----------------------------------------------------------


def test_null_service_is_safe_to_call():
    service = NullVisualTeachingService()

    assert service.prepare(make_capture(10, 10)) is None
    assert service.show_step(step("x", [action("box", 1, 1, width=2, height=2)])) is None
    assert service.show_actions([action("point", 1, 1)]) is None
    assert service.hide() is None


# --- service selection ------------------------------------------------------


def test_factory_returns_null_service_when_teaching_is_disabled(monkeypatch):
    monkeypatch.setattr(config, "TEACHING_ENABLED", False)

    assert isinstance(create_visual_teaching_service(), NullVisualTeachingService)


# --- architecture -----------------------------------------------------------


def test_coordinator_contains_no_drawing_code():
    source = Path(app.coordinator.__file__).read_text(encoding="utf-8").lower()

    for primitive in ("qpainter", "qwidget", "qcolor", "qpixmap", "qpen"):
        assert primitive not in source


def test_teaching_contract_has_no_gui_code():
    source = Path(service_module.__file__).read_text(encoding="utf-8").lower()

    for primitive in ("pyside6", "qpainter", "qwidget", "qcolor"):
        assert primitive not in source


def test_importing_teaching_does_not_load_the_qt_overlay():
    root = Path(app.teaching.__file__).resolve().parents[2]
    code = "import sys, app.teaching; print('app.teaching.overlay' in sys.modules)"

    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
