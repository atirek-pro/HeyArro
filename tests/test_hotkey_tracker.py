from app.hotkey.global_hotkey import HotkeyTracker, normalize_key


def test_normalize_key():
    assert normalize_key("left ctrl") == "ctrl"
    assert normalize_key("right control") == "ctrl"
    assert normalize_key("alt") == "alt"
    assert normalize_key("a") == "a"


def test_press_and_release_transitions():
    tracker = HotkeyTracker(("ctrl", "alt"))
    assert tracker.handle("ctrl", True) is None
    assert tracker.handle("alt", True) == "pressed"
    assert tracker.active is True
    assert tracker.handle("alt", False) == "released"
    assert tracker.active is False


def test_release_order_does_not_matter():
    tracker = HotkeyTracker(("ctrl", "alt"))
    tracker.handle("alt", True)
    tracker.handle("ctrl", True)
    assert tracker.handle("ctrl", False) == "released"


def test_unrelated_keys_are_ignored():
    tracker = HotkeyTracker(("ctrl", "alt"))
    assert tracker.handle("a", True) is None
    assert tracker.handle("a", False) is None
    assert tracker.active is False


def test_key_repeats_do_not_retrigger():
    tracker = HotkeyTracker(("ctrl", "alt"))
    tracker.handle("ctrl", True)
    assert tracker.handle("alt", True) == "pressed"
    assert tracker.handle("alt", True) is None
    assert tracker.handle("ctrl", True) is None
