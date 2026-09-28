"""Global hotkey detection for Ctrl + Alt (press and release).

Uses the ``keyboard`` package, which installs a low-level Windows keyboard
hook, so it works while the application is in the background and unfocused.
"""

import threading

from PySide6.QtCore import QObject, Signal

import keyboard


_KEY_ALIASES = {
    "ctrl": ("ctrl", "control"),
    "alt": ("alt",),
}


def normalize_key(name):
    """Map a raw key name reported by ``keyboard`` to a canonical name."""
    if not name:
        return ""
    lowered = name.lower()
    for canonical, aliases in _KEY_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                return canonical
    return lowered


class HotkeyTracker:
    """Track the pressed state of a key combination (pure logic, no OS access)."""

    def __init__(self, keys):
        self.keys = frozenset(keys)
        self._pressed = set()
        self._active = False

    @property
    def active(self):
        return self._active

    def handle(self, name, is_down):
        """Update state and return 'pressed', 'released' or None on transitions."""
        key = normalize_key(name)
        if key not in self.keys:
            return None

        if is_down:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)

        active = self.keys.issubset(self._pressed)
        if active and not self._active:
            self._active = True
            return "pressed"
        if not active and self._active:
            self._active = False
            return "released"
        return None

    def reset(self):
        self._pressed.clear()
        self._active = False


class GlobalHotkey(QObject):
    """Emits ``pressed`` / ``released`` signals for a global key combination."""

    pressed = Signal()
    released = Signal()

    def __init__(self, keys=("ctrl", "alt")):
        super().__init__()
        self._tracker = HotkeyTracker(keys)
        self._hook = None
        self._lock = threading.Lock()

    @property
    def active(self):
        return self._tracker.active

    def start(self):
        """Install the global keyboard hook."""
        if self._hook is None:
            self._hook = keyboard.hook(self._on_event)

    def stop(self):
        """Remove the global keyboard hook."""
        if self._hook is not None:
            keyboard.unhook(self._hook)
            self._hook = None
        self._tracker.reset()

    def _on_event(self, event):
        is_down = event.event_type == keyboard.KEY_DOWN
        with self._lock:
            transition = self._tracker.handle(event.name, is_down)

        if transition == "pressed":
            self.pressed.emit()
        elif transition == "released":
            self.released.emit()
