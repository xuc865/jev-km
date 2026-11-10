"""OS keyboard/mouse injection. Dispatches to Windows or macOS backends."""

from __future__ import annotations

import sys

if sys.platform == "darwin":
    from input_mac import (  # noqa: F401
        SCAN,
        foreground_window,
        is_window,
        key_down,
        key_up,
        mouse_click,
        mouse_click_at,
        mouse_move,
        move_cursor,
        set_foreground,
        tap_key,
    )
elif sys.platform.startswith("win"):
    from input_win import (  # noqa: F401
        SCAN,
        foreground_window,
        is_window,
        key_down,
        key_up,
        mouse_click,
        mouse_click_at,
        mouse_move,
        move_cursor,
        set_foreground,
        tap_key,
    )
else:
    raise RuntimeError(f"jevgamer input needs Windows or macOS, not {sys.platform!r}")
