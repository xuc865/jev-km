"""Capture a live game window. Dispatches to Windows or macOS backends."""

from __future__ import annotations

import sys

if sys.platform == "darwin":
    from capture_mac import (  # noqa: F401
        capture,
        client_box,
        cursor_pos,
        find_window,
        focus_window,
        save,
        set_cursor,
        window_box,
    )
elif sys.platform.startswith("win"):
    from capture_win import (  # noqa: F401
        capture,
        capture_hwnd,
        client_box,
        cursor_pos,
        find_window,
        find_window_by_pid,
        focus_window,
        save,
        set_cursor,
        window_box,
    )
else:
    raise RuntimeError(f"jevgamer capture needs Windows or macOS, not {sys.platform!r}")
