"""Windows SendInput keyboard / mouse backend."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.windll.user32
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
KEYEVENTF_EXTENDEDKEY = 0x0001
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

SCAN = {
    "W": 0x11, "A": 0x1E, "S": 0x1F, "D": 0x20,
    "Q": 0x10, "E": 0x12, "R": 0x13, "T": 0x14, "G": 0x22, "F": 0x21,
    "Z": 0x2C, "X": 0x2D, "C": 0x2E, "V": 0x2F, "B": 0x30, "N": 0x31, "M": 0x32,
    "H": 0x23, "J": 0x24, "K": 0x25, "L": 0x26,
    "SPACE": 0x39, "ENTER": 0x1C, "LSHIFT": 0x2A, "ESC": 0x01, "CTRL": 0x1D,
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05,
    "LEFT": 0x4B, "RIGHT": 0x4D, "UP": 0x48, "DOWN": 0x50,
}
EXTENDED = {"LEFT", "RIGHT", "UP", "DOWN"}

# Arrow keys keep the scancode + EXTENDEDKEY encoding. A virtual-key variant
# (wVk=0x25, dwFlags=EXTENDEDKEY) was tried and measured 0/3 in the same window
# where scancode+EXTENDEDKEY scored 3/3, so VK is the wrong path for GZDoom and
# was reverted. The earlier 0/3 reading for arrows came from a stale gzdoom.exe
# owning the captured window, not from the encoding.
VK: dict[str, int] = {}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]


def _send(*items: INPUT) -> None:
    array = (INPUT * len(items))(*items)
    user32.SendInput(len(items), array, ctypes.sizeof(INPUT))


def _key_event(name: str, up: bool) -> INPUT:
    key = str(name).upper()
    item = INPUT(type=INPUT_KEYBOARD)
    scan = SCAN[key]
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if key in EXTENDED:
        # Arrow keys are extended scancodes; without this bit GZDoom ignores
        # them entirely (measured 0/3 plain vs 3/3 extended).
        flags |= KEYEVENTF_EXTENDEDKEY
    item.union.ki = KEYBDINPUT(0, scan, flags, 0, 0)
    return item


def key_down(name: str) -> None:
    _send(_key_event(name, False))


def key_up(name: str) -> None:
    _send(_key_event(name, True))


def tap_key(name: str, hold: float = 0.14) -> None:
    """Press and release one key with a hold long enough to be sampled.

    A 0.05s hold is shorter than a couple of frames at 60fps and sits right on
    the edge of what the engine's input polling observes; a use/shoot event that
    lands between polls is simply lost. W survived the short tap only because
    walking keeps changing the frame for a while afterwards, so the press was
    measurable even when the key itself was never processed. Firing and using
    leave no such trail, so the hold is widened here instead.
    """
    key_down(name)
    time.sleep(max(0.05, float(hold)))
    key_up(name)


def mouse_move(dx: int, dy: int) -> None:
    item = INPUT(type=INPUT_MOUSE)
    item.union.mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, 0)
    _send(item)


def move_cursor(x: int, y: int) -> None:
    """Park the pointer at an absolute screen point.

    Relative motion only reaches the window under the cursor, so mouse-look has
    to be preceded by moving the cursor inside the game; otherwise the deltas are
    applied to the desktop and the view never rotates.
    """
    user32.SetCursorPos(int(x), int(y))


def mouse_click(*, right: bool = False) -> None:
    down_flag = MOUSEEVENTF_RIGHTDOWN if right else MOUSEEVENTF_LEFTDOWN
    up_flag = MOUSEEVENTF_RIGHTUP if right else MOUSEEVENTF_LEFTUP
    down = INPUT(type=INPUT_MOUSE)
    down.union.mi = MOUSEINPUT(0, 0, 0, down_flag, 0, 0)
    up = INPUT(type=INPUT_MOUSE)
    up.union.mi = MOUSEINPUT(0, 0, 0, up_flag, 0, 0)
    _send(down, up)


def mouse_click_at(x: int, y: int, *, right: bool = False, hold: float = 0.12) -> None:
    """Click at an absolute screen point and keep the button down briefly.

    Two problems are fixed here. A delta-less SendInput click is delivered to
    whatever window sits under the pointer, which for a background game is the
    user's desktop, so the pointer is parked inside the target first. And
    down+up in one SendInput batch can be consumed in the same polling tick,
    leaving the game with zero frames of "button held" — an FPS then records no
    shot at all. Holding the button across a few frames makes the press real.
    """
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.03)
    down_flag = MOUSEEVENTF_RIGHTDOWN if right else MOUSEEVENTF_LEFTDOWN
    up_flag = MOUSEEVENTF_RIGHTUP if right else MOUSEEVENTF_LEFTUP
    down = INPUT(type=INPUT_MOUSE)
    down.union.mi = MOUSEINPUT(0, 0, 0, down_flag, 0, 0)
    _send(down)
    time.sleep(max(0.02, float(hold)))
    up = INPUT(type=INPUT_MOUSE)
    up.union.mi = MOUSEINPUT(0, 0, 0, up_flag, 0, 0)
    _send(up)


def foreground_window() -> int:
    return int(user32.GetForegroundWindow() or 0)


def set_foreground(hwnd: int) -> None:
    if hwnd:
        user32.SetForegroundWindow(int(hwnd))


def is_window(hwnd: int) -> bool:
    return bool(hwnd) and bool(user32.IsWindow(int(hwnd)))
