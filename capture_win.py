"""Capture a live game window by copying the pixels currently on screen.

PrintWindow on Unity often returns a cached frame, so the loop would keep
seeing a menu that was already closed. The process is made DPI aware first
so the client rectangle matches the screen copy.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from pathlib import Path

from PIL import Image, ImageGrab

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32
PW_RENDERFULLCONTENT = 2
SRCCOPY = 0x00CC0020


def _enable_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        user32.SetProcessDPIAware()


_enable_dpi()


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def find_window(title: str, window_class: str = "") -> int:
    """Match by title substring. Browser tabs append chrome like 'Astray 和另外 N 个页面'."""
    found = []
    wanted = (title or "").strip()
    if not wanted:
        raise RuntimeError("empty window title")

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        if wanted in buf.value and (not window_class or cls.value == window_class):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    if not found:
        raise RuntimeError(f"no visible {title} window")
    return found[0]


def find_window_by_pid(pid: int) -> int:
    """Find the visible top-level window owned by one specific process.

    GZDoom titles are the loaded level name ("Hydroelectric Plant - Freedoom:
    Phase 2"), not a fixed string. A stale gzdoom.exe from an earlier run keeps
    an identical title, and find_window() returns whichever window it enumerates
    first — so the adapter can capture and click a dead window while its own live
    game sits behind it, and every input silently does nothing. Binding to the
    pid we just launched removes that ambiguity entirely.
    """
    if not pid:
        return 0
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid:
            return True
        # Skip hidden helper windows that some engines create alongside the
        # real render surface.
        if user32.GetWindowTextLengthW(hwnd) <= 0:
            return True
        found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found[0] if found else 0


def capture_hwnd(hwnd: int) -> Image.Image | None:
    """Capture one specific window without resolving it by title."""
    if not hwnd:
        return None
    bits = _client_bits(hwnd)
    if bits is not None:
        return bits
    left, top, right, bottom = client_box(hwnd)
    grabbed = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True).convert("RGB").copy()
    if not _mostly_blank(grabbed):
        return grabbed
    win_left, win_top, win_right, win_bottom = window_box(hwnd)
    printed = _print_window(hwnd, win_right - win_left, win_bottom - win_top)
    if printed is None or _mostly_blank(printed):
        return grabbed
    crop = (
        max(0, left - win_left),
        max(0, top - win_top),
        max(1, right - win_left),
        max(1, bottom - win_top),
    )
    cropped = printed.crop(crop)
    if _mostly_blank(cropped):
        return grabbed
    return cropped.copy()


def focus_window(hwnd: int, *, force: bool = False) -> None:
    """Bring a game window forward. Prefer not to use this every capture frame."""
    foreground = user32.GetForegroundWindow()
    if foreground == hwnd and not force:
        return
    user32.ShowWindow(hwnd, 9)
    this_thread = kernel32.GetCurrentThreadId()
    front_thread = user32.GetWindowThreadProcessId(foreground, None)
    user32.AttachThreadInput(this_thread, front_thread, True)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(this_thread, front_thread, False)
    time.sleep(0.02)


def cursor_pos() -> tuple[int, int]:
    point = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return int(point.x), int(point.y)


def set_cursor(x: int, y: int) -> None:
    user32.SetCursorPos(int(x), int(y))


def window_box(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def client_box(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, origin.x + rect.right, origin.y + rect.bottom


def _mostly_blank(image: Image.Image) -> bool:
    sample = image.resize((32, 18)).convert("L")
    total = 0
    count = 0
    near_white = 0
    for y in range(sample.height):
        for x in range(sample.width):
            value = sample.getpixel((x, y))
            total += value
            count += 1
            if value > 245:
                near_white += 1
    mean = total / max(count, 1)
    # BitBlt on GPU/WebGL windows often returns solid black or solid white.
    return mean < 8 or near_white / max(count, 1) > 0.92


def _client_bits(hwnd: int) -> Image.Image | None:
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    width, height = rect.right, rect.bottom
    window_dc = user32.GetDC(hwnd)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    gdi32.SelectObject(memory_dc, bitmap)
    ok = gdi32.BitBlt(memory_dc, 0, 0, width, height, window_dc, 0, 0, SRCCOPY)
    header = BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    header.biWidth = width
    header.biHeight = -height
    header.biPlanes = 1
    header.biBitCount = 32
    buffer = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer, ctypes.byref(header), 0)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memory_dc)
    user32.ReleaseDC(hwnd, window_dc)
    if not ok:
        return None
    image = Image.frombuffer("RGB", (width, height), buffer, "raw", "BGRX", 0, 1).copy()
    if _mostly_blank(image):
        return None
    return image


def _print_window(hwnd: int, width: int, height: int) -> Image.Image | None:
    if width <= 0 or height <= 0:
        return None
    window_dc = user32.GetWindowDC(hwnd)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    gdi32.SelectObject(memory_dc, bitmap)
    ok = user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT)
    header = BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    header.biWidth = width
    header.biHeight = -height
    header.biPlanes = 1
    header.biBitCount = 32
    buffer = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer, ctypes.byref(header), 0)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memory_dc)
    user32.ReleaseDC(hwnd, window_dc)
    if not ok:
        return None
    return Image.frombuffer("RGB", (width, height), buffer, "raw", "BGRX", 0, 1).copy()


def capture(title: str, window_class: str = "", *, focus: bool = False) -> Image.Image:
    hwnd = find_window(title, window_class)
    if focus:
        focus_window(hwnd)
    bits = _client_bits(hwnd)
    if bits is not None:
        return bits
    left, top, right, bottom = client_box(hwnd)
    grabbed = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True).convert("RGB").copy()
    if not _mostly_blank(grabbed):
        return grabbed
    win_left, win_top, win_right, win_bottom = window_box(hwnd)
    printed = _print_window(hwnd, win_right - win_left, win_bottom - win_top)
    if printed is None or _mostly_blank(printed):
        return grabbed
    crop = (
        max(0, left - win_left),
        max(0, top - win_top),
        max(1, right - win_left),
        max(1, bottom - win_top),
    )
    cropped = printed.crop(crop)
    if _mostly_blank(cropped):
        return grabbed
    return cropped.copy()


def save(path: Path, title: str, window_class: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = capture(title, window_class)
    image.convert("RGB").save(path, quality=85)
    return path


if __name__ == "__main__":
    from knowledge import default_id, load
    game = load(default_id())
    target = Path(__file__).resolve().parent / "runs" / "frame.jpg"
    saved = save(target, game.window, game.window_class)
    with Image.open(saved) as image:
        print(f"{saved} {image.size[0]}x{image.size[1]}")
