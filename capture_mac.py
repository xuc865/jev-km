"""macOS window find / focus / capture via Quartz + screencapture.

Requires Screen Recording permission for the terminal/IDE that runs the agent,
and Accessibility permission for focus/cursor helpers.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

try:
    import Quartz
except ImportError as exc:  # pragma: no cover - darwin-only dependency
    raise RuntimeError(
        "macOS capture needs pyobjc-framework-Quartz. "
        "Install with: pip install 'pyobjc-framework-Quartz>=10' 'pyobjc-framework-Cocoa>=10'"
    ) from exc


def _window_list() -> list[dict]:
    options = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements
    )
    raw = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
    return [dict(item) for item in raw]


def _bounds(info: dict) -> tuple[int, int, int, int]:
    box = info.get("kCGWindowBounds") or {}
    x = int(box.get("X", 0))
    y = int(box.get("Y", 0))
    w = int(box.get("Width", 0))
    h = int(box.get("Height", 0))
    return x, y, x + w, y + h


def _match_score(info: dict, title: str, window_class: str) -> int:
    name = str(info.get("kCGWindowName") or "")
    owner = str(info.get("kCGWindowOwnerName") or "")
    layer = int(info.get("kCGWindowLayer") or 0)
    if layer != 0:
        return -1
    if window_class and window_class not in {owner, name}:
        # On macOS window_class maps to owner app name when set.
        if window_class.lower() not in owner.lower() and window_class.lower() not in name.lower():
            return -1
    wanted = title.strip().lower()
    blob = f"{name} {owner}".lower()
    if wanted not in blob:
        return -1
    # Prefer titled content windows over bare owner matches.
    score = 10
    if wanted in name.lower():
        score += 50
    if wanted in owner.lower():
        score += 5
    area = int(info.get("kCGWindowBounds", {}).get("Width", 0)) * int(
        info.get("kCGWindowBounds", {}).get("Height", 0)
    )
    score += min(area // 50000, 20)
    return score


def find_window(title: str, window_class: str = "") -> int:
    """Return CGWindowID whose title/owner contains ``title``."""
    wanted = (title or "").strip()
    if not wanted:
        raise RuntimeError("empty window title")
    best_id = 0
    best_score = -1
    for info in _window_list():
        score = _match_score(info, wanted, window_class or "")
        if score > best_score:
            wid = int(info.get("kCGWindowNumber") or 0)
            if wid:
                best_id = wid
                best_score = score
    if not best_id:
        raise RuntimeError(f"no visible {title} window")
    return best_id


def _info_for(hwnd: int) -> dict:
    for info in _window_list():
        if int(info.get("kCGWindowNumber") or 0) == int(hwnd):
            return info
    raise RuntimeError(f"window {hwnd} gone")


def focus_window(hwnd: int, *, force: bool = False) -> None:
    """Activate the owning app so the game receives keys / stays awake."""
    info = _info_for(hwnd)
    pid = int(info.get("kCGWindowOwnerPID") or 0)
    if not pid:
        return
    script = (
        f'tell application "System Events" to set frontmost of '
        f'(first process whose unix id is {pid}) to true'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    if force:
        time.sleep(0.02)


def cursor_pos() -> tuple[int, int]:
    event = Quartz.CGEventCreate(None)
    point = Quartz.CGEventGetLocation(event)
    return int(point.x), int(point.y)


def set_cursor(x: int, y: int) -> None:
    point = Quartz.CGPointMake(float(x), float(y))
    Quartz.CGWarpMouseCursorPosition(point)
    Quartz.CGAssociateMouseAndMouseCursorPosition(True)


def window_box(hwnd: int) -> tuple[int, int, int, int]:
    return _bounds(_info_for(hwnd))


def client_box(hwnd: int) -> tuple[int, int, int, int]:
    # Browser/canvas demos: content ≈ window bounds for borderless-ish chrome.
    return window_box(hwnd)


def _mostly_blank(image: Image.Image) -> bool:
    sample = image.resize((32, 18)).convert("L")
    total = 0
    count = 0
    near_white = 0
    for yy in range(sample.height):
        for xx in range(sample.width):
            value = sample.getpixel((xx, yy))
            total += value
            count += 1
            if value > 245:
                near_white += 1
    mean = total / max(count, 1)
    return mean < 8 or near_white / max(count, 1) > 0.92


def _screencapture_window(hwnd: int) -> Image.Image | None:
    tool = shutil.which("screencapture") or "/usr/sbin/screencapture"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [tool, "-x", "-o", "-l", str(int(hwnd)), str(path)],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not path.is_file() or path.stat().st_size < 32:
            return None
        image = Image.open(path).convert("RGB").copy()
        if _mostly_blank(image):
            return None
        return image
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _quartz_image(hwnd: int) -> Image.Image | None:
    image_ref = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        int(hwnd),
        Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution,
    )
    if not image_ref:
        return None
    width = Quartz.CGImageGetWidth(image_ref)
    height = Quartz.CGImageGetHeight(image_ref)
    if width <= 0 or height <= 0:
        return None
    bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)
    data_provider = Quartz.CGImageGetDataProvider(image_ref)
    data = Quartz.CGDataProviderCopyData(data_provider)
    raw = bytes(data)
    # BGRA → RGB
    out = bytearray()
    bpp = 4
    for y in range(height):
        start = y * bytes_per_row
        chunk = raw[start : start + width * bpp]
        for i in range(0, len(chunk), bpp):
            b, g, r = chunk[i], chunk[i + 1], chunk[i + 2]
            out.extend((r, g, b))
    image = Image.frombytes("RGB", (width, height), bytes(out))
    if _mostly_blank(image):
        return None
    return image


def capture(title: str, window_class: str = "", *, focus: bool = False) -> Image.Image:
    hwnd = find_window(title, window_class)
    if focus:
        focus_window(hwnd)
    image = _screencapture_window(hwnd)
    if image is not None:
        return image
    image = _quartz_image(hwnd)
    if image is not None:
        return image
    left, top, right, bottom = client_box(hwnd)
    # Last resort: full-display crop via Quartz display capture is heavy; raise.
    raise RuntimeError(
        f"macOS capture failed for {title!r} (hwnd={hwnd}, box={left},{top},{right},{bottom}). "
        "Grant Screen Recording to this terminal/IDE and retry."
    )


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
