"""Generic HTML5 adapter for the map/ H5 games.

The H5 catalogue entry supplies an index.html; the adapter opens it in a
dedicated Chrome window, captures the window, and exposes a small generic key
set (arrows/space/enter/WASD). The policy/JEV decides which key is best; card or
click-heavy games will need a click-target candidate builder in a subclass.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import capture
import harness
import input_backend

from ..types import Action, Observation

ROOT = Path(__file__).resolve().parents[2]
MAP_ROOT = ROOT / "map"
CATALOG = json.loads((MAP_ROOT / "games_catalog.json").read_text(encoding="utf-8-sig"))

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
]

ACTIONS = [
    Action("up", "up / forward", kind="tap", keys=("UP",), seconds=0.1),
    Action("down", "down / back", kind="tap", keys=("DOWN",), seconds=0.1),
    Action("left", "left", kind="tap", keys=("LEFT",), seconds=0.1),
    Action("right", "right", kind="tap", keys=("RIGHT",), seconds=0.1),
    Action("jump", "jump / confirm", kind="tap", keys=("SPACE",), seconds=0.1),
    Action("enter", "enter", kind="tap", keys=("ENTER",), seconds=0.1),
]

CLICK_COLS = 6
CLICK_ROWS = 4


def click_actions() -> list[Action]:
    actions: list[Action] = []
    for row in range(CLICK_ROWS):
        for col in range(CLICK_COLS):
            actions.append(
                Action(
                    f"click_{col}_{row}",
                    f"click {col},{row}",
                    kind="click",
                    params={
                        "rx": (col + 0.5) / CLICK_COLS,
                        "ry": (row + 0.5) / CLICK_ROWS,
                    },
                )
            )
    return actions


def catalog_entry(game_id: str) -> dict:
    for item in CATALOG:
        if item.get("id") == game_id:
            return item
    raise KeyError(f"map game not found: {game_id}")


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return "chrome.exe"


class HtmlGameAdapter:
    name = "HTML5"

    def __init__(
        self,
        game_id: str,
        goal: str = "",
        profile: str = "jevgamer-h5",
        mode: str = "keys",
        embedded: bool = False,
        panel_title: str = "JevGamer",
    ) -> None:
        self.entry = catalog_entry(game_id)
        self.goal = goal
        self.mode = mode
        self.embedded = embedded
        self.panel_title = panel_title
        self.page = (MAP_ROOT / str(self.entry["entry_path"])).resolve()
        self.url = self.page.as_uri()
        self.profile = profile
        self.process: subprocess.Popen | None = None
        self.hwnd = 0
        self.candidates: list[dict] = []
        self.pending_click = None
        self.click_acked = False

    def reset(self) -> None:
        self.launch()

    def launch(self) -> None:
        if self.embedded:
            deadline = time.time() + 10.0
            while time.time() < deadline:
                try:
                    self.hwnd = capture.find_window(self.panel_title, "")
                except Exception:
                    self.hwnd = 0
                if self.hwnd:
                    return
                time.sleep(0.3)
            return
        if self.process is not None and self.process.poll() is None:
            return
        chrome = find_chrome()
        profile_dir = ROOT / "runs" / self.profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                chrome,
                f"--user-data-dir={profile_dir}",
                "--new-window",
                "--disable-features=Translate",
                self.url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 20.0
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                self.hwnd = capture.find_window("Chrome", "")
            except Exception:
                self.hwnd = 0
            if self.hwnd:
                break
        if self.hwnd:
            try:
                capture.focus_window(self.hwnd)
            except Exception:
                pass

    def observe(self) -> Observation:
        if self.embedded:
            try:
                self.hwnd = capture.find_window(self.panel_title, "")
            except Exception:
                self.hwnd = 0
        elif not self.hwnd:
            try:
                self.hwnd = capture.find_window("Chrome", "")
            except Exception:
                self.hwnd = 0
        if self.embedded and self.mode == "clicks" and not self.candidates:
            deadline = time.time() + 2.0
            while time.time() < deadline and not self.candidates:
                time.sleep(0.05)
        image = None
        if self.hwnd:
            try:
                image = capture.capture(
                    self.panel_title if self.embedded else "Chrome", ""
                )
            except Exception:
                image = None
        done = self.process is not None and self.process.poll() is not None
        if self.embedded and self.candidates:
            actions = [
                Action(
                    str(candidate.get("id") or f"dom_{index}"),
                    str(candidate.get("label") or candidate.get("id") or "click"),
                    kind="dom_click",
                    params={
                        "selector": candidate.get("selector"),
                        "x": candidate.get("x"),
                        "y": candidate.get("y"),
                    },
                )
                for index, candidate in enumerate(self.candidates)
            ]
        else:
            actions = list(ACTIONS) if self.mode != "clicks" else click_actions()
        return Observation(
            actions=actions,
            image=harness._image_url(image) if image is not None else "",
            state={"game": self.entry.get("id"), "url": self.url, "window": self.hwnd},
            level=1,
            done=done,
            success=False,
            raw={"entry": self.entry},
        )

    def act(self, action: Action) -> None:
        if not self.hwnd:
            return
        try:
            capture.focus_window(self.hwnd)
        except Exception:
            pass
        if action.kind == "dom_click":
            self.pending_click = action
            self.click_acked = False
            deadline = time.time() + 3.0
            while time.time() < deadline and not self.click_acked:
                time.sleep(0.05)
            self.pending_click = None
            return
        if action.kind == "click":
            left, top, right, bottom = capture.window_box(self.hwnd)
            width = max(1, right - left)
            height = max(1, bottom - top)
            if self.embedded:
                # left game pane of the live panel: below header, ~62% width
                gx = left + int((0.01 + 0.60 * float(action.params.get("rx") or 0.5)) * width)
                gy = top + int((0.09 + 0.86 * float(action.params.get("ry") or 0.5)) * height)
            else:
                gx = left + int(float(action.params.get("rx") or 0.5) * width)
                gy = top + int(float(action.params.get("ry") or 0.5) * height)
            capture.set_cursor(gx, gy)
            time.sleep(0.05)
            input_backend.mouse_click()
            return
        for key in action.keys:
            input_backend.tap_key(str(key))

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
