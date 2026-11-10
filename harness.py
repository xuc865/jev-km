"""Fast/slow game loop for a live Windows or macOS game window.

The slow model sees one screenshot and writes a closed set of action
conditions. JEV only picks the option whose condition is satisfied. This
process then sends that choice as a key or mouse event. No process memory
access.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

import capture
import input_backend
from feed import FrameClient
from knowledge import Game, default_id, load

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()
TOKEN = (os.environ.get("JEVGAMER_TOKEN") or os.environ.get("LLM_TOKEN") or "").strip()
CHAT = os.environ.get("JEVGAMER_CHAT", "http://v2.open.venus.oa.com/llmproxy/chat/completions").strip()
# The planner and the scorer draw on the same per-minute quota. When several
# collectors run at once their combined traffic exceeded it and the throttled
# replies silently became unlabelled rows. The local proxy owns that budget in
# one place, so every chat call goes through it when it is listening.
def _chat_url() -> str:
    if os.environ.get("JEVGAMER_CHAT_NO_PROXY"):
        return CHAT
    proxy = os.environ.get("JEVGAMER_CHAT_PROXY", "http://127.0.0.1:8799/v1/chat/completions").strip()
    try:
        import socket
        from urllib.parse import urlparse
        parts = urlparse(proxy)
        with socket.create_connection((parts.hostname, parts.port or 80), timeout=0.5):
            return proxy
    except Exception:
        return CHAT
SYSTEMONE = os.environ.get("JEVGAMER_SYSTEMONE", "http://29.116.237.135:8080/v1/systemone").strip()
# Default WeMM-JEV needs no auth. Hy (Taiji) uses its own key; never fall back
# to the chat token, or it would be sent to whichever SystemOne host is set.
SYSTEMONE_KEY = (os.environ.get("JEVGAMER_SYSTEMONE_KEY") or "").strip()
# Optional latest-frame JPEG endpoint (e.g. Astray's /__agent/image.jpg).
# When set, every JEV call attaches the current game frame so WeMM-JEV can
# see the scene. Comment it out for text-only backends such as Hy.
JEV_IMAGE_URL = os.environ.get("JEVGAMER_JEV_IMAGE_URL", "").strip()
SLOW = os.environ.get("JEVGAMER_SLOW", "deepseek-v4.1-flash").strip()
# One SystemOne model, or a comma-separated list used round-robin per call.
# Default is the local WeMM-JEV (unlimited, text+image). Hy stays available as
# a config backup: point JEVGAMER_SYSTEMONE at Taiji, set a Hy model list and
# JEVGAMER_SYSTEMONE_KEY.
FAST_MODELS = [
    item.strip()
    for item in (os.environ.get("JEVGAMER_FAST") or "WeMM-JEV").split(",")
    if item.strip()
]
FAST = FAST_MODELS[0]
SUPERVISOR = os.environ.get("JEVGAMER_SUPERVISOR", "gpt-6.1-terra").strip()
# JEV acts only when its top option is the compiled action and clears the
# confidence gate; otherwise the planner's explicit choice stands.
USE_JEV = os.environ.get("JEVGAMER_JEV", "1").strip().lower() not in {"0", "false", "no", "off"}

# JevAgentHarness uses the top-option probability. Its published 0.99 and 1.00
# fits were for irreversible tool calls among near-duplicate tools. A game step
# can be corrected on the next frame, and a vague contract here previously
# scored about 0.44 while a unique one scored 1.0. 0.6 keeps the clear picks.
# SystemOne's own confidence uses a different formula, so recalibrate per model
# via JEVGAMER_JEV_CONFIDENCE instead of reusing another vendor's threshold.
try:
    JEV_CONFIDENCE = float(os.environ.get("JEVGAMER_JEV_CONFIDENCE", "0.6"))
except ValueError:
    JEV_CONFIDENCE = 0.6
# Round-robin cursor + per-model 429 cooldown for the SystemOne fast models.
_JEV_ROTATION = 0
_JEV_BACKOFF: dict[str, float] = {}
# When SystemOne is not answering within the per-attempt timeout, every single
# step paid that timeout before falling back: measured at 8.6-9.1s per step
# across a whole episode, capping collection at roughly 30 usable steps in five
# minutes even though the scoring itself takes 0.3s. The cause was queueing
# under load, not a dead host, so the breaker no longer diverts to another
# channel — it widens the timeout so the queued call gets its turn.
_JEV_SLOW_UNTIL = 0.0
_JEV_SLOW_SPAN = float(os.environ.get("JEVGAMER_JEV_SLOW_SPAN") or 120)
_JEV_SLOW_TIMEOUT = float(os.environ.get("JEVGAMER_JEV_SLOW_TIMEOUT") or 45)
SCAN = input_backend.SCAN


def _next_jev_model() -> str:
    """Next model in the rotation that is not in a 429 cooldown."""
    global _JEV_ROTATION
    if not FAST_MODELS:
        return ""
    now = time.time()
    for offset in range(len(FAST_MODELS)):
        index = (_JEV_ROTATION + offset) % len(FAST_MODELS)
        candidate = FAST_MODELS[index]
        if now >= _JEV_BACKOFF.get(candidate, 0.0):
            _JEV_ROTATION = (index + 1) % len(FAST_MODELS)
            return candidate
    return ""


def _fetch_jev_image() -> list[str] | None:
    """Latest game frame as a base64 data URL for WeMM-JEV vision input."""
    if not JEV_IMAGE_URL:
        return None
    try:
        with urllib.request.urlopen(JEV_IMAGE_URL, timeout=1.5) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    if not raw:
        return None
    return ["data:image/jpeg;base64," + base64.b64encode(raw).decode()]


def chat_generation(*, thinking: bool | None = None, max_output_tokens: int | None = None) -> dict:
    """Resolve public generation settings without credentials or model messages."""
    settings = {}
    cap = max_output_tokens if max_output_tokens is not None else os.environ.get("JEVGAMER_MAX_OUTPUT_TOKENS")
    if cap is not None and cap != "":
        cap = int(cap)
        if cap < 1: raise ValueError("max_output_tokens must be positive")
        settings["max_tokens"] = cap
    if thinking is None:
        configured = os.environ.get("JEVGAMER_CHAT_THINKING")
        thinking = configured == "1" if configured in {"0","1"} else None
    if thinking is not None:
        if type(thinking) is not bool: raise TypeError("thinking must be a boolean")
        settings["chat_template_kwargs"] = {"thinking": thinking}
    return settings


def chat(messages: list[dict], token: str, *, model: str | None = None, temperature: float = 0,
         thinking: bool | None = None, max_output_tokens: int | None = None) -> dict:
    payload = {"model": model or SLOW, "temperature": temperature, "messages": messages}
    payload.update(chat_generation(thinking=thinking,max_output_tokens=max_output_tokens))
    url = _chat_url()
    try:
        return post(url, payload, token)
    except Exception as exc:
        # Some deployments only accept their own default temperature. Retry once
        # without the field instead of losing the whole postmortem.
        if "temperature" not in str(exc): raise
        payload.pop("temperature", None)
        return post(url, payload, token)


def locked_messages(game: Game, task: str, user_content, thread=None) -> list[dict]:
    """System-lock self+goal + Codex/OpenGCUA thread history + current user turn.

    History is mandatory. Passing ``thread=None`` still builds a one-shot call,
    but live control must pass ``world.thread``.
    """
    system = game.system_lock(task)
    if thread is not None:
        return thread.messages_for_prompt(system, user_content)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def post(url: str, payload: dict, token: str, *, timeout: float = 90, attempts: int = 3) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    last: dict = {}
    failures: list[dict] = []
    total_attempts = max(1, int(attempts))
    for attempt in range(total_attempts):
        retryable = True
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode())
                if not isinstance(body, dict):
                    raise ValueError("Model service returned non-object JSON")
                body["_transport_attempts"] = attempt + 1
                if failures: body["_transport_failures"] = failures
                return body
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", "replace")
            last = {"_status": exc.code, "_error": message[:400]}
            # Retry the identical authorized request; never change credentials,
            # endpoint or headers. Authentication failures are definitive, including HTML.
            retryable = exc.code in {408,425,429,500,502,503,504} or (
                exc.code not in {401,403} and message.lstrip().lower().startswith(("<!doctype html","<html")))
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError,
                ValueError, UnicodeError) as exc:
            last = {"_error": str(exc)[:400]}
        failures.append({"status": last.get("_status"), "error": last.get("_error","")[:160]})
        if not retryable or attempt == total_attempts - 1: break
        time.sleep(1.5 * (attempt + 1))
    last["_transport_attempts"] = attempt + 1
    last["_transport_failures"] = failures
    return last


def find_window(title_part: str) -> int:
    try:
        return capture.find_window(title_part, "")
    except RuntimeError:
        return 0


def focus(hwnd: int) -> None:
    capture.focus_window(hwnd)


def tap(name: str) -> None:
    input_backend.tap_key(str(name).upper())


def hold_keys(names: tuple[str, ...], seconds: float) -> None:
    for name in names:
        input_backend.key_down(name)
    time.sleep(seconds)
    for name in names:
        input_backend.key_up(name)


_held: list[str] = []
_last_look = ""
_bridge_woke_hwnd = 0


def wake_bridge_canvas(game: Game) -> None:
    """Click the game client once so WebGL/rAF leave a frozen background state."""
    global _bridge_woke_hwnd
    hwnd = capture.find_window(game.window, game.window_class)
    if not hwnd:
        return
    # Re-arm when the window handle changes (new Edge tab / reload).
    if _bridge_woke_hwnd == hwnd:
        return
    click_at(0.5, 0.52, game.window, game.window_class, keep_cursor=True)
    _bridge_woke_hwnd = hwnd
    time.sleep(0.08)


def reset_intent() -> None:
    global _last_look
    _last_look = ""


def reset_control() -> None:
    """Drop held keys and look memory so the next episode starts clean."""
    release_all()
    reset_intent()


def last_look() -> str:
    return _last_look


def release_all() -> None:
    global _held
    for name in _held:
        input_backend.key_up(name)
    _held = []


def set_move(names: tuple[str, ...]) -> None:
    global _held
    wanted = list(names)
    if wanted == _held:
        return
    release_all()
    for name in wanted:
        input_backend.key_down(name)
    _held = wanted


def pulse_held() -> None:
    """Re-fire held keydowns (OS auto-repeat). Needed by canvas games that clear axis each frame."""
    for name in _held:
        input_backend.key_down(name)


# HTML bridge axis: left/right -> dx, down/up -> dy (KeyboardJS bind.axis order).
_BRIDGE_AXIS = {
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "DOWN": (0, -1),
    "UP": (0, 1),
}


def bridge_hold(url: str, keys: tuple[str, ...], seconds: float) -> bool:
    """POST a timed axis hold to the local game bridge. Returns False on failure."""
    dx = dy = 0
    dir_name = ""
    for name in keys:
        key = str(name).upper()
        pair = _BRIDGE_AXIS.get(key)
        if not pair:
            continue
        dx += pair[0]
        dy += pair[1]
        if key in {"LEFT", "RIGHT", "UP", "DOWN"} and not dir_name:
            dir_name = key.lower()
    dx = max(-1, min(1, dx))
    dy = max(-1, min(1, dy))
    ms = max(50, int(float(seconds) * 1000))
    payload = json.dumps({"dx": dx, "dy": dy, "dir": dir_name, "ms": ms}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw or "{}")
        return bool(data.get("ok", True))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def bridge_release(url: str) -> None:
    payload = json.dumps({"release": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=1.0).read()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


def bridge_telemetry(url: str) -> dict:
    """GET live game telemetry from the local bridge (player_x / curve / laps)."""
    base = str(url or "").strip()
    if not base:
        return {}
    telemetry_url = base.replace("/__agent/axis", "/__agent/telemetry")
    if telemetry_url == base:
        telemetry_url = base.rstrip("/") + "/../telemetry"
    req = urllib.request.Request(telemetry_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return {}


def look(dx: int, dy: int, title: str, window_class: str, *, keep_cursor: bool = False) -> None:
    hwnd = capture.find_window(title, window_class)
    saved = capture.cursor_pos() if keep_cursor else None
    if not keep_cursor:
        left, top, right, bottom = capture.client_box(hwnd)
        capture.set_cursor((left + right) // 2, (top + bottom) // 2)
    input_backend.mouse_move(int(dx), int(dy))
    if saved is not None:
        capture.set_cursor(*saved)


def click(right: bool = False) -> None:
    input_backend.mouse_click(right=bool(right))


def click_at(x_ratio: float, y_ratio: float, title: str, window_class: str, *, keep_cursor: bool = False) -> None:
    """Click a normalized point in the game client (used for modal buttons)."""
    hwnd = capture.find_window(title, window_class)
    left, top, right, bottom = capture.client_box(hwnd)
    x = int(left + max(0.0, min(1.0, x_ratio)) * (right - left))
    y = int(top + max(0.0, min(1.0, y_ratio)) * (bottom - top))
    saved = capture.cursor_pos() if keep_cursor else None
    capture.set_cursor(x, y)
    click()
    if saved is not None:
        capture.set_cursor(*saved)


def _with_game_input(hwnd: int, quiet: bool, body):
    """Send keys briefly without keeping the user's mouse or desktop focus."""
    previous = input_backend.foreground_window()
    saved = capture.cursor_pos()
    if previous != hwnd:
        capture.focus_window(hwnd)
    try:
        return body()
    finally:
        if quiet:
            capture.set_cursor(*saved)
            if previous and previous != hwnd and input_backend.is_window(previous):
                input_backend.set_foreground(previous)


def sleep_while(seconds: float, should_stop=None) -> None:
    end = time.time() + max(0.0, seconds)
    while time.time() < end:
        if should_stop and should_stop():
            return
        time.sleep(min(0.05, end - time.time()))


def chosen_seconds(choice: str, game: Game, contract: dict) -> float:
    fallback = game.default_seconds(choice)
    raw = (contract.get("seconds") or {}).get(choice, fallback)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = fallback
    spec = game.action(choice)
    kind = str(((spec or {}).get("input") or {}).get("type") or "release")
    if choice == "done":
        return 0.0
    if choice == "wait" or kind == "release":
        return min(max(value, 0.1), 0.4)
    if kind in {"tap", "click", "prompt"}:
        return min(max(value, 0.15), 0.6)
    return min(max(value, 0.5), 4.0)


def parse_choices(raw, game: Game) -> tuple[list[str], str]:
    options = game.option_ids()
    prompt_action = game.prompt_action_id()
    prompt_key = ""
    if isinstance(raw, str):
        items = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
    else:
        items = []
    picked: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item
        if key not in options and prompt_action and game.prompt_of(key):
            prompt_key = key
            key = prompt_action
        if key not in options:
            continue
        track = game.track_of(key)
        if track in seen:
            continue
        seen.add(track)
        picked.append(key)
    return picked, prompt_key


def tap_prompt(name: str, game: Game) -> str:
    item = game.prompt_of(name)
    key = str((item or {}).get("id") or name).strip().upper()
    if key in {"RMB", "MOUSE_RIGHT"}:
        click(True)
        return "鼠标右键"
    if key in SCAN:
        tap(key)
        return key
    return "不按键"


def _run_once(choice: str, game: Game, prompt_key: str) -> str:
    spec = game.action(choice)
    payload = (spec or {}).get("input") or {}
    kind = str(payload.get("type") or "release")
    if choice == "detach":
        hwnd = capture.find_window(game.window, game.window_class)
        focus(hwnd)
        for name in ("X", "SPACE"):
            input_backend.key_down(name)
            time.sleep(0.22)
            input_backend.key_up(name)
            time.sleep(0.18)
            focus(hwnd)
        return str((spec or {}).get("key") or choice)
    if kind == "prompt":
        pressed = tap_prompt(prompt_key, game)
        return pressed if pressed != "不按键" else str((spec or {}).get("key") or choice)
    if kind == "tap":
        for name in payload.get("keys") or []:
            tap(str(name))
        return str((spec or {}).get("key") or choice)
    if kind == "tap_long":
        duration = max(0.12, min(0.8, float(payload.get("seconds") or 0.25)))
        for name in payload.get("keys") or []:
            input_backend.key_down(str(name))
            time.sleep(duration)
            input_backend.key_up(str(name))
        return str((spec or {}).get("key") or choice)
    if kind == "repeat_tap":
        count = max(1, min(8, int(payload.get("count") or 3)))
        gap = max(0.04, min(0.3, float(payload.get("gap") or 0.12)))
        for _ in range(count):
            for name in payload.get("keys") or []:
                tap(str(name))
            time.sleep(gap)
        return str((spec or {}).get("key") or choice)
    if kind == "click":
        click()
        return str((spec or {}).get("key") or choice)
    if kind == "click_at":
        click_at(
            float(payload.get("x") or 0.5),
            float(payload.get("y") or 0.5),
            game.window,
            game.window_class,
            keep_cursor=game.quiet_input(),
        )
        return str((spec or {}).get("key") or choice)
    return str((spec or {}).get("key") or choice)


def act_bundle(choices: list[str], game: Game, contract: dict, prompt_key: str = "", should_stop=None) -> tuple[str, float]:
    """Run move, look and a tap together. No hold this step means release walking."""
    global _last_look
    try:
        hwnd = capture.find_window(game.window, game.window_class)
    except Exception:
        # Feed games can live in an iframe of the console; the bridge is enough.
        hwnd = 0
    quiet = game.quiet_input()
    bridge = game.bridge_url()
    if not choices or "done" in choices:
        release_all()
        if bridge:
            bridge_release(bridge)
        return "停止" if "done" in choices else "不按键", 0.0

    def run(*, use_bridge: bool = False) -> tuple[str, float]:
        global _last_look
        labels: list[str] = []
        move = next((name for name in choices if game.track_of(name) == "move"), "")
        look_choice = next((name for name in choices if game.track_of(name) == "look"), "")
        once = next((name for name in choices if game.track_of(name) == "once"), "")
        move_spec = game.action(move) if move else None
        move_payload = (move_spec or {}).get("input") or {}
        move_kind = str(move_payload.get("type") or "")
        move_keys = tuple(str(k) for k in (move_payload.get("keys") or ()))
        look_seconds = chosen_seconds(look_choice, game, contract) if look_choice else 0.0
        move_seconds = chosen_seconds(move, game, contract) if move_kind == "hold" else 0.0
        once_seconds = chosen_seconds(once, game, contract) if once else 0.0
        total = max(look_seconds, move_seconds, once_seconds, 0.0)
        bridged = False
        if move_kind == "hold" and use_bridge and bridge and move_keys:
            bridged = bridge_hold(bridge, move_keys, move_seconds or 0.45)
            if bridged:
                labels.append(str(move_spec.get("key") or move))
            else:
                use_bridge = False
        if move_kind == "hold" and not bridged:
            if hwnd:
                set_move(move_keys)
                labels.append(str(move_spec.get("key") or move))
        elif move_kind != "hold":
            release_all()
            if bridge:
                bridge_release(bridge)
            if move:
                labels.append(str((move_spec or {}).get("key") or move))
        if once and hwnd:
            labels.append(_run_once(once, game, prompt_key))
        start = time.time()
        end = start + total
        look_payload = (game.action(look_choice) or {}).get("input") or {} if look_choice else {}
        look_dx = int(look_payload.get("dx") or 0)
        look_dy = int(look_payload.get("dy") or 0)
        next_look = start
        # Some canvas games zero the axis every frame; bridge re-applies in-page,
        # otherwise OS key-repeat / pulse keeps KeyboardJS axis alive.
        while time.time() < end:
            if should_stop and should_stop():
                break
            if look_choice and hwnd and time.time() >= next_look:
                look(look_dx, look_dy, game.window, game.window_class, keep_cursor=quiet)
                next_look = time.time() + 0.12
            if move_kind == "hold" and not bridged:
                pulse_held()
            time.sleep(0.05)
        if bridged and bridge:
            bridge_release(bridge)
        if look_choice and hwnd:
            labels.append(str((game.action(look_choice) or {}).get("key") or look_choice))
            _last_look = look_choice
        return (" + ".join(labels) if labels else "不按键"), total

    if bridge:
        # Axis is set via HTTP, but Chromium throttles rAF in background tabs
        # so the ball never integrates — briefly focus during the hold, then
        # restore the user's previous window (still no mouse steal). An
        # embedded iframe has no OS window: just use the bridge.
        def run_bridge() -> tuple[str, float]:
            try:
                if hwnd:
                    try:
                        wake_bridge_canvas(game)
                    except Exception:
                        pass
                return run(use_bridge=True)
            finally:
                release_all()

        if hwnd:
            return _with_game_input(hwnd, True, run_bridge)
        return run_bridge()
    if quiet:
        # Cardinal/quiet games must not leave held keys after focus returns,
        # or arrows keep typing into the user's active window.
        def run_quiet() -> tuple[str, float]:
            try:
                return run(use_bridge=False)
            finally:
                release_all()

        if hwnd:
            return _with_game_input(hwnd, True, run_quiet)
        return run_quiet()
    if hwnd:
        focus(hwnd)
    return run(use_bridge=False)


def act(choice: str, game: Game, seconds: float | None = None, prompt_key: str = "", should_stop=None) -> str:
    contract = {"seconds": {choice: seconds} if seconds is not None else {}}
    pressed, _total = act_bundle([choice], game, contract, prompt_key, should_stop)
    return pressed


def act_feed(choice: str, game: Game, seconds: float, should_stop=None, frame: dict | None = None) -> tuple[str, float]:
    """Execute one feed move as short pulses until the ball changes cell.

    A fixed 0.15s hold can cross several cells; that makes a discovered-cell
    graph and the shortest-path plan wrong. Pulses stop on the first observed
    cell change, so one decision is one edge in the discovered maze.
    """
    spec = game.action(choice) or {}
    payload = (spec.get("input") or {})
    kind = str(payload.get("type") or "release")
    bridge = game.bridge_url()
    if kind == "hold" and bridge:
        keys = tuple(str(k) for k in payload.get("keys") or ())
        state = frame.get("state") if isinstance(frame, dict) else {}
        start_cell = state.get("cell") if isinstance(state, dict) else None
        feed_url = game.feed_url()
        started = time.time()
        if feed_url and isinstance(start_cell, (list, tuple)) and len(start_cell) == 2:
            start = (int(start_cell[0]), int(start_cell[1]))
            client = FrameClient(feed_url, timeout=0.25)
            deadline = started + max(0.35, float(seconds) * 3.0)
            while time.time() < deadline:
                if should_stop and should_stop():
                    break
                held = bridge_hold(bridge, keys, 0.08)
                if not held:
                    break
                time.sleep(0.08)
                bridge_release(bridge)
                now = client.fetch() or {}
                now_state = now.get("state") if isinstance(now, dict) else {}
                cell = now_state.get("cell") if isinstance(now_state, dict) else None
                if isinstance(cell, (list, tuple)) and len(cell) == 2:
                    if (int(cell[0]), int(cell[1])) != start:
                        break
            bridge_release(bridge)
            settle_deadline = time.time() + 1.0
            while time.time() < settle_deadline:
                now = client.fetch() or {}
                now_state = now.get("state") if isinstance(now, dict) else {}
                if isinstance(now_state, dict) and now_state.get("settled"):
                    break
                time.sleep(0.05)
            return str(spec.get("key") or choice), max(0.05, time.time() - started)
        held = bridge_hold(bridge, keys, seconds)
        if held:
            time.sleep(max(0.05, float(seconds)))
            bridge_release(bridge)
            return str(spec.get("key") or choice), float(seconds)
        bridge_release(bridge)
        return "不按键", 0.0
    if kind == "release" or choice == "wait":
        if bridge:
            bridge_release(bridge)
        return "不按键", 0.0
    return act_bundle([choice], game, {"seconds": {choice: seconds}}, "", should_stop)


def recent_context(history: list[str]) -> str:
    if not history:
        return ""
    recent = history[-12:]
    lines = ["这次任务里已经发生的事，从早到晚。相反方向的转镜头会抵消刚才的转向。这不是下一步的命令："]
    lines.extend(f"- {item}" for item in recent)
    return "\n".join(lines)


def _image_url(image: Image.Image, max_edge: int = 2048, quality: int = 92) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    if longest > max_edge > 0:
        scale = max_edge / longest
        rgb = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _answer_token(text: str) -> str:
    if not text or not text.strip():
        return ""
    token = re.split(r"\s+", text.strip().lower())[0].strip(".,!。！，")
    if token.startswith("yes") or token in {"y", "是", "有"}:
        return "yes"
    if token.startswith("no") or token in {"n", "否", "没有", "无"}:
        return "no"
    return ""


def flat_cover(image: Image.Image, s_max: int, v_min: int) -> float:
    """Largest connected flat bright region, as a fraction of the frame."""
    small = image.convert("RGB").resize((64, 36)).convert("HSV")
    width, height = small.size
    pix = small.load()
    mask = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            _hue, sat, val = pix[x, y]
            mask[y][x] = sat < s_max and val > v_min
    seen = [[False] * width for _ in range(height)]
    best = 0
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or seen[y][x]:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if area > best:
                best = area
    return best / (width * height)


def check_result(image: Image.Image, item: dict, token: str) -> str:
    """Run one pack check. A measure reads the pixels. An ask calls the model."""
    if item.get("measure") == "flat_cover":
        value = flat_cover(image, int(item.get("s_max") or 50), int(item.get("v_min") or 160))
        passed = value >= float(item.get("min") or 1)
        return f"{'yes' if passed else 'no'} {value:.3f}"
    if item.get("ask"):
        return screen_check(image, str(item["ask"]), token)
    return ""


def screen_check(image: Image.Image, question: str, token: str) -> str:
    """Ask one image-only question. The action prompt is not included."""
    body = post(_chat_url(), {
        "model": SLOW,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": _image_url(image)}},
            ],
        }],
    }, token)
    if body.get("_error"):
        return ""
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return _answer_token(text)


def slow_contract(image: Image.Image, task: str, token: str, game: Game, history: list[str] | None = None, loaded: list[str] | None = None, note: str = "") -> dict:
    context = recent_context(history or [])
    prompt = (
        "你是游戏操作的慢模型。系统已锁定你是谁、目标是什么；禁止改写。\n"
        "当前执行任务：" + (game.locked_goal(task) or task.strip()) + "\n"
        + ((context + "\n") if context else "")
        + "先看这一张图，再对照已经发生的操作和已载入的知识，选定下一步。choice 写成数组，移动和转镜头可以同时选。\n"
        "动作只作用在锁定的 self 上。不要把刚转过的方向立刻反转。同时为每个操作写条件，供快模型按轨道判断。它判断不了或置信度不够时，执行你的 choice。\n"
        "如果目录里某条知识和当前画面有关、但全文还没载入，把 load 设成它的 id。\n\n"
        + game.manual(loaded or [])
        + (("\n" + note) if note else "")
    )
    body = chat(
        locked_messages(game, task, [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_url(image)}},
        ]),
        token,
        temperature=0.2,
    )
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not text:
        text = ((body.get("choices") or [{}])[0].get("message") or {}).get("reasoning") or ""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    options = game.option_ids()
    criteria = {}
    for name in options:
        value = parsed.get(name)
        criteria[name] = value if isinstance(value, str) and value else "Not this step."
    raw_load = parsed.get("load")
    requested = raw_load.strip() if isinstance(raw_load, str) else ""
    raw_prompt = parsed.get("prompt")
    prompt_key = raw_prompt.strip() if isinstance(raw_prompt, str) else ""
    choices, mapped_prompt = parse_choices(parsed.get("choice"), game)
    if mapped_prompt:
        prompt_key = mapped_prompt
    prompt_action = game.prompt_action_id()
    if prompt_action in choices:
        current = str(criteria.get(prompt_action) or "")
        if not current.lower().startswith("correct only"):
            criteria[prompt_action] = "Correct only when the on-screen prompt matches this key."
    choice = choices[0] if len(choices) == 1 else choices
    raw_seconds = parsed.get("seconds")
    seconds = {}
    if isinstance(raw_seconds, dict):
        for name in options:
            value = raw_seconds.get(name)
            try:
                seconds[name] = float(value)
            except (TypeError, ValueError):
                seconds[name] = game.default_seconds(name)
    else:
        seconds = {name: game.default_seconds(name) for name in options}
    return {
        "criteria": criteria, "choice": choice, "choices": choices, "load": requested, "prompt": prompt_key,
        "seconds": seconds, "raw": text[:800], "error": body.get("_error"), "checks": {}, "rejected": [],
    }


def enforce_checks(image: Image.Image, game: Game, contract: dict, token: str, answers: dict | None = None) -> dict:
    """A pack check is an image-only question. A failed check cannot be selected."""
    answers = dict(answers or {})
    rejected: list[str] = []
    if not contract.get("error"):
        for name in game.option_ids():
            item = game.check_of(name)
            if not item:
                continue
            marked = "correct only" in str(contract["criteria"].get(name, "")).lower()
            selected = name in (contract.get("choices") or []) or contract.get("choice") == name
            if not marked and not selected:
                continue
            check_id = str(item["id"])
            if check_id not in answers:
                answers[check_id] = check_result(image, item, token)
            verdict = str(answers[check_id]).split(" ", 1)[0]
            expect = str(item.get("expect") or "yes").lower()
            if verdict != expect:
                rejected.append(name)
        for name in rejected:
            contract["criteria"][name] = "Not this step."
        kept = [name for name in (contract.get("choices") or []) if name not in rejected]
        if contract.get("choice") in rejected:
            contract["choice"] = kept[0] if len(kept) == 1 else kept
        contract["choices"] = kept
    contract["checks"] = answers
    contract["rejected"] = rejected
    return contract


def confirm_choice(choice: str, game: Game, token: str) -> dict:
    """Recapture the live window and refuse a checked action that no longer matches."""
    item = game.check_of(choice)
    if item is None:
        return {"ok": True, "image": None, "check": ""}
    image = capture.capture(game.window, game.window_class)
    result = check_result(image, item, token)
    verdict = str(result).split(" ", 1)[0]
    expect = str(item.get("expect") or "yes").lower()
    return {
        "ok": verdict == expect,
        "image": image,
        "check": f"{item['id']}={result or 'unclear'}",
    }


def confirm_bundle(choices: list[str], game: Game, token: str) -> tuple[list[str], str, object]:
    kept: list[str] = []
    notes: list[str] = []
    image = None
    for name in choices:
        if not game.check_of(name):
            kept.append(name)
            continue
        confirmed = confirm_choice(name, game, token)
        if confirmed.get("check"):
            notes.append(confirmed["check"])
        if confirmed.get("image") is not None:
            image = confirmed["image"]
        if confirmed["ok"]:
            kept.append(name)
    return kept, " ".join(notes), image


def gate_choice(image: Image.Image, task: str, token: str, game: Game, history: list[str] | None, loaded: list[str] | None, contract: dict) -> dict:
    """Reject a visually unchecked action, then let the slow model pick again."""
    contract = enforce_checks(image, game, contract, token)
    if contract.get("error") or not contract.get("rejected"):
        return contract
    blocked = ", ".join(contract["rejected"])
    seen = ", ".join(f"{key}={value or 'unclear'}" for key, value in contract["checks"].items())
    note = (
        f"看图核对结果：{seen}。\n"
        f"这些操作和核对结果不符，这一步不能选，必须写成 Not this step.：{blocked}。\n"
        "请在其余操作里重新选定 choice。"
    )
    second = slow_contract(image, task, token, game, history, loaded, note)
    if second.get("error"):
        return contract
    return enforce_checks(image, game, second, token, contract.get("checks"))


def _jev_timeout() -> float:
    """Per-attempt timeout for a fast-model call.

    WeMM-JEV answers in about 0.3s when idle and keeps that up under light load,
    but measured latency grows with concurrency (4.2s for a burst of 48), so the
    deadline has to clear the queueing delay, not just the compute. The earlier
    8s default treated a busy service as a broken one: steps timed out, the
    episode fell back to a rate-limited channel, and most collected rows ended
    up without a label.
    """
    try:
        value = float(os.environ.get("JEVGAMER_JEV_TIMEOUT") or 30)
    except ValueError:
        value = 30.0
    return max(3.0, value)


def _jev_attempts() -> int:
    """How many times to retry one fast-model call before giving up."""
    try:
        value = int(os.environ.get("JEVGAMER_JEV_ATTEMPTS") or 1)
    except ValueError:
        value = 2
    return max(1, min(5, value))



def _fallback_scores(criteria: dict, grouped: dict, memory: str = "",
                     images: list | None = None) -> dict:
    """Score the action group with the chat channel when SystemOne is dead.

    Only the `action` track is needed to pick a step: the other tracks exist for
    the multi-track harness. If anything about the fallback fails the caller
    treats it as an ordinary scoring error, so this stays a pure improvement over
    returning an empty table.

    Off by default. Measurement showed WeMM-JEV answering in 0.3s and finishing
    48 concurrent calls in 4.2s, so what looked like a dead endpoint was really
    queueing past the per-attempt timeout. Routing to the chat channel instead
    pushed every step onto a shared key with a hard per-minute quota, where the
    throttled replies turned steps into unlabelled rows — strictly worse than
    waiting a second longer. Enable it only via JEVGAMER_JEV_FALLBACK=1.
    """
    if os.environ.get("JEVGAMER_JEV_FALLBACK", "").strip() not in {"1", "true", "yes"}:
        return {"probabilities": {}, "tracks": {}, "error": "fallback disabled"}
    try:
        from general_agent import chat_scorer
    except Exception:
        return {"probabilities": {}, "tracks": {}, "error": "fallback unavailable"}
    names = tuple(grouped.get("action") or ())
    subset = {name: criteria[name] for name in names if name in criteria}
    if not subset:
        subset = dict(criteria)
    try:
        return chat_scorer.score(subset, memory=memory, images=images)
    except Exception as exc:
        return {"probabilities": {}, "tracks": {}, "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}


def _chat_share() -> float:
    """Fraction of steps whose label may come from the chat channel.

    WeMM-JEV sustains about 45 calls/min and stays flat from three to sixteen
    callers, so it is a serial queue: more workers only stretch each step and
    push calls toward the timeout. The chat channel measured 131 calls/min at
    0.53s per call, and on 40 already-labelled steps its top-1 pick matched JEV
    82.5% of the time (92.5% within top-3). Splitting the load raises total
    throughput while keeping what a row means unchanged: both channels answer
    the same question over the same criteria.
    """
    raw = os.environ.get("JEVGAMER_CHAT_SHARE", "0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, value))


def _chat_first(has_images: bool) -> bool:
    """True when this particular call should be answered by the chat channel."""
    share = _chat_share()
    if share <= 0.0:
        return False
    if has_images and os.environ.get("JEVGAMER_CHAT_NEEDS_IMAGE", "").strip() in {"1", "true", "yes"}:
        # The chat model here is text-only; a vision-required step must keep its
        # frame or the label would describe a state the scorer never saw.
        return False
    return random.random() < share


def jev_scores(criteria: dict, token: str, tracks: dict[str, tuple[str, ...]] | None = None, memory: str = "", images: list[str] | None = None) -> dict:

    if not USE_JEV:
        return {"probabilities": {}, "tracks": {}, "error": "jev disabled; use slow-model contract"}
    model = _next_jev_model()
    if not model:
        return {"probabilities": {}, "tracks": {}, "error": "jev cooling down after rate limit"}
    if images is None and JEV_IMAGE_URL:
        images = _fetch_jev_image()
    grouped = tracks or {"action": tuple(criteria.keys())}
    if _chat_first(bool(images)):
        try:
            from general_agent import chat_scorer as _chat_scorer

            picked = {name: criteria[name] for name in grouped.get("action", ()) if name in criteria}
            chat = _chat_scorer.score(picked or dict(criteria), memory=memory)
        except Exception as exc:
            chat = {"probabilities": {}, "tracks": {},
                    "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
        if chat.get("probabilities"):
            chat["channel"] = "chat"
            return chat


    # A call that timed out once is most likely queued, not rejected.
    under_load = time.time() < _JEV_SLOW_UNTIL
    questions = {}
    for track, names in grouped.items():
        subset = {name: criteria[name] for name in names if name in criteria}
        if not subset:
            continue
        questions[track] = {
            "type": "choice",
            "instructions": "Pick the single option whose usage condition is satisfied. Using `world`.",
            "criteria": subset,
        }
    world = ("Use the supplied images and current public state to pick the option whose condition is satisfied."
             if images else "Use the current public state and slow-planner context. No image is attached. Pick the option whose condition is satisfied.")
    if memory:
        world += " " + memory
    payload = {
        "model": model,
        "state": {"world": world},
        "questions": questions,
    }
    if images:
        # WeMM-JEV accepts base64 data URLs here; Hy ignores/rejects the field.
        payload["images"] = list(images)
    body = post(SYSTEMONE, payload, SYSTEMONE_KEY,
                timeout=_JEV_SLOW_TIMEOUT if under_load else _jev_timeout(),
                attempts=_jev_attempts())
    if body.get("_status") == 429:
        # Per-key QPM is low on some SystemOne models; rotate past this one
        # instead of re-hitting the limit on every game step.
        _JEV_BACKOFF[model] = time.time() + 60
    if body.get("_error"):
        error = str(body["_error"])[:400]
        lowered = error.lower()
        timed_out = "timed out" in lowered or "connection" in lowered or "reset" in lowered
        if timed_out:
            # Treat it as load, not death: widen the window for the following
            # calls so a queued request gets answered instead of being cut off
            # at the same deadline again.
            globals()["_JEV_SLOW_UNTIL"] = time.time() + _JEV_SLOW_SPAN
            fallback = _fallback_scores(criteria, grouped, memory, images)
            if (fallback.get("probabilities") or {}):
                fallback["primary_error"] = error
                return fallback
        return {"probabilities": {}, "tracks": {}, "error": error, "model": model}

    answers = body.get("answers") or {}
    clean = {}
    per_track = {}
    for track, payload in answers.items():
        probabilities = (payload or {}).get("probabilities") or {}
        track_clean = {}
        for name, value in probabilities.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            track_clean[str(name)] = number
            clean[str(name)] = number
        per_track[str(track)] = track_clean
    return {
        "probabilities": clean,
        "tracks": per_track,
        "error": None,
        "model": model,
        "usage": body.get("usage") or {},
        "image_count": len(images or []),
    }


def share_control(criteria: dict, explicit_choice: str, scores: dict, options: tuple[str, ...]) -> dict:
    """JevAgentHarness cascade: Jev acts only when it can judge and is confident."""
    probabilities = {
        name: value for name, value in (scores.get("probabilities") or {}).items() if name in options
    }
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_name = ranked[0][0] if ranked else ""
    top_p = ranked[0][1] if ranked else 0.0
    leaders = [name for name, value in ranked if abs(value - top_p) <= 1e-6]
    marked = [
        name for name in options
        if isinstance(criteria.get(name), str) and criteria[name].lower().startswith("correct only")
    ]
    fallback = explicit_choice if explicit_choice in options else (marked[0] if len(marked) == 1 else "")
    can_judge = not scores.get("error") and len(leaders) == 1 and top_name in options
    if can_judge and top_p >= JEV_CONFIDENCE:
        return {"choice": top_name, "by": "jev", "confidence": top_p, "jev_choice": top_name}
    return {
        "choice": fallback,
        "by": "slow",
        "confidence": top_p,
        "jev_choice": top_name if can_judge else "",
    }


def decide_bundle(game: Game, contract: dict, scores: dict) -> dict:
    """Pick at most one action per track. JEV cannot reverse the last look."""
    explicit, _prompt = parse_choices(contract.get("choices") or contract.get("choice"), game)
    picked: list[str] = []
    by_track: list[str] = []
    jev_names: list[str] = []
    confidences: list[float] = []
    for track, names in game.track_ids().items():
        if track == "stop":
            explicit_stop = next((name for name in explicit if name == "done"), "")
            if explicit_stop:
                return {"choices": ["done"], "by": "slow", "confidence": 0.0, "jev_choice": ""}
            continue
        explicit_one = next((name for name in explicit if game.track_of(name) == track), "")
        track_scores = {
            "probabilities": (scores.get("tracks") or {}).get(track) or {
                name: value for name, value in (scores.get("probabilities") or {}).items() if name in names
            },
            "error": scores.get("error"),
        }
        decision = share_control(contract["criteria"], explicit_one, track_scores, names)
        choice = decision["choice"]
        if track == "look" and choice and last_look() and game.opposes_look(choice, last_look()) and decision["by"] == "jev":
            choice = explicit_one if explicit_one and not game.opposes_look(explicit_one, last_look()) else ""
            decision["by"] = "slow"
        if choice and choice in names and choice != "wait":
            picked.append(choice)
            by_track.append(decision["by"])
            if decision.get("jev_choice"):
                jev_names.append(decision["jev_choice"])
            confidences.append(float(decision.get("confidence") or 0))
        elif choice == "wait":
            picked.append("wait")
            by_track.append(decision["by"])
            confidences.append(float(decision.get("confidence") or 0))
    by = "jev" if by_track and all(item == "jev" for item in by_track) else ("slow" if by_track and all(item == "slow" for item in by_track) else "mixed")
    return {
        "choices": picked,
        "by": by,
        "confidence": max(confidences) if confidences else 0.0,
        "jev_choice": "+".join(jev_names),
    }


def play(game_id: str, task: str) -> None:
    from supervisor import Supervisor
    game = load(game_id)
    RUNS.mkdir(exist_ok=True)
    try:
        capture.find_window(game.window, game.window_class)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    log = []

    def on_row(row: dict, result: dict) -> None:
        del result
        log.append(row)
        print(json.dumps({
            "step": row.get("step"), "choice": row.get("choice"), "pressed": row.get("pressed"),
            "by": row.get("by"), "see": row.get("see"), "seconds": row.get("seconds") or 0,
        }, ensure_ascii=False), flush=True)

    try:
        result = Supervisor(game, TOKEN).drive(task, learn=True, on_row=on_row)
        if result.get("error"):
            raise SystemExit(str(result["error"]))
    finally:
        release_all()
        (RUNS / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    play(
        sys.argv[1] if len(sys.argv) > 1 else default_id(),
        sys.argv[2] if len(sys.argv) > 2 else "完成当前画面上能做的一步",
    )
