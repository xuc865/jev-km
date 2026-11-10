"""GZDoom map adapter for the general harness.

The map/ collection ships one shared GZDoom runtime and ~95 WADs. This adapter
launches a map's run.bat, captures the GZDoom window, exposes a small FPS macro
action space, and terminates the game on close. The only map-specific data is
the catalog entry (run.bat path, description).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import time
from pathlib import Path

import capture
import harness
import input_backend

from .. import progress_probe
from ..types import Action, Observation

ROOT = Path(__file__).resolve().parents[2]
MAP_ROOT = ROOT / "map"
CATALOG = json.loads((MAP_ROOT / "games_catalog.json").read_text(encoding="utf-8-sig"))
WINDOW_TITLE = "Freedoom"

ACTIONS = [
    Action("forward", "walk forward", kind="hold", keys=("W",), seconds=0.35),
    Action("back", "walk back", kind="hold", keys=("S",), seconds=0.35),
    Action("left", "strafe left", kind="hold", keys=("A",), seconds=0.3),
    Action("right", "strafe right", kind="hold", keys=("D",), seconds=0.3),
    # Turning is a mouse-look action, not an arrow-key action. Measured in one
    # window with a 0.0000 idle baseline: mouse dx +/-160 turned the view 4/4
    # while LEFT/RIGHT changed nothing 0/4 in every encoding tried (scancode,
    # scancode+EXTENDEDKEY, virtual key, 0.3s and 1.0s holds). Arrow keys simply
    # never reach this build, so keeping them as actions would hand JEV choices
    # that are guaranteed to have no consequence.
    Action("turn_left", "turn left", kind="mouse", seconds=0.0, params={"dx": -160}),
    Action("turn_right", "turn right", kind="mouse", seconds=0.0, params={"dx": 160}),
    Action("fire", "fire (left mouse)", kind="click", keys=(), seconds=0.1),
    # GZDoom's stock `use` binding is Space; E is not bound in the shipped
    # config, which is why tapping E never opened anything.
    Action("use", "open/use", kind="tap", keys=("SPACE",), seconds=0.14),
]

# Candidate text is what the fast model actually reads, so each action states
# the concrete in-game effect. Vague labels made JEV spam `fire` while standing
# still in an empty corridor.
USAGE = {
    "forward": "Correct only when the way ahead is clear and the player should advance toward the next room, door or enemy.",
    "back": "Correct only when an enemy or hazard is too close and the player should retreat to regain space.",
    "left": "Correct only when the player should sidestep left to dodge fire or slip past an obstacle.",
    "right": "Correct only when the player should sidestep right to dodge fire or slip past an obstacle.",
    "turn_left": "Correct only when a target is left of the crosshair and the view must be rotated to bring the crosshair onto it.",
    "turn_right": "Correct only when a target is right of the crosshair and the view must be rotated to bring the crosshair onto it.",
    # "an enemy is visible in the frame" is satisfied by almost every frame in a
    # combat room, so the condition collapsed into "fire is always correct" and
    # the collected episodes fired 46 times in 71 steps while the enemy sat left
    # of the crosshair and never took damage. The condition has to name the
    # alignment, not mere visibility.
    "fire": "Correct only when an enemy is already under the crosshair in this frame, so the shot will connect. If the enemy is off to one side, turn first.",
    "use": "Correct only when the player stands directly against a door, switch or lift that must be activated to continue.",
}


def catalog_entry(game_id: str) -> dict:
    for item in CATALOG:
        if item.get("id") == game_id:
            return item
    raise KeyError(f"map game not found: {game_id}")


def env_flags() -> list[str]:
    """Windowed surface flags for the gzdoom command line.

    GZDoom honours these for the render surface, but the shipped config still
    starts in fullscreen exclusive mode in practice, so ``shrink_window`` below
    finishes the job once the window exists.
    """
    return ["+set", "vid_fullscreen", "0", "+set", "vid_defwidth", "1280",
            "+set", "vid_defheight", "720"]


def shrink_window(hwnd: int, width: int = 960, height: int = 600) -> None:
    """Force a game window out of fullscreen into a small desktop window.

    The shipped GZDoom config opens fullscreen exclusive, which ignores the
    +set window hints and covers the entire desktop. Restoring and resizing the
    window after creation is the only reliable way to keep a batch run from
    taking over the screen.
    """
    if not hwnd:
        return
    try:
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.MoveWindow(hwnd, 40, 40, int(width), int(height), True)
    except Exception:
        pass


def _blank(image) -> bool:
    """True when the captured frame still looks like an unrendered surface."""
    try:
        sample = image.convert("L").resize((32, 18))
    except Exception:
        return False
    pixels = list(sample.getdata())
    if not pixels:
        return False
    mean = sum(pixels) / len(pixels)
    return mean < 8


def _frame_signature(image) -> str:
    """Coarse hash of the rendered frame, used as the public dynamic state.

    GZDoom publishes no text state at all. Without something that changes, the
    recurrence tracker sees one identical state for the whole episode and the
    world text keeps telling the model "visits_in_window: N" for a screen it has
    already left. Downsampling first keeps the hash about composition and HUD
    content rather than one-pixel dithering, so it changes when the view
    actually changes and stays stable when the player stands still.
    """
    if image is None:
        return ""
    try:
        small = image.convert("L").resize((24, 14))
    except Exception:
        return ""
    return hashlib.sha256(bytes(small.getdata())).hexdigest()[:16]


def _tokens(line: str) -> list[str]:
    """Split a Windows command line into tokens, honouring double quotes."""
    tokens: list[str] = []
    current: list[str] = []
    quoted = False
    for char in line:
        if char == '"':
            quoted = not quoted
            continue
        if char.isspace() and not quoted:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def gzdoom_command(run_bat: Path) -> list[str] | None:
    """Rebuild the gzdoom command line declared in a map's run.bat.

    The bat forwards nothing to gzdoom, so appending window flags to the bat
    invocation would be ignored. Parsing the single gzdoom line lets us launch
    the exact same IWAD/map with a windowed surface instead of the fullscreen
    default the shipped config uses.
    """
    try:
        text = run_bat.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if "gzdoom.exe" not in stripped.lower():
            continue
        parts = _tokens(stripped)
        if not parts:
            continue
        exe = (run_bat.parent / parts[0]).resolve()
        return [str(exe)] + parts[1:]
    return None


class GzdoomAdapter:
    name = "GZDoom"

    def __init__(self, game_id: str = "basic", goal: str = "") -> None:
        self.entry = catalog_entry(game_id)
        self.goal = goal
        self.run_bat = MAP_ROOT / str(self.entry["entry_path"])
        self.process: subprocess.Popen | None = None
        self.hwnd = 0
        self.first_frame_ready = False
        self._last_signature = ""
        self._distinct_views = 0
        self._frame_count = 0
        self._last_action = ""
        self.last_execution: dict = {}
        # Callers count `executions` to prove the agent actually pressed keys.
        # Web adapters keep this list; the Doom adapter used to lack it, so every
        # Doom run reported 0 actions and looked like the agent never moved.
        self.executions: list[dict] = []
        # The engine publishes its own counters through the ZScript probe in
        # map/_agent/agentprobe.pk3: one AGENT line per second, written to a log
        # file. Before that bridge the adapter captured a native window, so it
        # could only read pixels and every map reported "no metric" even while
        # the player was killing monsters. The log is the missing channel.
        self.log_dir = ROOT / "runs" / "_doomlogs"
        self.log_path = self.log_dir / f"{self.entry.get('id')}.log"
        self._agent_record: dict = {}
        self._last_log_size = 0
        self._origin: tuple[int, int] | None = None
        self.progress: dict = progress_probe.absent()
        self.progress_metric = ""
        self.progress_best: float | None = None
        self.progress_note = "engine probe not loaded yet"
        # A map can load, print its name and still never start: cig-MAP01 exits
        # with "No player 1 start", so the probe writes one final line with every
        # counter at zero. Recording that as "0.0 player-widths travelled" claims
        # a measurement where no episode happened, so track whether the engine
        # ever ticked the world at all.
        self.episode_started = False

    def reset(self) -> None:
        self.launch()

    def _own_window(self) -> int:
        """Window belonging to the process this adapter launched."""
        pid = self.process.pid if self.process is not None else 0
        if not pid:
            return 0
        try:
            return capture.find_window_by_pid(pid)
        except Exception:
            return 0

    def launch(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        # A leftover gzdoom.exe from an earlier run (or a manual session) keeps
        # the same window title, and title lookup would bind this run to that
        # dead process: screenshots show someone else's level and every key and
        # click goes to a window the engine no longer reads. Kill the leftovers
        # so this episode owns the only Doom window on the desktop.
        self._kill_stale_games()
        # Keep the game in a small window: the shipped config opens fullscreen,
        # which would cover the whole desktop for the length of a batch run.
        command = gzdoom_command(self.run_bat)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.unlink(missing_ok=True)
        self._last_log_size = 0
        self._agent_record = {}
        self._origin = None
        # The peak is per episode, not per adapter. Without this reset the second
        # run of a map inherits the first run's best kill count, so two columns
        # labelled "independent seeds" would carry the same number and the sheet
        # would claim two runs where only one had a real peak.
        self.progress_best = None
        self.progress_metric = ""
        self.progress = progress_probe.absent()
        self.executions = []
        self.episode_started = False
        probe = MAP_ROOT / "_agent" / "agentprobe.pk3"
        # Both halves are needed: -file loads the ZScript that reads the engine's
        # counters, +logfile sends the console output somewhere readable. Without
        # the log there is no channel, because GZDoom has no IPC and the adapter
        # only ever sees pixels.
        probe_flags = ["-file", str(probe), "+logfile", str(self.log_path)] if probe.is_file() else []
        # Deathmatch-only maps (cig-MAP01 and its siblings) have no single-player
        # start, so a plain -warp exits with "No player 1 start" and the probe
        # writes one zeroed line. That is a launch-parameter problem, not a broken
        # map: adding -deathmatch makes the same WAD tick normally. Detect it from
        # the catalog's genre instead of hardcoding ids, and only when the run.bat
        # did not already ask for it.
        extra_flags: list[str] = []
        genre = str(self.entry.get("genre", ""))
        if "竞技" in genre or "deathmatch" in genre.lower() or "deathmatch" in str(self.entry.get("id", "")).lower():
            if not any("deathmatch" in str(part).lower() for part in (command or [])):
                extra_flags = ["-deathmatch", "2"]
        if command:
            # +set must precede the file/map arguments: GZDoom executes console
            # commands in order and a trailing one would run after the map load.
            argv = [command[0], *env_flags(), *command[1:], *extra_flags, *probe_flags]
            cwd = str(self.run_bat.parent)
        else:
            argv = ["cmd", "/c", str(self.run_bat), *extra_flags, *probe_flags]
            cwd = str(self.run_bat.parent)
        self.process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 30.0
        while time.time() < deadline:
            time.sleep(0.5)
            # Prefer our own pid; fall back to the title only if the process
            # somehow has no visible window yet.
            self.hwnd = self._own_window() or 0
            if self.hwnd:
                break
        if self.hwnd:
            shrink_window(self.hwnd)
            try:
                capture.focus_window(self.hwnd)
            except Exception:
                pass
            self._wait_first_frame()

    @staticmethod
    def _kill_stale_games() -> None:
        """Terminate gzdoom.exe processes this adapter does not own."""
        try:
            listing = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq gzdoom.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=15,
            ).stdout
        except Exception:
            return
        for line in listing.splitlines():
            parts = [part.strip('" ') for part in line.split(",")]
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            pid = int(parts[1])
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True, timeout=15)
            except Exception:
                continue
        if listing.strip():
            # Give the OS a moment to release the window class and GL context.
            time.sleep(1.5)

    def _wait_first_frame(self, timeout: float = 25.0) -> None:
        """Block until the window paints real pixels.

        GZDoom shows a black surface while it loads the WAD and builds the
        level. Capturing then hands the fast model an all-black image, so the
        first decisions are made blind. Poll until a frame has actual content.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                image = capture.capture_hwnd(self.hwnd)
            except Exception:
                image = None
            if image is not None and not _blank(image):
                self.first_frame_ready = True
                return
            time.sleep(0.5)
        self.first_frame_ready = False

    def _read_probe_log(self) -> dict:
        """Parse the newest AGENT line the engine wrote.

        Read incrementally: the log grows every second for the whole episode and
        re-reading a minute of it on every step would make the tail of a long run
        quadratically slower. Only bytes after the previous offset are scanned,
        and a partial trailing line is left for the next call.
        """
        if not self.log_path.is_file():
            return {}
        try:
            size = self.log_path.stat().st_size
            if size <= self._last_log_size:
                return self._agent_record
            with self.log_path.open("rb") as handle:
                handle.seek(self._last_log_size)
                chunk = handle.read(size - self._last_log_size)
        except OSError:
            return self._agent_record
        # Keep the offset at the last complete line so a half-written record is
        # not parsed as truncated numbers.
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return self._agent_record
        self._last_log_size += cut + 1
        text = chunk[:cut].decode("utf-8", errors="ignore")
        for row in text.splitlines():
            if "AGENT|" not in row:
                continue
            record: dict[str, int] = {}
            for part in row.split("AGENT|", 1)[1].split("|"):
                key, _, value = part.partition("=")
                try:
                    record[key.strip()] = int(float(value))
                except ValueError:
                    continue
            if record:
                self._agent_record = record
        return self._agent_record

    def _publish_progress(self) -> dict:
        """Turn the engine's counters into the progress record callers expect.

        `killed` is the level's kill count, which is what "how far did this map
        get" means in Doom. `total` rides along because 3 kills out of 3 is a
        cleared level and 3 out of 40 is barely started, and a bare kill number
        cannot tell those apart. Health is kept as a separate field: it is state,
        not progress, and folding it into the metric would let a player inflate
        "progress" by taking damage.

        This is called from both observe() and the progress getter. The getter
        runs on a different cadence than the step loop, so relying on observe()
        alone would let a sample read whatever the last step happened to see.
        """
        record = self._read_probe_log()
        if not record:
            self.progress = progress_probe.absent()
            self.progress["note"] = ("window-capture adapter found no AGENT line from the "
                                     "engine probe; the map may still be loading")
            return dict(self.progress)
        killed = record.get("killed", 0)
        total = record.get("total", 0)

        # The probe's first sample arrives at tics=35. A record with tics==0 is
        # the shutdown write from WorldUnloaded, which fires even when the map
        # refused to start ("No player 1 start"). Seeing a positive tic count at
        # least once is the only evidence here that the world actually ran.
        if record.get("tics", 0) > 0:
            self.episode_started = True
        if not self.episode_started:
            self.progress = progress_probe.absent()
            self.progress["note"] = ("the engine never ticked this map: it loaded "
                                     "and exited without a playable start")
            self.progress["started"] = False
            return dict(self.progress)

        # A map with no monsters (a corridor run, a health-gathering course, a
        # navigation test) has a kill count that is structurally always zero.
        # Reporting it would record a constant and look like measured progress,
        # which is exactly the failure that produced 29 rows of identical values
        # in this sheet. For those maps distance travelled is the honest signal:
        # it starts at 0 and only grows while the player actually moves.
        if total <= 0:
            here = (record.get("x", 0), record.get("y", 0))
            if self._origin is None:
                self._origin = here
            moved = ((here[0] - self._origin[0]) ** 2 + (here[1] - self._origin[1]) ** 2) ** 0.5
            # Map units: 64 units is roughly one player width in Doom, so divide
            # to keep the number in a readable range instead of six digits.
            travelled = round(moved / 64.0, 1)
            self.progress_metric = "distance"
            self.progress_best = travelled if self.progress_best is None else max(self.progress_best, travelled)
            self.progress = {
                "metric": "distance",
                "sense": "high",
                "value": travelled,
                "best": float(self.progress_best),
                "episode_best": float(self.progress_best),
                "total": 0,
                "health": record.get("health"),
                "alive": bool(record.get("alive")),
                "source": "engine-log",
                # Use the peak, not this sample: `best` is what callers read as
                # "how far did this run get", and a note built from the latest
                # sample would report a different number than that same call.
                "note": f"map has no monsters; {self.progress_best} player-widths travelled",
            }
            return dict(self.progress)

        self.progress_metric = "kills"
        if self.progress_best is None:
            self.progress_best = float(killed)
        else:
            self.progress_best = max(self.progress_best, float(killed))
        self.progress = {
            "metric": "kills",
            "sense": "high",
            "value": float(killed),
            "best": float(self.progress_best),
            "episode_best": float(self.progress_best),
            "total": total,
            "health": record.get("health"),
            "alive": bool(record.get("alive")),
            "source": "engine-log",
            "note": f"{int(self.progress_best)}/{total} monsters killed",
        }
        return dict(self.progress)

    def refresh_progress(self) -> dict:
        """Public hook so the collector can pull the latest counters.

        The SFT sampler reads progress on its own cadence, not inside the step
        loop, so without this it would publish whatever the last observe() saw.
        """
        return self._publish_progress()

    def observe(self) -> Observation:
        if not self.hwnd:
            self.hwnd = self._own_window()
        if self.hwnd:
            # A resolution switch mid-game can re-enter fullscreen; keep the
            # window small so the desktop stays usable.
            try:
                left, top, right, bottom = capture.window_box(self.hwnd)
                if (right - left) > 1600 or (bottom - top) > 1000:
                    shrink_window(self.hwnd)
            except Exception:
                pass
        image = None
        if self.hwnd:
            for attempt in range(6):
                try:
                    image = capture.capture_hwnd(self.hwnd)
                except Exception:
                    image = None
                if image is not None and not _blank(image):
                    break
                time.sleep(0.4)
        done = self.process is not None and self.process.poll() is not None
        signature = _frame_signature(image)
        changed = bool(signature) and signature != self._last_signature
        if signature:
            if changed:
                self._distinct_views += 1
            self._last_signature = signature
        frame_variety = (
            round(self._distinct_views / self._frame_count, 3) if self._frame_count else 0.0
        )
        self._frame_count += 1
        progress = self._publish_progress()
        actions = []
        for action in ACTIONS:
            label = action.label
            if action.id in USAGE:
                # Surface the usage condition in the action label so the same
                # text reaches both the criteria list and the world summary.
                label = f"{action.label} — {USAGE[action.id]}"
            actions.append(Action(action.id, label, kind=action.kind,
                                  keys=action.keys, seconds=action.seconds,
                                  params=dict(action.params)))
        return Observation(
            actions=actions,
            image=harness._image_url(image) if image is not None else "",
            state={
                "game": self.entry.get("id"),
                "description": self.entry.get("description"),
                "window": self.hwnd,
                "view": {
                    # The only dynamic signal this adapter can produce. Without
                    # it every step claims to be the same state, which both
                    # confuses the recurrence tracker and teaches the model that
                    # its actions have no effect.
                    "signature": signature,
                    "changed_since_last_step": changed,
                    "distinct_views": self._distinct_views,
                    "frames_observed": self._frame_count,
                    "change_ratio": frame_variety,
                },
                "last_action": self._last_action or "none",
                "last_execution": self.last_execution or "first step, nothing executed yet",
                "progress": {key: progress.get(key) for key in
                             ("metric", "value", "best", "total", "health", "alive", "note")},
                "controls": {
                    "fire": "left mouse button (cursor is re-seated inside the game window before every shot)",
                    "turn": "mouse look left/right",
                    "move": "W/S forward-back, A/D strafe, Space use",
                },
            },
            level=1,
            done=done,
            success=False,
            raw={"entry": self.entry, "frame_signature": signature},
        )

    def _center(self) -> tuple[int, int]:
        """Screen point inside the game client area, where mouse input lands."""
        try:
            left, top, right, bottom = capture.client_box(self.hwnd)
            return (left + right) // 2, (top + bottom) // 2
        except Exception:
            try:
                left, top, right, bottom = capture.window_box(self.hwnd)
                return (left + right) // 2, (top + bottom) // 2
            except Exception:
                return (0, 0)

    def act(self, action: Action) -> None:
        if not self.hwnd:
            return
        try:
            capture.focus_window(self.hwnd)
        except Exception:
            pass
        keys = [str(key) for key in action.keys]
        self._last_action = action.id
        self.last_execution = {
            "action": action.id,
            "ok": True,
            "kind": action.kind,
            "keys": keys,
            "seconds": action.seconds,
        }
        if action.kind == "hold" and keys:
            for key in keys:
                input_backend.key_down(key)
            try:
                time.sleep(max(0.05, action.seconds))
            finally:
                for key in keys:
                    input_backend.key_up(key)
        elif action.kind in {"tap", "key"}:
            for key in keys:
                input_backend.tap_key(key)
        elif action.kind == "click":
            # A delta-less click is routed to the window under the pointer. The
            # pointer normally rests on the user's desktop, so the game never
            # saw the shot; re-seat it inside the client area first, then hold
            # the button long enough for the engine to sample it.
            x, y = self._center()
            input_backend.mouse_click_at(x, y, hold=max(0.12, float(action.seconds or 0.1)))
            self.last_execution["click_at"] = [x, y]
        elif action.kind == "mouse":
            # Relative mouse motion is delivered to the window under the pointer.
            # The pointer normally rests on the desktop, so a bare mouse_move
            # rotates nothing; seat it in the client area first.
            x, y = self._center()
            input_backend.move_cursor(x, y)
            time.sleep(0.03)
            dx = int(action.params.get("dx") or 0)
            dy = int(action.params.get("dy") or 0)
            input_backend.mouse_move(dx, dy)
            self.last_execution["mouse_move"] = [dx, dy]
            self.last_execution["cursor_at"] = [x, y]
        if self.last_execution:
            self.executions.append(dict(self.last_execution))

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
