"""Generic headless adapter for HTML games that have no knowledge pack.

Games with a ``knowledge/<id>/runtime.json`` pack get a hand-written state
projection, which is the right thing when someone has studied the game. The
rest of the catalog has no pack, and the older fallback launched a real Chrome
window and drove it through the OS cursor. That window is intrusive and the
captured frame is a desktop screenshot rather than the page itself.

This adapter keeps everything inside a headless browser: the page is served by
the same private host, the observation is the DOM snapshot plus a real page
screenshot, and actions are keys or clicks sent to the page. No window opens,
nothing takes over the pointer.

It deliberately knows nothing about any individual game. The action space is
generic (arrow/WASD movement, confirm, and every visible control), and the model
decides what those mean from the screenshot and the visible text.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from webenv.host import BrowserHost
from webenv.timing import timing_contract
from webenv.dom import terminal as dom_terminal, phase as dom_phase

from ..types import Action, Observation
from .. import progress_probe

ROOT = Path(__file__).resolve().parents[2]
MAP_ROOT = Path(os.environ.get("JEVGAMER_MAP_ROOT") or (ROOT / "map")).resolve()

# Generic input the overwhelming majority of web games accept. These are
# offered unconditionally because without a pack there is no declarative
# "when" condition to gate them on; the model picks between them.
MOVE_KEYS = [
    ("move_up", "向上 / 前进（ArrowUp）", ["ArrowUp"]),
    ("move_down", "向下 / 后退（ArrowDown）", ["ArrowDown"]),
    ("move_left", "向左（ArrowLeft）", ["ArrowLeft"]),
    ("move_right", "向右（ArrowRight）", ["ArrowRight"]),
    ("move_up_alt", "向上 / 前进（W）", ["w"]),
    ("move_down_alt", "向下 / 后退（S）", ["s"]),
    ("move_left_alt", "向左（A）", ["a"]),
    ("move_right_alt", "向右（D）", ["d"]),
    ("confirm", "确认 / 开火 / 跳跃（Space）", ["Space"]),
    ("enter", "确认 / 开始（Enter）", ["Enter"]),
    ("escape", "取消 / 暂停（Escape）", ["Escape"]),
    ("restart", "重新开始（R）", ["r"]),
]

# The bare minimum a game needs to be *played* once it has started. Used only
# after the letter tiers take over, so the per-step candidate budget does not
# fill up with arrows that have already been proven ineffective.
CORE_KEYS = [
    ("move_up", "向上 / 前进（ArrowUp）", ["ArrowUp"]),
    ("move_down", "向下 / 后退（ArrowDown）", ["ArrowDown"]),
    ("move_left", "向左（ArrowLeft）", ["ArrowLeft"]),
    ("move_right", "向右（ArrowRight）", ["ArrowRight"]),
    ("confirm", "确认 / 开火 / 跳跃（Space）", ["Space"]),
    ("enter", "确认 / 开始（Enter）", ["Enter"]),
]

# A lot of the catalogue does not listen to arrows at all. Four-key rhythm games
# bind D/F/J/K, piano games bind whole QWERTY rows, card and board games pick a
# choice with a digit. With only arrows+WASD+Space a run of those games sat on
# the instruction screen pressing keys nobody was listening for, the screenshot
# never changed, and the progress metric stayed empty for the whole episode --
# which is exactly why so many archived episodes had no numeric value at all.
# They are offered in tiers once the base set has demonstrably failed, otherwise
# forty keys would crowd the arrows out of the candidate budget from step one.
LETTER_KEYS = [
    ("key_d", "按字母键 D（四键音游 / 连击键位）", ["d"]),
    ("key_f", "按字母键 F（四键音游 / 连击键位）", ["f"]),
    ("key_j", "按字母键 J（四键音游 / 连击键位）", ["j"]),
    ("key_k", "按字母键 K（四键音游 / 连击键位）", ["k"]),
    ("key_q", "按字母键 Q（钢琴 / 技能键位）", ["q"]),
    ("key_w", "按字母键 W（钢琴 / 技能键位）", ["w"]),
    ("key_e", "按字母键 E（钢琴 / 技能键位）", ["e"]),
    ("key_r", "按字母键 R（钢琴 / 技能键位）", ["r"]),
    ("key_t", "按字母键 T（钢琴 / 技能键位）", ["t"]),
    ("key_y", "按字母键 Y（钢琴 / 技能键位）", ["y"]),
    ("key_u", "按字母键 U（钢琴 / 技能键位）", ["u"]),
    ("key_i", "按字母键 I（钢琴 / 技能键位）", ["i"]),
    ("key_o", "按字母键 O（钢琴 / 技能键位）", ["o"]),
    ("key_p", "按字母键 P（钢琴 / 技能键位）", ["p"]),
    ("key_a", "按字母键 A（左手键位 / 移动）", ["a"]),
    ("key_s", "按字母键 S（左手键位 / 移动）", ["s"]),
    ("key_g", "按字母键 G（左手键位）", ["g"]),
    ("key_h", "按字母键 H（右手键位）", ["h"]),
    ("key_l", "按字母键 L（右手键位）", ["l"]),
    ("key_z", "按字母键 Z（辅助键位）", ["z"]),
    ("key_x", "按字母键 X（辅助键位）", ["x"]),
    ("key_c", "按字母键 C（辅助键位）", ["c"]),
    ("key_v", "按字母键 V（辅助键位）", ["v"]),
    ("key_b", "按字母键 B（辅助键位）", ["b"]),
    ("key_n", "按字母键 N（辅助键位）", ["n"]),
    ("key_m", "按字母键 M（辅助键位）", ["m"]),
]

# Card, board and menu screens take a bare digit as the pick. The whole row is
# offered because hotbar slots, dialogue choices and deck picks are numbered and
# nothing in the DOM says which numbers are live.
DIGIT_KEYS = [
    ("digit_0", "按数字键 0（选牌 / 选项 / 快捷栏）", ["0"]),
    ("digit_1", "按数字键 1（选牌 / 选项 / 快捷栏）", ["1"]),
    ("digit_2", "按数字键 2（选牌 / 选项 / 快捷栏）", ["2"]),
    ("digit_3", "按数字键 3（选牌 / 选项 / 快捷栏）", ["3"]),
    ("digit_4", "按数字键 4（选牌 / 选项 / 快捷栏）", ["4"]),
    ("digit_5", "按数字键 5（选牌 / 选项 / 快捷栏）", ["5"]),
    ("digit_6", "按数字键 6（选牌 / 选项 / 快捷栏）", ["6"]),
    ("digit_7", "按数字键 7（选牌 / 选项 / 快捷栏）", ["7"]),
    ("digit_8", "按数字键 8（选牌 / 选项 / 快捷栏）", ["8"]),
    ("digit_9", "按数字键 9（选牌 / 选项 / 快捷栏）", ["9"]),
]

# Menu traversal that is not movement: focus is moved by Tab in a lot of
# desktop-style web games, and Backspace closes the panel that blocked the play.
NAV_KEYS = [
    ("key_tab", "按 Tab（切换焦点 / 菜单翻页）", ["Tab"]),
    ("key_backspace", "按 Backspace（退格 / 返回上一页）", ["Backspace"]),
]

DOM_SNAPSHOT = "window.__JEV_DOM__.snapshot(document)"

# Canvas grids for click-driven games. A tower defence or card game ignores the
# keyboard entirely, so without these the model could only press arrows and the
# metric stayed pinned at its opening value for the whole episode. The points are
# a coarse 3x3 lattice with the centre weighted in, which is enough to hit a
# tile, a card or a button without exploding the action list.
CLICK_GRID = [
    ("click_center", "点击画面中心", 0.50, 0.50),
    ("click_top_left", "点击左上区域", 0.25, 0.25),
    ("click_top", "点击上方区域", 0.50, 0.25),
    ("click_top_right", "点击右上区域", 0.75, 0.25),
    ("click_left", "点击左侧区域", 0.25, 0.50),
    ("click_right", "点击右侧区域", 0.75, 0.50),
    ("click_bottom_left", "点击左下区域", 0.25, 0.75),
    ("click_bottom", "点击下方区域", 0.50, 0.75),
    ("click_bottom_right", "点击右下区域", 0.75, 0.75),
]

# A coarse 3x3 lattice cannot hit a specific tile in a tower-defence board or a
# single card in a hand: the points land between targets and nothing happens.
# This finer lattice is the same picture at 4x4 spacing, used only after the
# coarse points have failed, so the extra sixteen candidates do not compete with
# the keys while a game is still answering them.
FINE_CLICK_GRID = [
    (f"click_g{row}{col}", f"点击第 {row + 1} 行第 {col + 1} 列细网格", (col + 1) / 5, (row + 1) / 5)
    for row in range(4)
    for col in range(4)
]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-") or "game"


def _page_key(url: str) -> str:
    """Page identity for restart detection: path only, ignoring the query string.

    A self-reload keeps the path but may rewrite `?jev_seed=`, so comparing raw
    URLs would miss it, while a real navigation almost always changes the path.
    """
    text = str(url or "")
    if "://" in text:
        text = text.split("://", 1)[1]
    return text.split("?", 1)[0].split("#", 1)[0]


# Per-game record of which inputs the slow model has ruled out. Kept outside
# knowledge/<id>/ because that directory is the hand-written pack contract, and
# these games deliberately have no pack. The slow model rewrites this file on
# every review it makes, so a stale decision costs one review rather than
# poisoning every later episode.
SUBSET_ROOT = ROOT / "knowledge" / "_subsets"

# Whatever the slow model decides, the episode must keep enough input to act:
# dropping the whole keyboard on a wrong guess would end the run in place.
# A bare count was the wrong floor. `core` is played by holding Space, so the
# correct verdict ("46 of 50 keys are useless") left three usable ids and was
# refused for being small rather than for being unusable. What actually matters
# is whether something able to drive the game survived, which _is_play_key
# answers; the count only guards against a literally empty keyboard.
MIN_KEYS = 1


def _is_play_key(key: str) -> bool:
    """True when the id can drive a game rather than only open or leave a menu.

    `escape`, `restart`, `key_tab` and `key_backspace` navigate UI chrome. A
    subset made only of those passes an "is it empty" test while leaving nothing
    to play, which is how a committed verdict could still stall an episode.
    """
    if key in {"move_up", "move_down", "move_left", "move_right",
               "move_up_alt", "move_down_alt", "move_left_alt", "move_right_alt",
               "confirm", "enter"}:
        return True
    if key.startswith("digit_"):
        return True
    return key.startswith("key_") and key not in {"key_tab", "key_backspace"}

# Clicks must keep a foothold too. A tower defence, card table or menu page is
# only operable by pointing, so the keyboard cannot be allowed to consume the
# entire candidate budget now that it publishes a much larger key table.
MIN_CLICK_SLOTS = 12


class GenericHtmlAdapter:
    """Headless, pack-free adapter: DOM + screenshot in, keys/clicks out."""

    name = "HTML5 通用无窗口"

    # Consecutive steps without a new distinct view before the page-level start
    # nudge fires. Roughly ten steps is long enough that a slow-loading level is
    # not mistaken for a title screen, and short enough to matter inside a
    # ten-minute budget.
    MENU_STUCK_LIMIT = 10

    # Steps without the progress metric moving before the letter/digit keys are
    # unlocked. Four is roughly when a game that was going to respond to arrows
    # already has; waiting longer just wastes the episode budget.
    KEY_TIER_STEPS = 4

    # Keyboard candidates that must survive the candidate cap no matter how many
    # DOM controls the page publishes. Below this the model can be left with
    # nothing playable at all.
    MIN_KEY_SLOTS = 14

    # Mirror of the module-level floor for click candidates. Defined on the class
    # as well because the budget split reads it per instance.
    MIN_CLICK_SLOTS = 12

    def __init__(self, game_id: str, goal: str = "", entry: dict | None = None,
                 press_frames: int = 24, max_candidates: int = 64) -> None:
        self.game_id = game_id
        self.goal = goal
        self.entry = entry or {}
        entry_path = str(self.entry.get("entry_path") or "")
        self.entry_path = entry_path.lstrip("/\\")
        self.press_frames = max(1, int(press_frames))
        self.max_candidates = max(2, int(max_candidates))
        # No pack means no declarative timing contract beyond the default step
        # clock, and no source hooks: the page is driven exactly as shipped.
        self.config: dict = {
            "time": {"mode": "step", "dt_ms": 1000 / 60, "initial_frames": 2},
            "snapshot": DOM_SNAPSHOT,
            "observation": "dom",
            # The default 1920x1080 viewport is what the curated packs recorded at,
            # but for pack-free games the model only needs to recognise sprites and
            # read the HUD. Rendering and PNG-encoding four full-HD frames per step
            # pinned all 24 cores at 48 workers, so steps stretched and throughput
            # fell. 1280x720 keeps every HUD legible at roughly half the pixel cost;
            # JEVGAMER_VIEWPORT can override it for a game whose text becomes unreadable.
            "viewport": {"width": 1280, "height": 720},
            # This adapter publishes the full input universe in its observation,
            # which only the slow model can act on, so the review that carries the
            # pruning decision must actually run. Without this the slow model would
            # never be asked and the subset would stay empty. The per-action
            # conditions it writes are injected by JevPolicy._criteria on every
            # step regardless of context mode, because this adapter has no
            # compact `fast_state` projection to switch to.
            "slow_action_criteria": True,
        }
        self.host: BrowserHost | None = None
        self.artifacts: Path | None = None
        self.preview = ""
        self._state: dict = {}
        self.last_execution: dict = {}
        self.executions: list[dict] = []
        self.steps = 0
        self._page = 0
        self._terminal_settled = False
        # Progress accounting. `best` is the peak over the episode, which is the
        # number the WeCom sheet records, so both stay comparable. Seeded from the
        # static catalogue spec so even the first sample, taken before the game
        # starts, names the metric instead of carrying an empty field.
        self.progress: dict = progress_probe.pending(self.game_id, None)
        self.progress_best: float | None = None
        self.progress_metric = ""
        self._progress_injected = False
        # Optional per-game pruning decided by the slow model. Empty means the
        # full universe is offered, exactly as before this feature existed.
        self.excluded_note = ""
        self.excluded: set[str] = self._load_subset()
        self.subset_history: list[dict] = []

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._terminal_settled = False
        folder = Path(self.artifacts) if self.artifacts else Path(tempfile.mkdtemp(prefix="jevgamer-generic-"))
        self.artifacts = folder
        # `Math.random` is replaced by a seeded generator (webenv/clock.js), so a
        # fixed seed makes every rerun of a game replay the same opening hands,
        # spawns and item drops. Collecting the same game repeatedly is how the
        # sample count grows, but with seed 42 hardcoded it grew by duplicating
        # one trajectory instead of covering different ones. JEVGAMER_SEED lets
        # the batch runner vary it per episode.
        seed = int(os.environ.get("JEVGAMER_SEED") or 42)
        self.host = BrowserHost(
            self.entry_path, self.config, ROOT / "knowledge" / _slug(self.game_id),
            folder, seed=seed, record=False, asset_root=MAP_ROOT,
        )
        self.host.start()
        self.progress = progress_probe.pending(self.game_id, None)
        self.progress_best = None
        self.progress_metric = ""
        self._progress_injected = progress_probe.inject(self.host, self.game_id)

    def close(self) -> None:
        if self.host is not None:
            try:
                self.host.close()
            finally:
                self.host = None

    # -- slow-model input pruning -----------------------------------------
    @property
    def subset_path(self) -> Path:
        return SUBSET_ROOT / f"{_slug(self.game_id)}.json"

    def _load_subset(self) -> set[str]:
        """Read the slow model's last exclusions for this game, if any.

        Only ids that exist in the current universe survive: the key tables grow
        over time, and a file written before a key existed must not silently
        suppress it. A corrupt file is treated as "no pruning" rather than
        failing the episode.
        """
        try:
            raw = json.loads(self.subset_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        if not isinstance(raw, dict):
            return set()
        items = raw.get("exclude")
        if not isinstance(items, list):
            return set()
        self.excluded_note = str(raw.get("note") or "")[:300]
        known = {key for key, _label, _keys in self._universe_keys()}
        stored = {str(item) for item in items if str(item) in known}
        # Older files stored only the id the model named, which can be one of
        # several ids sending the same physical key. Expand them on load so the
        # aliases are honoured for every episode, not just the one that wrote it.
        physical: dict[str, set[str]] = {}
        for key, _label, keys in self._universe_keys():
            physical[key] = {str(item) for item in keys}
        banned = set()
        for item in stored:
            banned |= physical.get(item, set())
        return {key for key, keys in physical.items() if keys & banned}

    def _universe_keys(self) -> list:
        """Every keyboard input the generic adapter is able to send.

        Built from the widest tables rather than from a single tier: the slow
        model prunes against this list, so a key the harness can send but the
        list omits would make a correct exclusion look like an invented ID and
        the whole response would be refused. Order is stable and duplicates are
        dropped by id.
        """
        seen: set[str] = set()
        universe: list = []
        for item in MOVE_KEYS + LETTER_KEYS + DIGIT_KEYS + NAV_KEYS:
            if item[0] in seen:
                continue
            seen.add(item[0])
            universe.append(item)
        return universe

    def apply_subset(self, exclude, note: str = "", source: str = "slow") -> dict:
        """Record the slow model's verdict on which keys this game ignores.

        The model is asked for a subset of the full universe, which is stated in
        the prompt, so an exclusion is an explicit claim that the key is useless
        for this game rather than an accident of the tier that happened to be
        active when it looked. Anything outside the universe is discarded, and
        the result is clamped so a wrong guess cannot empty the keyboard.
        """
        known = {key for key, _label, _keys in self._universe_keys()}
        requested = {str(item) for item in (exclude or []) if str(item) in known}
        # Several action ids can send the same physical key: `key_w` and
        # `move_up_alt` are both W, `key_a` and `move_left_alt` are both A. If
        # only the named id were dropped, the game would still receive the key
        # through its alias and the pruning would silently do nothing. Aliases
        # are therefore removed together, matched on the physical key they send.
        physical: dict[str, set[str]] = {}
        for key, _label, keys in self._universe_keys():
            physical[key] = {str(item) for item in keys}
        banned = set()
        for item in requested:
            banned |= physical.get(item, set())
        expanded = {key for key, keys in physical.items() if keys & banned}
        keep = [key for key, _label, _keys in self._universe_keys() if key not in expanded]
        playable = [key for key in keep if _is_play_key(key)]
        if len(keep) < MIN_KEYS or not playable:
            # Refusing is better than obeying: an empty or menu-only keyboard
            # leaves the model with nothing that advances the game. The reason
            # names what survived so the next review can correct itself instead
            # of resubmitting the same rejected list every step.
            reason = ("subset would leave no play key" if not playable
                      else f"subset would leave {len(keep)} keys, minimum {MIN_KEYS}")
            self.subset_history.append({"source": source, "requested": sorted(requested),
                                        "exclude": sorted(expanded), "note": reason,
                                        "rejected": True, "keep": keep})
            return {"ok": False, "reason": reason, "excluded": sorted(self.excluded),
                    "requested": sorted(requested), "keep": keep}
        self.excluded = expanded
        self.excluded_note = str(note or "")[:300]
        self.subset_history.append({"source": source, "requested": sorted(requested),
                                    "exclude": sorted(expanded), "note": self.excluded_note})
        record = {"game": self.game_id, "requested": sorted(requested), "exclude": sorted(expanded),
                  "keep": keep, "note": self.excluded_note, "source": source}
        try:
            self.subset_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.subset_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.subset_path)
        except OSError:
            pass
        return {"ok": True, "excluded": sorted(expanded), "requested": sorted(requested),
                "kept": len(keep)}

    # -- observation -------------------------------------------------------
    def _read(self) -> dict:
        state = self.host.snapshot()
        if not isinstance(state, dict):
            raise RuntimeError("DOM snapshot must be an object")
        # Deliberately left as the raw DOM signature. This value gates the
        # "stale observation" check, which compares the signature attached to an
        # action against the one read just before executing it. Pixels must not be
        # mixed in here: an animating game produces a different hash every frame,
        # so every action would be rejected as out of date and never reach the
        # page. Change detection uses view.signature instead, which is where
        # pixel sensitivity is actually wanted.
        return state

    def _pixel_signature(self) -> str:
        try:
            shoot = getattr(self.host, "thumb", None)
            shot = shoot() if callable(shoot) else self.host.screenshot()
        except Exception:
            return ""
        if not shot:
            return ""
        try:
            from io import BytesIO
            from PIL import Image
            image = Image.open(BytesIO(shot)).convert("L").resize((32, 20))
            return hashlib.sha256(bytes(image.getdata())).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(shot).hexdigest()[:16]

    def _progress_record(self) -> dict:
        """Sample the game's own progress metric and keep the episode peak.

        Games run a real clock, so the peak can occur between two observations.
        `act()` also samples after each action, which is why `best` is maintained
        here rather than recomputed from the observation history.
        """
        record = progress_probe.read(self.host) if self.host is not None else None
        if record is None:
            # A game that reloads itself (mkjs binds Space to location.reload)
            # destroys the injected script along with the old document. Reading
            # null afterwards would look like "the game has no metric" for the
            # rest of the episode, hiding every hit that happens after the
            # restart. Re-inject once and re-read before giving up.
            if self.host is not None and not progress_probe.live(self.host):
                if progress_probe.inject(self.host, self.game_id, wait_s=1.0):
                    record = progress_probe.read(self.host)
            if record is None:
                self.progress = progress_probe.pending(self.game_id, self.progress_best)
                return dict(self.progress)
        best = record.get("best")
        if isinstance(best, (int, float)):
            lower_is_better = str(record.get("sense") or "high") == "low"
            if self.progress_best is None:
                self.progress_best = best
            elif lower_is_better:
                self.progress_best = min(self.progress_best, best)
            else:
                self.progress_best = max(self.progress_best, best)
        self.progress_metric = str(record.get("metric") or self.progress_metric)
        record["source"] = "game"
        record["episode_best"] = self.progress_best
        self.progress = dict(record)
        return dict(self.progress)

    def _key_tier(self) -> int:
        """How wide the keyboard action space should be, given how stuck we are.

        Offering forty letter keys from the first step would push the arrows the
        model actually needs out of the candidate budget, so the extra keys are
        unlocked in tiers. The counter that drives this is the honest one: it
        only grows while the game's own progress metric refuses to advance, so a
        game that is genuinely being played keeps tier 0 and sees the same action
        space as before this change.
        """
        stalled = getattr(self, "_no_progress_steps", 0)
        if stalled < self.KEY_TIER_STEPS:
            return 0
        if stalled < self.KEY_TIER_STEPS * 2:
            return 1
        return 2

    def _declared_keys(self) -> list:
        """Keys the page itself names as its controls, in declaration order.

        A game whose title screen says "G / J 移动, A 重拳" is telling us exactly
        which keys work, yet the generic table offers arrows instead. mkjs is the
        pathological case: it is a two-player page where arrows drive the second
        fighter, so an agent pressing arrows moves its opponent and can never
        close the distance to attack - the metric reads 0 for a reason that is
        pure harness limitation.

        Only explicit key mentions count. A bare letter in prose is not control
        documentation, so the match requires a nearby 键/移动/攻击/shortcut word,
        which keeps this from firing on ordinary sentences.
        """
        state = getattr(self, "_state", None) or {}
        text = str(state.get("text") or "")
        if not text:
            return []
        hints = re.findall(
            r"([A-Za-z0-9][A-Za-z0-9 /+\-]{0,24})\s*(?:键|快捷键|移动|攻击|跳跃|跳跃|格挡|重拳|轻拳|踢|开打)",
            text,
        )
        wanted: list[str] = []
        for chunk in hints:
            for token in re.findall(r"[A-Za-z0-9]", chunk):
                wanted.append(token.lower())
        if not wanted:
            return []
        table: list = []
        seen: set[str] = set()
        for token in wanted:
            for item in self._universe_keys():
                key_id, _label, keys = item
                if key_id in seen:
                    continue
                if any(str(k).lower() == token for k in keys):
                    table.append(item)
                    seen.add(key_id)
        return table

    def _key_table(self) -> list:
        tier = self._key_tier()

        def broadest() -> list:
            # Everything, minus the slow model's exclusions. The order matters:
            # core keys stay first so the budget cap can never remove the ability
            # to keep moving.
            table = CORE_KEYS + LETTER_KEYS + DIGIT_KEYS + NAV_KEYS
            extra = [item for item in self._universe_keys()
                     if item[0] not in {entry[0] for entry in table}]
            return table + extra

        def narrowed(table: list) -> list:
            # The slow model's judgement narrows the tier, it does not replace it:
            # the surviving keys keep their tier order so a pruned run does not
            # suddenly offer a letter before the arrows the model needs to move.
            if not self.excluded:
                return table
            return [item for item in table if item[0] not in self.excluded]

        if tier == 0:
            base = MOVE_KEYS
        elif tier == 1:
            # Rhythm and piano games are by far the most common of the games that
            # ignore arrows, and they bind a handful of neighbouring letters.
            base = CORE_KEYS + LETTER_KEYS[:14] + DIGIT_KEYS[:3]
        else:
            base = broadest()

        kept = narrowed(base)
        # Keys the page documents outrank the generic guess. They are placed
        # first rather than replacing the table: if the reading was wrong, the
        # arrows are still there, so a misparse costs a few steps instead of
        # leaving the agent with nothing playable.
        declared = narrowed(self._declared_keys())
        if declared:
            head = {item[0] for item in declared}
            kept = declared + [item for item in kept if item[0] not in head]
        if not self.excluded or len(kept) >= MIN_KEYS:
            return kept
        # The tier is a heuristic about games in general; the subset is a verdict
        # about this game. When the tier happens to hold almost only keys the
        # verdict already ruled out, widening to the tier's untrimmed table would
        # hand back exactly the keys that were excluded. Re-select from the pruned
        # universe instead, so a small but correct subset still takes effect.
        widened = narrowed(broadest())
        return widened if widened else base

    def _actions(self, frame: dict, signature: str) -> list[Action]:
        actions: list[Action] = []
        for key, label, keys in self._key_table():
            actions.append(Action(key, label, kind="web_keys", keys=tuple(keys),
                                  seconds=self.press_frames / 60,
                                  params={"frames": self.press_frames, "signature": signature}))
        clicks: list[Action] = []
        for item in frame.get("candidates") or []:
            if not item.get("id") or not item.get("selector"):
                continue
            clicks.append(Action(
                str(item["id"]), str(item.get("label") or item["id"]),
                kind="dom_click", seconds=0,
                params={"selector": item["selector"], "signature": signature, "frames": 1},
            ))
        # Games whose metric is readable but has not budged are usually waiting
        # for the mouse: a tower defence wants a tile clicked, a card game wants
        # a card picked. Keys cannot express that, so the whole episode used to
        # press arrows while the score stayed at its opening value. Once the run
        # has gone a few steps without any advance, the canvas lattice is offered
        # alongside the keys and the model can finally try pointing at things.
        clicking = getattr(self, "_no_progress_steps", 0) >= 4
        if clicking:
            for key, label, xr, yr in CLICK_GRID:
                clicks.append(Action(
                    key, label, kind="canvas_click", seconds=0,
                    params={"x": xr, "y": yr, "signature": signature, "frames": 1},
                ))
            # The coarse points can land between two tiles and do nothing at all.
            # Once they have had their chance, the finer lattice is appended so a
            # board, a hand or a build slot can actually be hit.
            if getattr(self, "_no_progress_steps", 0) >= 8:
                for key, label, xr, yr in FINE_CLICK_GRID:
                    clicks.append(Action(
                        key, label, kind="canvas_click", seconds=0,
                        params={"x": xr, "y": yr, "signature": signature, "frames": 1},
                    ))
        # Both families must survive the cap. The old rule put every click ahead
        # of every key, so a page that published a long list of DOM candidates
        # (a lobby with dozens of buttons, a card table of hand slots) consumed
        # the entire budget and the keyboard list was silently truncated to
        # nothing. The model then had no key to press at all and the episode
        # ended on a screen it could never operate. Keys are what actually play
        # a game, so they are placed first and clicks take what is left, with a
        # floor reserved for clicks so a click-driven game stays playable.
        key_budget = min(len(actions), max(self.MIN_KEY_SLOTS,
                                           self.max_candidates - self.MIN_CLICK_SLOTS))
        click_budget = max(0, self.max_candidates - key_budget)
        if len(clicks) > click_budget:
            # Coarse points come first in `clicks`, so truncation drops the fine
            # lattice before it drops the points that at least hit a region.
            clicks = clicks[:click_budget]
        actions = actions[:key_budget] + clicks
        if len(actions) > self.max_candidates:
            actions = actions[: self.max_candidates]
        return actions

    def observe(self) -> Observation:
        frame = self._read()
        done, success = dom_terminal(frame.get("headings") or [], self.config)
        phase = dom_phase(frame, self.config)
        signature = str(frame.get("signature") or "")
        observed_at = self.steps
        self._state = frame
        self.preview = "data:image/png;base64," + base64.b64encode(self.host.screenshot()).decode()
        text = str(frame.get("text") or "")[:22000]
        # Track how the rendered frame evolves. Canvas games publish no text, so
        # without this the observation is byte-identical between steps and the
        # model cannot tell a dead action from a slow one.
        pixel = self._pixel_signature()
        changed = bool(pixel) and pixel != getattr(self, "_last_pixel", "")
        if pixel and changed:
            self._distinct_views = getattr(self, "_distinct_views", 0) + 1
        if pixel:
            self._last_pixel = pixel
            self._frames_seen = getattr(self, "_frames_seen", 0) + 1
        view = {
            "signature": pixel,
            "changed_since_last_step": changed if pixel else None,
            "distinct_views": getattr(self, "_distinct_views", 0),
            "frames_observed": getattr(self, "_frames_seen", 0),
            "change_ratio": round(
                getattr(self, "_distinct_views", 0) / max(1, getattr(self, "_frames_seen", 0)), 3
            ),
        }
        state = {
            "game": self.game_id,
            "phase": phase,
            "screen": text,
            "headings": list(frame.get("headings") or []),
            "public_fields": dict(frame.get("fields") or {}),
            "view": view,
            "timing_contract": timing_contract(self.config),
            "last_execution": self.last_execution,
            "progress": self._progress_record(),
            # The slow model is the only layer that can reason across the whole
            # catalogue, so it is shown the complete input universe rather than
            # just the tier that happens to be live. Without it the pruning task
            # would be "which of these twelve keys is useless", which cannot
            # discover that the game wants Q or a digit. Only the ids are listed:
            # the labels are already attached to every legal action below, and
            # repeating 44 sentences would crowd the screenshot reasoning out.
            "input_universe": [key for key, _label, _keys in self._universe_keys()],
            "input_excluded": sorted(self.excluded),
            "input_excluded_note": self.excluded_note,
            "input_subset_history": self.subset_history[-2:],
            "instructions": (
                "这是没有专用知识包的网页游戏。只依据当前截图与可见文字行动。"
                "先用点击类候选操作页面控件（开始、菜单、按钮），再用方向键类候选控制游戏角色。"
                "如果连续按方向键/确认键后画面或指标都毫无变化，说明这款游戏要的不是方向键："
                "音游、钢琴类请改用 key_* 字母键（如 D/F/J/K，或 Q~P 一整排）；"
                "卡牌、棋牌、选项菜单请改用 digit_* 数字键；"
                "塔防、卡牌、点选类请改用 click_* 点击画面里的格子、卡牌、按钮或敌人所在区域。"
                "每次只选一个候选，执行后重新观察，不要凭想象假设按键已被接受。"
            ),
        }
        actions: list[Action] = [] if done else self._actions(frame, signature)
        for action in actions:
            action.params["observed_at"] = observed_at
        return Observation(
            actions=actions,
            image=self.preview,
            state=state,
            level=1,
            done=done,
            success=success,
            game=str(self.game_id),
            raw={"signature": signature, "artifacts": str(self.artifacts), "observed_at": observed_at},
        )

    # -- action ------------------------------------------------------------
    def _menu_watchdog(self) -> None:
        """Force a start attempt when the game is visibly stuck before it begins.

        A large share of the catalogue opens on a title screen that ignores the
        arrow keys the model prefers, so an episode could burn its whole budget
        pressing left/right while `distinct_views` never left 2. Rather than let
        the model rediscover the start gesture over dozens of steps, this page-
        level nudge clicks the canvas centre and then tries the usual confirm
        keys. It only fires while the run shows no progress and no view change,
        so it cannot interfere with a game that is already being played.
        """
        stuck = getattr(self, "_stuck_steps", 0)
        stalled = getattr(self, "_no_progress_steps", 0)
        if stuck < self.MENU_STUCK_LIMIT and stalled < self.MENU_STUCK_LIMIT * 2:
            return
        # Only before the game has begun: once any metric has advanced, the run
        # is genuinely in play and a stray Enter could pause it.
        if isinstance(self.progress_best, (int, float)) and self.progress_best > 0:
            return
        self._stuck_steps = 0
        self._no_progress_steps = 0
        # There is no mouse-move helper on the host, and a plain click(selector)
        # needs a selector we cannot know for every game. Synthesising the event
        # on the canvas centre is what most title screens listen for, and it is
        # harmless on pages that ignore it.
        try:
            self.host.frame.evaluate(
                """() => {
                  var c = document.querySelector('canvas') ||
                          document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2);
                  if (!c) { return false; }
                  var r = c.getBoundingClientRect ? c.getBoundingClientRect() : {left:0, top:0, width:0, height:0};
                  var x = (r.left || 0) + (r.width || window.innerWidth) / 2;
                  var y = (r.top || 0) + (r.height || window.innerHeight) / 2;
                  var opts = {bubbles: true, cancelable: true, clientX: x, clientY: y, view: window};
                  ['mousedown', 'mouseup', 'click'].forEach(function (t) {
                    try { c.dispatchEvent(new MouseEvent(t, opts)); } catch (e) {}
                  });
                  try { if (c.focus) { c.focus(); } } catch (e) {}
                  return true;
                }"""
            )
        except Exception:
            pass
        # Canvas-centre clicks only work on title screens that listen for a click
        # anywhere. Games that open on a real menu (a champion grid, a character
        # select, a neon lobby) need a different gesture: click the element whose
        # own text is the start verb. Matching only `button`/`.btn` missed pages
        # that put the affordance on a div, and a single click is not enough for
        # two-stage lobbies, so this picks the innermost matching element and
        # clicks it repeatedly (each call is one nudge, the watchdog fires again).
        try:
            self.host.frame.evaluate(
                """() => {
                  var words = /^(?:立即|马上|开始|进入|继续|新的|新|开始游戏|进入战场|进入游戏|进入峡谷|立即进入战场|start|play|begin|continue|new game|single player)$/i;
                  var soft = /开始|进入|start|play|begin|continue/i;
                  var all = Array.prototype.slice.call(document.querySelectorAll('*'));
                  var hits = all.filter(function (el) {
                    if (el.children.length > 2) return false;
                    var t = String((el.innerText || el.textContent || '')).trim();
                    if (!t || t.length > 18) return false;
                    if (!(words.test(t) || soft.test(t))) return false;
                    var s = getComputedStyle(el);
                    if (s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') return false;
                    if (el.disabled) return false;
                    var r = el.getBoundingClientRect();
                    return r.width > 8 && r.height > 8;
                  });
                  if (!hits.length) return 'no-start';
                  var el = hits[hits.length - 1];
                  ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'].forEach(function (t) {
                    try { el.dispatchEvent(new MouseEvent(t, {bubbles: true, cancelable: true, view: window})); } catch (e) {}
                  });
                  try { el.click(); } catch (e) {}
                  return 'clicked:' + String((el.innerText || '').trim()).slice(0, 12);
                }"""
            )
        except Exception:
            pass
        for keys in (["Enter"], ["Space"], ["Space"]):
            try:
                self.host.keys(keys, self.press_frames)
            except Exception:
                pass

    def act(self, action: Action) -> None:
        before = self._read()
        if action.params.get("signature") != str(before.get("signature") or ""):
            self.last_execution = {"action": action.id, "ok": False, "error": "stale observation; reobserve"}
            self.executions.append(dict(self.last_execution))
            return
        before_pixel = self._pixel_signature()
        try:
            url_before = self.host.frame.url
        except Exception:
            url_before = ""
        try:
            if action.kind == "dom_click":
                self.host.click(action.params["selector"], action.params.get("nth"))
                self.host.advance(int(action.params.get("frames") or 1))
            elif action.kind == "canvas_click":
                self.host.click_at(float(action.params.get("x") or 0.5),
                                   float(action.params.get("y") or 0.5),
                                   int(action.params.get("frames") or 1))
            elif action.kind in {"web_scroll", "scroll"}:
                self.host.scroll(action.params["selector"], action.params.get("direction", "down"))
            elif action.kind in {"web_keys", "key", "tap", "hold"}:
                keys = [str(key) for key in action.keys]
                frames = int(action.params.get("frames") or self.press_frames)
                self.host.keys(keys, frames)
            elif action.kind == "wait":
                self.host.advance(int(action.params.get("frames") or 24))
            else:
                raise ValueError(f"unsupported generic action kind: {action.kind}")
            ok, error = True, ""
        except Exception as exc:  # page removed the node, focus loss, ...
            ok, error = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        # Some games bind a key to `location.reload()`: mkjs restarts the match on
        # Space. Pressing it wipes the injected probe and every point scored so
        # far, so the episode keeps replaying its opening seconds and the metric
        # never leaves zero. Detect the navigation instead of trusting the action
        # to have been harmless, and drop that key from the action set.
        navigated = "Execution context was destroyed" in error or "navigation" in error.lower()
        if navigated:
            # Only a return to the very same document is a restart. Landing on a
            # new page is how many games leave their title screen, and banning
            # that key would lock the agent out of the only way forward.
            try:
                navigated = bool(url_before) and _page_key(self.host.frame.url) == _page_key(url_before)
            except Exception:
                navigated = False
        after = self._read()
        after_pixel = self._pixel_signature()
        self.steps += 1
        # A canvas game exposes no DOM text, so both signatures are equal and
        # changed would read false even when the screen visibly moved. Reporting
        # the pixel delta is what lets the model tell "my action did nothing"
        # apart from "my action was slow", which is the feedback it needs to link
        # cause and effect across steps.
        self.last_execution = {
            "action": action.id,
            "label": action.label,
            "ok": ok,
            "error": error,
            "before": before_pixel,
            "after": after_pixel,
            "changed": bool(before_pixel) and before_pixel != after_pixel,
            "dom_changed": str(before.get("signature") or "") != str(after.get("signature") or ""),
            "phase_after": dom_phase(after, self.config),
            "navigated": navigated,
        }
        self.executions.append(dict(self.last_execution))
        if navigated:
            # Keep the key out of the next action set: it restarts the game and
            # throws away the score accumulated so far, so every later step would
            # be spent replaying the opening instead of making progress.
            left = set(action.keys or ())
            ids = {key for key, _label, keys in self._key_table()
                   if left and set(keys) & left}
            ids.add(action.id)
            self.excluded |= ids
        # Count steps that produced no new visible frame. The counter only grows
        # while the picture is unchanged, so a game that is genuinely running
        # keeps it at zero and the start nudge never fires.
        if self.last_execution.get("changed"):
            self._stuck_steps = 0
        else:
            self._stuck_steps = getattr(self, "_stuck_steps", 0) + 1
        # Pixel movement alone is a poor stuck signal: a looping title animation
        # repaints every frame and reports "changed" forever while the game never
        # starts. The metric is the honest witness, so it drives a second counter
        # that only resets when the score-bearing value actually advances.
        best_now = self.progress_best
        if isinstance(best_now, (int, float)) and best_now > 0:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps = getattr(self, "_no_progress_steps", 0) + 1
        self._menu_watchdog()
        # Sample immediately after the step: the game clock keeps running, so a
        # peak reached by this action would otherwise be missed until the next
        # observation and the episode peak would read low.
        self._progress_record()
