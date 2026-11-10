"""Game knowledge packs.

The harness loads one pack by game name. Topics and experience facts are
layered by priority: core > high > normal > low. Core topics always enter the
prompt; other topics load on demand and stay for the rest of the task.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from experience import Memory, priority_rank

ROOT = Path(__file__).resolve().parent / "knowledge"
COMMON = ROOT / "common"

LAYER_LABEL = {0: "core", 1: "high", 2: "normal", 3: "low"}
LEVEL_NUMERALS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

# Shared playbooks. Game packs opt in with `"common": ["maze"]` and only keep
# title-specific topics (window/HUD/sprites). Do not copy these into each pack.
# Progressive load: only a thin always/core seed enters the prompt.
# Everything else stays catalog-only until the observer sets load=<id>.
COMMON_PRESETS: dict[str, list[dict]] = {
    "maze": [
        {
            "id": "explore",
            "summary": "迷宫通用：少回头、优先未探索",
            "priority": "core",
            "always": True,
        },
        {
            "id": "right-hand",
            "summary": "迷宫右手贴墙：右→直→左→死路才回头",
            "priority": "high",
            "always": False,
        },
    ],
    "drive": [
        {
            "id": "drive",
            "summary": "驾驶通用：保持加速、回正车道、少对晃",
            "priority": "high",
            "always": False,
        },
    ],
    "timing": [
        {
            "id": "timing",
            "summary": "时机通用：默认等待，对准再点一下",
            "priority": "core",
            "always": True,
        },
    ],
}


# Packs describe geometry. The controller compiles observations; this brief is a schema.


class Game:
    def __init__(self, data: dict, bodies: dict[str, str]) -> None:
        self.id = str(data["id"])
        self.window = str(data["window"])
        self.window_class = str(data.get("window_class") or "")
        self.actions = list(data["actions"])
        self.prompts = list(data.get("prompts") or data.get("extra") or [])
        self.extra = self.prompts
        self.topic_index = list(data.get("topics") or [])
        self.topic_body = bodies
        self.checks = {str(item["id"]): item for item in (data.get("checks") or [])}
        self.examples = list(data.get("examples") or [])
        self.notes = str(data.get("notes") or "")
        self.goal = dict(data.get("goal") or {})
        self.agent = dict(data.get("agent") or {})
        self.observe = dict(data.get("observe") or {})
        self.views = dict(data.get("views") or {})
        self.hud = dict(data.get("hud") or {})
        self.control = str(data.get("control") or "camera").strip().lower() or "camera"
        self.input = dict(data.get("input") or {})
        self.common = list(data.get("common") or [])
        self.common_experience = bool(data.get("common_experience"))
        self.folder = Path()
        self.memory = Memory(Path())

    def quiet_input(self) -> bool:
        """Keyboard games should not steal the user's mouse or keep focus."""
        if "quiet" in self.input:
            return bool(self.input.get("quiet"))
        return self.control == "cardinal"

    def bridge_url(self) -> str:
        """Optional same-origin HTTP axis bridge (e.g. Astray serve.py)."""
        return str(self.input.get("bridge") or "").strip()

    def feed_url(self) -> str:
        """Optional structured frame feed: per-frame legal actions + local state.

        Instrumented games opt in with ``input.feed``. Without it the harness
        keeps using the screenshot → observe → compile path.
        """
        return str(self.input.get("feed") or "").strip()

    def feed_hold(self) -> float:
        """Seconds per feed move. Short holds keep one decision ≈ one cell."""
        raw = self.input.get("feed_hold")
        try:
            return max(0.05, min(1.5, float(raw))) if raw is not None else 0.18
        except (TypeError, ValueError):
            return 0.18

    def goal_locked(self) -> bool:
        if "goal_locked" in self.agent:
            return bool(self.agent.get("goal_locked"))
        return self.control == "cardinal"

    def locked_goal(self, task: str = "") -> str:
        """The page task is the locked goal; the pack goal is only the default.

        ``goal_locked`` still stops the outer HUD/supervisor from replacing
        that goal mid-run. Without a task (e.g. a bare harness call) the pack's
        ``agent.goal`` keeps the old behavior.
        """
        requested = str(task or "").strip()
        if requested:
            return requested
        return str(self.agent.get("goal") or "").strip() or str((self.examples or [{}])[0].get("task") or "").strip()

    def target_level(self, task: str = "") -> int:
        """Level the run must reach (e.g. 通关第2关 -> Level 3).

        Only ``level_ge`` packs use this. A bare "进入第N关/到第N关" is a
        destination, not a clear target, so the example
        "通关第1关进入第2关" still ends at Level 2.
        """
        if str(self.goal.get("done_when") or "") != "level_ge":
            return 0
        default = int(self.goal.get("level") or 2)
        text = str(task or "")
        targets = [default]
        for match in re.finditer(r"第\s*([0-9]+|[一二三四五六七八九十])\s*关", text):
            raw = match.group(1)
            number = int(raw) if raw.isdigit() else LEVEL_NUMERALS.get(raw, 0)
            if not number:
                continue
            before = text[max(0, match.start() - 6):match.start()]
            if re.search(r"(进入|到|至|抵达|reach|enter)\s*$", before):
                targets.append(number)
            else:
                targets.append(number + 1)
        return max(targets)

    def locked_self(self) -> str:
        return str(self.agent.get("self") or "").strip()

    def system_lock(self, task: str = "") -> str:
        """Immutable DeepSeek contract: who I am in-frame + which goal is fixed."""
        self_text = self.locked_self() or "the controllable player avatar in this game window"
        not_text = str(self.agent.get("self_not") or "").strip()
        goal_text = self.locked_goal(task)
        lines = [
            "SYSTEM LOCK — immutable. Do not reinterpret, replace, or invent around these facts.",
            f"YOU ARE (in the screenshot): {self_text}",
        ]
        if not_text:
            lines.append(f"YOU ARE NOT: {not_text}")
        lines.extend([
            f"LOCKED GOAL: {goal_text}",
            "Rules:",
            "- Every open/guide/move judgment is relative to YOU (the locked self), never to walls, HUD text, or scenery.",
            "- Do not change the locked goal, invent a side quest, or chase a different objective.",
            "- Do not role-play as the camera, the maze, an NPC, or a spectator.",
            "- If the image is unclear, say so in JSON fields; still keep the same self and goal.",
        ])
        extra = str(self.agent.get("lock") or "").strip()
        if extra:
            lines.append(extra)
        return "\n".join(lines)

    def reload_memory(self) -> None:
        folder = self.folder if str(self.folder) not in {"", "."} else self.memory.folder
        self.memory = Memory(folder)
        _apply_common_experience(self.memory, self.common, self.common_experience)
        # Keep progressive activation empty until caller syncs loaded topics.
        self.memory.sync_common(self.family(), self.always_ids())

    def sync_experience(self, loaded: list[str] | None = None) -> list[str]:
        """Activate common experience facts for family + currently loaded topics."""
        topics = self.seed_loaded(loaded)
        return self.memory.sync_common(self.family(), topics)

    def finished(self, see: dict) -> bool:
        rule = str(self.goal.get("done_when") or "")
        guide = str(see.get("guide") or "none")
        if rule in {"tracker_absent", "marker_absent"}:
            return see.get("tracker") is False and guide == "none"
        if rule == "level_ge":
            try:
                return int(see.get("level") or 0) >= int(self.goal.get("level") or 2)
            except (TypeError, ValueError):
                return False
        if rule == "score_ge":
            try:
                return int(see.get("score") or 0) >= int(self.goal.get("score") or 0)
            except (TypeError, ValueError):
                return False
        if rule == "flag":
            return bool(see.get(str(self.goal.get("flag") or "won")))
        return False

    def sense_opens(self) -> bool:
        """Maze-family wall sensor flag. Default on only when family is maze."""
        if "sense_opens" in self.input:
            return bool(self.input.get("sense_opens"))
        # Infer after family would recurse; use declared family / control only.
        raw = str(self.input.get("family") or "").strip().lower()
        if raw:
            return raw == "maze"
        return self.control == "cardinal"

    def family(self) -> str:
        """Harness behavior family: maze | drive | timing | puzzle | camera.

        All compile/observe heuristics must gate on this — never on a game id.
        """
        raw = str(self.input.get("family") or "").strip().lower()
        if raw in {"maze", "drive", "timing", "puzzle", "camera"}:
            return raw
        if self.control != "cardinal":
            return "camera"
        if "sense_opens" in self.input and bool(self.input.get("sense_opens")):
            return "maze"
        if "drive" in (self.common or []):
            return "drive"
        defaults = self.input.get("default_opens")
        if isinstance(defaults, list):
            names = {str(item).strip().lower() for item in defaults}
            if names and names <= {"up"}:
                return "timing"
            if not names:
                return "timing"
        return "puzzle"

    def role_id(self, role: str, prefer: bool = True) -> str:
        preferred = ""
        fallback = ""
        for item in self.actions:
            if str(item.get("role") or "") != role:
                continue
            if not fallback:
                fallback = str(item["id"])
            if item.get("prefer"):
                preferred = str(item["id"])
            elif prefer is False:
                return str(item["id"])
        if prefer is False:
            return fallback
        return preferred or fallback

    def bearing_action(self, guide: str) -> str:
        wanted = str(guide or "").strip().lower()
        if not wanted or wanted in {"none", "ahead"}:
            return ""
        for item in self.actions:
            if str(item.get("bearing") or "").strip().lower() == wanted:
                return str(item["id"])
        return ""

    def topic_priority(self, item: dict) -> int:
        if item.get("priority") is not None:
            return priority_rank(item.get("priority"))
        # always-on topics default above on-demand ones
        return 1 if item.get("always") else 2

    def always_ids(self) -> list[str]:
        ranked = sorted(
            (
                item
                for item in self.topic_index
                if item.get("always") and str(item["id"]) in self.topic_body
            ),
            key=lambda item: (self.topic_priority(item), str(item["id"])),
        )
        return [str(item["id"]) for item in ranked]

    def seed_loaded(self, loaded: list[str] | None = None) -> list[str]:
        """Core/always topics start loaded; keep caller order for the rest."""
        ordered: list[str] = []
        seen: set[str] = set()
        for topic_id in self.always_ids() + list(loaded or []):
            if topic_id in self.topic_body and topic_id not in seen:
                ordered.append(topic_id)
                seen.add(topic_id)
        return ordered

    def active_topics(self, loaded: list[str] | None = None) -> list[dict]:
        wanted = set(self.seed_loaded(loaded))
        items = [
            item
            for item in self.topic_index
            if str(item["id"]) in wanted and str(self.topic_body.get(str(item["id"])) or "").strip()
        ]
        return sorted(items, key=lambda item: (self.topic_priority(item), str(item["id"])))

    def knowledge_layers(self, loaded: list[str] | None = None, *, max_rank: int | None = None) -> str:
        """Render active topics as priority layers. Lower rank = higher priority."""
        lines: list[str] = []
        current: int | None = None
        for item in self.active_topics(loaded):
            rank = self.topic_priority(item)
            if max_rank is not None and rank > max_rank:
                continue
            label = LAYER_LABEL.get(rank, "normal")
            if rank != current:
                current = rank
                lines.append(f"## Knowledge [{label}]")
            topic_id = str(item["id"])
            lines.append(f"### {topic_id}")
            lines.append(str(self.topic_body.get(topic_id) or "").strip())
        return "\n".join(lines)

    def option_ids(self) -> tuple[str, ...]:
        return tuple(str(item["id"]) for item in self.actions)

    def track_of(self, choice: str) -> str:
        if choice == "done":
            return "stop"
        spec = self.action(choice)
        kind = str(((spec or {}).get("input") or {}).get("type") or "release")
        if kind == "look":
            return "look"
        if kind in {"tap", "click", "prompt"}:
            return "once"
        return "move"

    def track_ids(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for name in self.option_ids():
            grouped.setdefault(self.track_of(name), []).append(name)
        return {key: tuple(value) for key, value in grouped.items()}

    def look_delta(self, choice: str) -> tuple[int, int]:
        payload = ((self.action(choice) or {}).get("input") or {})
        return int(payload.get("dx") or 0), int(payload.get("dy") or 0)

    def opposes_look(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        ax, ay = self.look_delta(left)
        bx, by = self.look_delta(right)
        return ax * bx < 0 or ay * by < 0

    def action(self, choice: str) -> dict | None:
        for item in self.actions:
            if item["id"] == choice:
                return item
        return None

    def check_of(self, choice: str) -> dict | None:
        spec = self.action(choice)
        if not spec or not spec.get("check"):
            return None
        return self.checks.get(str(spec["check"]))

    def prompt_action_id(self) -> str:
        for item in self.actions:
            if str((item.get("input") or {}).get("type") or "") == "prompt":
                return str(item["id"])
        return ""

    def prompt_of(self, name: str) -> dict | None:
        wanted = name.strip().upper()
        for item in self.prompts:
            if str(item.get("id") or item.get("key") or "").strip().upper() == wanted:
                return item
        return None

    def default_seconds(self, choice: str) -> float:
        spec = self.action(choice)
        payload = (spec or {}).get("input") or {}
        if payload.get("seconds") is not None:
            try:
                return float(payload["seconds"])
            except (TypeError, ValueError):
                pass
        kind = str(payload.get("type") or "release")
        return {"hold": 2.0, "look": 1.2, "tap": 0.25, "click": 0.25, "prompt": 0.25, "release": 0.2}.get(kind, 0.4)

    def observer_brief(self, loaded: list[str] | None = None) -> str:
        lines = [
            f"Game: {self.id}",
            f"Self: {self.locked_self() or '(pack missing agent.self)'}",
            f"Goal: {self.locked_goal()}",
            "Actions:",
        ]
        for item in self.actions:
            bits = [item["id"], str((item.get("input") or {}).get("type") or "")]
            if item.get("bearing"):
                bits.append("bearing=" + str(item["bearing"]))
            if item.get("role"):
                bits.append("role=" + str(item["role"]))
            lines.append("- " + " ".join(part for part in bits if part))
        if self.prompts:
            lines.append("Prompt keys: " + ", ".join(str(item.get("id") or item.get("key")) for item in self.prompts))
        if self.observe:
            lines.append("Observe:")
            for key, value in self.observe.items():
                lines.append(f"- {key}: {value}")
        if self.views:
            lines.append("Read the minimap crop first. Report every icon as a raw mark. Do not assign roles.")
            for name, spec in self.views.items():
                if isinstance(spec, dict) and spec.get("read"):
                    lines.append(f"- {name}: {spec['read']}")
        learned = [fact for fact in self.memory.facts if fact.get("kind") == "symbol"]
        if learned:
            lines.append("Learned minimap marks from play:")
            for fact in learned:
                looks = fact.get("looks") or {}
                role = (fact.get("do") or fact.get("then") or {}).get("role")
                lines.append(
                    f"- {looks.get('color')} {looks.get('shape')} = {role}"
                    + (f" ({fact.get('why')})" if fact.get("why") else "")
                )
        if str(self.goal.get("done_when") or "") in {"tracker_absent", "marker_absent"}:
            lines.append("The run ends only when the minimap has no marker and no quest list is tracking this task.")
        # Progressive skills: roster is cheap; only always + loaded bodies enter.
        from agent_harness.skills import SkillCard, render_loaded_bodies, render_roster

        active = set(self.seed_loaded(loaded))
        cards: list[SkillCard] = []
        for item in sorted(self.topic_index, key=lambda row: (self.topic_priority(row), str(row["id"]))):
            topic_id = str(item["id"])
            cards.append(
                SkillCard(
                    name=topic_id,
                    summary=str(item.get("summary") or topic_id),
                    priority=LAYER_LABEL.get(self.topic_priority(item), "normal"),
                    always=bool(item.get("always")),
                    body=str(self.topic_body.get(topic_id) or ""),
                )
            )
        if cards:
            lines.append(render_roster(cards, loaded=active))
            bodies = render_loaded_bodies(cards, active)
            if bodies:
                lines.append("## Knowledge [loaded bodies]")
                lines.append(bodies)
        return "\n".join(lines)

    def manual(self, loaded: list[str]) -> str:
        # Same progressive stack as the observer — no second full dump.
        return self.observer_brief(loaded)


def game_ids() -> list[str]:
    if not ROOT.exists():
        return []
    return sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_dir() and path.name != "common" and (path / "game.json").is_file()
    )


def default_id() -> str:
    ids = game_ids()
    if not ids:
        raise RuntimeError("knowledge 目录里没有游戏包")
    return ids[0]


def _topic_body(folder: Path, topic_id: str) -> str:
    """Pack-local markdown wins; otherwise fall through to knowledge/common/."""
    for base in (folder, COMMON):
        path = base / f"{topic_id}.md"
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return ""


def _with_common_topics(data: dict) -> dict:
    topics = list(data.get("topics") or [])
    seen = {str(item.get("id") or "") for item in topics}
    injected: list[dict] = []
    for preset in data.get("common") or []:
        for item in COMMON_PRESETS.get(str(preset), []):
            topic_id = str(item["id"])
            if topic_id in seen:
                continue
            injected.append(dict(item))
            seen.add(topic_id)
    if injected:
        data = {**data, "topics": injected + topics}
    return data


def _apply_common_experience(memory: Memory, common: list, common_experience: bool = False) -> None:
    """Register shared facts into the dormant common library.

    Maze facts stay in experience.json; other presets use <preset>.experience.json.
    Activation happens later via Memory.sync_common(family, loaded_topics).
    """
    for preset in common or []:
        name = str(preset)
        if name == "maze":
            path = COMMON / "experience.json"
            family = "maze"
        else:
            path = COMMON / f"{name}.experience.json"
            family = name
        if path.is_file():
            memory.overlay_common(path, family=family)
    if common_experience:
        memory.overlay_common(COMMON / "experience.json", family="maze")


def load(game_id: str) -> Game:
    folder = ROOT / game_id
    data = _with_common_topics(json.loads((folder / "game.json").read_text(encoding="utf-8-sig")))
    bodies = {}
    for item in data.get("topics") or []:
        topic_id = str(item["id"])
        bodies[topic_id] = _topic_body(folder, topic_id)
    game = Game(data, bodies)
    game.folder = folder
    game.memory = Memory(folder)
    _apply_common_experience(game.memory, game.common, game.common_experience)
    # Seed only always/core topics' experience facts.
    game.sync_experience(game.always_ids())
    return game
