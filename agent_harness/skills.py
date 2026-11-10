"""Progressive skill roster / body loader (in-tree).

Stage 1 — announcement: only ``name + summary`` (cheap, always in prompt).
Stage 2 — skill_view / topic load: full body enters context and stays for the episode.

Game packs keep markdown under ``knowledge/<Game>/``; this module is the
loading protocol, not a second content store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class SkillCard:
    name: str
    summary: str
    priority: str = "normal"
    always: bool = False
    body: str = ""


_HEADER = (
    "## Skills (progressive)\n"
    "Scan the roster below. If the core goal matches a skill's primary scene, "
    "load it by setting JSON `load` to that exact bare name (one id). "
    "Do not prophylactically load edge matches — each body stays in context.\n"
)

_FOOTER = (
    "No match → act without loading. Same skill loaded twice in one episode "
    "returns already-loaded (no duplicate body)."
)


def render_roster(skills: Iterable[SkillCard], *, loaded: set[str] | None = None) -> str:
    loaded = loaded or set()
    lines = [_HEADER, "<available_skills>"]
    pending: list[SkillCard] = []
    active: list[SkillCard] = []
    for skill in skills:
        if skill.name in loaded or skill.always:
            active.append(skill)
        else:
            pending.append(skill)
    if active:
        lines.append("  loaded:")
        for skill in active:
            mark = "always" if skill.always else "viewed"
            lines.append(f"    - {skill.name}[{skill.priority}/{mark}]: {skill.summary}")
    if pending:
        lines.append("  pending:")
        for skill in pending:
            lines.append(f"    - {skill.name}[{skill.priority}]: {skill.summary}")
    lines.append("</available_skills>")
    lines.append(_FOOTER)
    return "\n".join(lines)


def render_loaded_bodies(skills: Iterable[SkillCard], loaded: set[str]) -> str:
    """Full bodies only for always + explicitly viewed skills."""
    blocks: list[str] = []
    for skill in skills:
        if not skill.body.strip():
            continue
        if skill.always or skill.name in loaded:
            blocks.append(f"### skill:{skill.name}\n{skill.body.strip()}")
    return "\n\n".join(blocks)


@dataclass
class SkillSession:
    """Per-episode progressive activation with dedup."""

    cards: dict[str, SkillCard] = field(default_factory=dict)
    viewed: set[str] = field(default_factory=set)
    always: set[str] = field(default_factory=set)

    def register(self, card: SkillCard) -> None:
        self.cards[card.name] = card
        if card.always:
            self.always.add(card.name)

    def roster_text(self) -> str:
        return render_roster(self.cards.values(), loaded=self.viewed | self.always)

    def bodies_text(self) -> str:
        return render_loaded_bodies(self.cards.values(), self.viewed | self.always)

    def view(self, name: str) -> dict[str, Any]:
        """Load one skill body. Returns status payload for trace/tool result."""
        key = str(name or "").strip()
        if not key or key not in self.cards:
            return {"ok": False, "error": "unknown_skill", "name": key}
        if key in self.viewed or key in self.always:
            return {"ok": True, "name": key, "already_loaded": True}
        self.viewed.add(key)
        card = self.cards[key]
        return {
            "ok": True,
            "name": key,
            "already_loaded": False,
            "chars": len(card.body or ""),
            "summary": card.summary,
        }

    def loaded_ids(self) -> list[str]:
        ordered = []
        seen = set()
        for name in list(self.always) + list(self.viewed):
            if name in self.cards and name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered
