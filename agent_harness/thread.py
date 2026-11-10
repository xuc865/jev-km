"""Thread / Turn / Item with in-tree context compression.

Every model sample = system lock + effective prior turns + current observation.
Compression is non-destructive (CondenseStore). Skills stay progressive.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .condense_store import CondenseStore
from .context import ContextConfig, ContextEngine, LLMCaller
from .skills import SkillCard, SkillSession
from .trace import TraceStore


@dataclass
class Item:
    kind: str  # observation | agent_message | tool_call | tool_result | skill_view
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    msg_uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Turn:
    turn_id: str
    step: int
    items: list[Item] = field(default_factory=list)
    completed: bool = False

    def add(self, kind: str, text: str, **meta: Any) -> Item:
        item = Item(kind=kind, text=text, meta=dict(meta))
        self.items.append(item)
        return item


class Thread:
    """Conversation history that must feed every DeepSeek / OpenGCUA sample."""

    def __init__(
        self,
        *,
        history_turns: int = 48,
        context: ContextConfig | None = None,
        llm: LLMCaller | None = None,
    ) -> None:
        self.history_turns = max(4, int(history_turns))
        self.turns: list[Turn] = []
        self._open: Turn | None = None
        self.trace = TraceStore()
        self.store = CondenseStore()
        self.engine = ContextEngine(
            store=self.store,
            config=context or ContextConfig(),
            trace=self.trace,
            llm=llm,
        )
        self.skills = SkillSession()
        self.last_compress: dict[str, Any] = {}

    def bind_llm(self, llm: LLMCaller | None) -> None:
        self.engine.llm = llm

    def clear(self) -> None:
        self.turns.clear()
        self._open = None
        self.store.clear()
        self.trace.clear()
        self.skills = SkillSession()
        self.last_compress = {}

    def begin_turn(self, step: int) -> Turn:
        if self._open and not self._open.completed:
            self._open.completed = True
            self.turns.append(self._open)
        self._open = Turn(turn_id=uuid.uuid4().hex[:12], step=step)
        return self._open

    def current(self) -> Turn | None:
        return self._open

    def record(self, kind: str, text: str, **meta: Any) -> Item:
        if self._open is None:
            self.begin_turn(meta.get("step") or len(self.turns) + 1)
        assert self._open is not None
        return self._open.add(kind, text, **meta)

    def complete_turn(self) -> None:
        if self._open is None:
            return
        self._open.completed = True
        self.turns.append(self._open)
        self._open = None

    def register_skill(self, card: SkillCard) -> None:
        self.skills.register(card)

    def view_skill(self, name: str) -> dict[str, Any]:
        result = self.skills.view(name)
        self.trace.emit("skill_loaded", result)
        if result.get("ok") and not result.get("already_loaded"):
            self.record("skill_view", json.dumps(result, ensure_ascii=False), skill=name)
        return result

    def _visible_turns(self) -> list[Turn]:
        done = list(self.turns[-self.history_turns :])
        if self._open is not None and self._open.items:
            done = done + [self._open]
        return done

    def history_digest(self) -> str:
        turns = self._visible_turns()
        if not turns:
            return "Thread history: (empty — first turn)."
        lines = [
            "Thread history (effective after condense). "
            "Do not reverse or ignore prior tool results without new evidence:"
        ]
        # Digest from effective chat projection when possible.
        raw = self._raw_history_messages()
        effective = self.engine.effective(raw)
        if effective:
            for msg in effective[-24:]:
                role = msg.get("role")
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                preview = content.strip().replace("\n", " ")
                if len(preview) > 220:
                    preview = preview[:217] + "..."
                lines.append(f"- [{role}] {preview}")
            pressure = self.engine.pressure(raw)
            lines.append(
                f"(context tokens~{pressure['tokens']} / thr {pressure['threshold']}; "
                f"segments={pressure['segments']})"
            )
            return "\n".join(lines)
        for turn in turns:
            lines.append(f"## turn {turn.step} id={turn.turn_id}")
            for item in turn.items:
                preview = item.text.strip().replace("\n", " ")
                if len(preview) > 240:
                    preview = preview[:237] + "..."
                lines.append(f"- [{item.kind}] {preview}")
        return "\n".join(lines)

    def _raw_history_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in self.turns[-self.history_turns :]:
            obs = next((item for item in turn.items if item.kind == "observation"), None)
            agent = next((item for item in turn.items if item.kind == "agent_message"), None)
            tool = next((item for item in turn.items if item.kind == "tool_result"), None)
            skill = next((item for item in turn.items if item.kind == "skill_view"), None)
            if obs:
                messages.append({
                    "role": "user",
                    "content": f"# Observation step {turn.step}\n{obs.text}",
                    "_msg_uuid": obs.msg_uuid,
                    "_turn": turn.step,
                })
            if agent:
                messages.append({
                    "role": "assistant",
                    "content": agent.text,
                    "_msg_uuid": agent.msg_uuid,
                    "_turn": turn.step,
                })
            if tool:
                messages.append({
                    "role": "user",
                    "content": f"# Tool result step {turn.step}\n{tool.text}",
                    "_msg_uuid": tool.msg_uuid,
                    "_turn": turn.step,
                })
            if skill:
                messages.append({
                    "role": "user",
                    "content": f"# Skill view\n{skill.text}",
                    "_msg_uuid": skill.msg_uuid,
                    "_turn": turn.step,
                })
        return messages

    def messages_for_prompt(self, system: str, current_user_content: Any) -> list[dict[str, Any]]:
        """Build the sampling request the way vibe ContextEngine does.

        Prior turns become effective (condensed) user/assistant text pairs.
        Historical images stay out; only the current observation may carry HD frames.
        """
        messages: list[dict[str, Any]] = [{
            "role": "system",
            "content": system,
            "_msg_uuid": "system",
        }]
        history = self._raw_history_messages()
        # Compress under pressure before projecting.
        self.last_compress = self.engine.maybe_compress(history)
        effective = self.engine.effective(history)
        for msg in effective:
            clean = {"role": msg.get("role") or "user", "content": msg.get("content")}
            messages.append(clean)
        messages.append({"role": "user", "content": current_user_content})
        return messages

    def maybe_compress(self) -> dict[str, Any]:
        history = self._raw_history_messages()
        self.last_compress = self.engine.maybe_compress(history)
        return self.last_compress

    def record_step(
        self,
        *,
        step: int,
        observation: dict[str, Any],
        decision: dict[str, Any],
        pressed: str,
        because: str = "",
        why: dict[str, Any] | None = None,
    ) -> None:
        """Close one observe→decide→act cycle into the thread."""
        turn = self.begin_turn(step)
        opens = observation.get("opens") or []
        obs_text = json.dumps(
            {
                "guide": observation.get("guide"),
                "opens": opens,
                "blocked": observation.get("blocked"),
                "level": observation.get("level"),
                "score": observation.get("score"),
                "won": observation.get("won"),
                "quest": observation.get("quest"),
            },
            ensure_ascii=False,
        )
        turn.add("observation", obs_text)
        choice = decision.get("choice") or "+".join(decision.get("choices") or []) or "wait"
        why = why or {}
        turn.add(
            "agent_message",
            json.dumps(
                {
                    "choice": choice,
                    "by": decision.get("by"),
                    "intent": decision.get("intent") or "",
                    "blame": why.get("blame") or decision.get("blame") or "",
                    "because": (because or why.get("summary") or "")[:320],
                },
                ensure_ascii=False,
            ),
        )
        turn.add(
            "tool_result",
            json.dumps({"pressed": pressed, "choice": choice}, ensure_ascii=False),
            pressed=pressed,
        )
        self.complete_turn()
        self.trace.emit(
            "step",
            {
                "step": step,
                "choice": choice,
                "pressed": pressed,
                "blame": why.get("blame") or decision.get("blame") or "",
            },
        )
        # Opportunistic compress after each closed turn.
        self.maybe_compress()
