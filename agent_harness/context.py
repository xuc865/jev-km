"""Game-agent context pressure + prune + LLM condense.

In-tree mechanisms:
- single-source soft/hard input budgets vs condense threshold
- non-destructive CondenseStore APPLY
- MAP prune for sticky identical turns
- REPLACE LLM summary for older turns when pressure remains
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .condense_store import CondenseStore, SegmentKind
from .trace import TraceStore

CHARS_PER_TOKEN = 3.5
DEFAULT_TOKEN_ESTIMATE_SCALE = 1.3
DEFAULT_CONTEXT_WINDOW = 128000
DEFAULT_MAX_OUTPUT_TOKENS = 16000
DEFAULT_CONDENSE_THRESHOLD = 70
LLM_INPUT_BUDGET_SOFT_RATIO = 0.88
LLM_INPUT_BUDGET_HARD_RATIO = 0.96
LLM_INPUT_BUDGET_SOFT_RESERVE = 4096
LLM_INPUT_BUDGET_HARD_RESERVE = 1024
LLM_INPUT_BUDGET_FLOOR = 4096
COMPRESS_TRIGGER_CEILING_RATIO = 0.95


def llm_input_budget(context_window: int, max_output_tokens: int, *, hard: bool = False) -> int:
    limit = int(context_window or 0) or DEFAULT_CONTEXT_WINDOW
    max_out = int(max_output_tokens or 0) or DEFAULT_MAX_OUTPUT_TOKENS
    reserve = max_out + (LLM_INPUT_BUDGET_HARD_RESERVE if hard else LLM_INPUT_BUDGET_SOFT_RESERVE)
    ratio = LLM_INPUT_BUDGET_HARD_RATIO if hard else LLM_INPUT_BUDGET_SOFT_RATIO
    raw = min(max(0, limit - reserve), int(limit * ratio))
    # vibe uses a 4k floor for production windows; never raise the floor above
    # the window itself (breaks tiny/test budgets and blocks compress forever).
    floor = LLM_INPUT_BUDGET_FLOOR if limit >= LLM_INPUT_BUDGET_FLOOR * 2 else max(256, limit // 8)
    return max(floor, raw) if raw > 0 else floor


def estimate_tokens(text: str, *, scale: float = DEFAULT_TOKEN_ESTIMATE_SCALE) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    return max(1, int((len(raw) / CHARS_PER_TOKEN) * max(1.0, float(scale))))


def estimate_messages_tokens(messages: list[dict[str, Any]], *, scale: float = DEFAULT_TOKEN_ESTIMATE_SCALE) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content, scale=scale)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(str(part.get("text") or ""), scale=scale)
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    # Current-turn images only; bill a conservative visual chunk.
                    total += 800
        else:
            total += estimate_tokens(json.dumps(content, ensure_ascii=False)[:2000], scale=scale)
    return total


@dataclass
class ContextConfig:
    auto_condense: bool = True
    condense_threshold_percent: int = DEFAULT_CONDENSE_THRESHOLD
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    protect_last_n_turns: int = 4
    sticky_map_streak: int = 3
    min_turns_before_llm_condense: int = 8

    def soft_budget(self) -> int:
        return llm_input_budget(self.context_window, self.max_output_tokens, hard=False)

    def hard_budget(self) -> int:
        return llm_input_budget(self.context_window, self.max_output_tokens, hard=True)

    def threshold_tokens(self) -> int:
        soft = self.soft_budget()
        raw = int(self.context_window * max(1, min(100, self.condense_threshold_percent)) / 100)
        ceiling = int(soft * COMPRESS_TRIGGER_CEILING_RATIO)
        # Keep routine compress reachable before soft overflow (vibe invariant).
        floor = min(LLM_INPUT_BUDGET_FLOOR, max(256, soft // 4))
        return max(floor, min(raw, ceiling))


LLMCaller = Callable[[list[dict[str, Any]]], str]


class ContextEngine:
    """Owns pressure checks and compress ops for one Thread."""

    def __init__(
        self,
        store: CondenseStore | None = None,
        config: ContextConfig | None = None,
        trace: TraceStore | None = None,
        llm: LLMCaller | None = None,
    ) -> None:
        self.store = store or CondenseStore()
        self.config = config or ContextConfig()
        self.trace = trace or TraceStore()
        self.llm = llm
        self.store.set_trace_emitter(self._on_store_event)

    def _on_store_event(self, event: str, segment, extra: dict) -> None:
        self.trace.emit(
            "context_condensed" if event == "add" else "context_segment",
            {
                "event": event,
                "condense_id": segment.condense_id,
                "kind": segment.kind.value,
                "token_before": segment.token_before,
                "token_after": segment.token_after,
                "lineage_size": len(segment.raw_lineage),
                **(extra or {}),
            },
        )

    def effective(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.store.construct_effective_messages(messages)

    def pressure(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        effective = self.effective(messages)
        tokens = estimate_messages_tokens(effective)
        thr = self.config.threshold_tokens()
        soft = self.config.soft_budget()
        hard = self.config.hard_budget()
        return {
            "tokens": tokens,
            "threshold": thr,
            "soft_budget": soft,
            "hard_budget": hard,
            "should_compress": bool(self.config.auto_condense and tokens >= thr),
            "overflow_soft": tokens >= soft,
            "overflow_hard": tokens >= hard,
            "effective_messages": len(effective),
            "raw_messages": len(messages),
            "segments": len(self.store.segments),
        }

    def maybe_compress(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run MAP sticky prune, then optional LLM REPLACE. Returns pressure after."""
        report: dict[str, Any] = {"actions": []}
        if not self.config.auto_condense:
            report["pressure"] = self.pressure(messages)
            return report

        before = self.pressure(messages)
        if not before["should_compress"] and not before["overflow_soft"]:
            report["pressure"] = before
            return report

        mapped = self._map_sticky_prune(messages)
        if mapped:
            report["actions"].append(mapped)

        mid = self.pressure(messages)
        if mid["should_compress"] or mid["overflow_soft"]:
            replaced = self._llm_replace(messages)
            if replaced:
                report["actions"].append(replaced)
            elif mid["overflow_soft"]:
                # Deterministic fallback when LLM unavailable: hard REPLACE truncate.
                fallback = self._fallback_replace(messages)
                if fallback:
                    report["actions"].append(fallback)
                    self.trace.emit("context_truncated", fallback)

        report["pressure"] = self.pressure(messages)
        return report

    def _turn_groups(self, messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group user/assistant(/user tool) triples; keep system alone."""
        groups: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "system":
                groups.append([msg])
                i += 1
                continue
            chunk = [msg]
            i += 1
            while i < len(messages) and messages[i].get("role") != "user":
                chunk.append(messages[i])
                i += 1
            # Include following tool-result user that starts with "# Tool result"
            if i < len(messages):
                nxt = messages[i]
                text = nxt.get("content") if isinstance(nxt.get("content"), str) else ""
                if nxt.get("role") == "user" and str(text).startswith("# Tool result"):
                    chunk.append(nxt)
                    i += 1
            groups.append(chunk)
        return groups

    def _sig(self, group: list[dict[str, Any]]) -> str:
        blob = []
        for msg in group:
            c = msg.get("content")
            if isinstance(c, str):
                blob.append(c[:280])
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        blob.append(str(part.get("text") or "")[:180])
        text = " | ".join(blob)
        # Focus on guide/choice markers when present.
        guide = re.search(r'"guide":\s*"([^"]+)"', text)
        choice = re.search(r'"choice":\s*"([^"]+)"', text)
        pressed = re.search(r'"pressed":\s*"([^"]+)"', text)
        return f"{guide.group(1) if guide else ''}|{choice.group(1) if choice else ''}|{pressed.group(1) if pressed else ''}|{text[:120]}"

    def _map_sticky_prune(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        """MAP-compress middle messages of a sticky identical streak (keep first+last)."""
        protect = max(1, self.config.protect_last_n_turns)
        streak_need = max(3, self.config.sticky_map_streak)
        groups = self._turn_groups(messages)
        # Ignore trailing protected groups + leading system.
        work = groups[:-protect] if len(groups) > protect else []
        if len(work) < streak_need:
            return None

        # Find longest trailing sticky run inside work (excluding system-only).
        run: list[list[dict[str, Any]]] = []
        run_sig = ""
        best: list[list[dict[str, Any]]] = []
        for group in work:
            if group and group[0].get("role") == "system":
                continue
            sig = self._sig(group)
            if not run or sig == run_sig:
                run.append(group)
                run_sig = sig
            else:
                if len(run) > len(best):
                    best = run
                run = [group]
                run_sig = sig
        if len(run) > len(best):
            best = run
        if len(best) < streak_need:
            return None

        # Keep first and last of sticky run; MAP-shrink middle msgs to stubs (1:1 uuid).
        middle = best[1:-1]
        if not middle:
            return None
        ordered_msgs = [m for g in middle for m in g if m.get("_msg_uuid")]
        if len(ordered_msgs) < 2:
            return None
        lineage: set[str] = set()
        condensed: list[dict[str, Any]] = []
        before = 0
        after = 0
        sig = self._sig(best[0])
        for msg in ordered_msgs:
            uid = str(msg["_msg_uuid"])
            if uid in lineage:
                continue
            lineage.add(uid)
            text = msg.get("content") if isinstance(msg.get("content"), str) else ""
            before += estimate_tokens(str(text))
            stub_text = f"[sticky-pruned] {sig[:160]}"
            after += estimate_tokens(stub_text)
            condensed.append({
                "role": msg.get("role") or "user",
                "content": stub_text,
                "_msg_uuid": uid,
            })
        if len(condensed) != len(lineage) or not lineage:
            return None
        seg = self.store.add_segment(
            condensed_messages=condensed,
            raw_lineage=lineage,
            kind=SegmentKind.MAP,
            token_before=before,
            token_after=after,
        )
        return {"op": "map_sticky", "condense_id": seg.condense_id, "n": len(lineage)}

    def _collect_replace_targets(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        protect = max(1, self.config.protect_last_n_turns)
        groups = self._turn_groups(messages)
        body = [g for g in groups if not (len(g) == 1 and g[0].get("role") == "system")]
        if len(body) <= protect + 2:
            return []
        old = body[: max(0, len(body) - protect)]
        flat: list[dict[str, Any]] = []
        for group in old:
            flat.extend(group)
        return [m for m in flat if m.get("_msg_uuid") and not m.get("_condense_parent")]

    def _llm_replace(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.llm is None:
            return None
        targets = self._collect_replace_targets(messages)
        if len(targets) < self.config.min_turns_before_llm_condense:
            return None
        lineage = {str(m["_msg_uuid"]) for m in targets}
        before = estimate_messages_tokens(targets)
        digest_lines = []
        for msg in targets:
            c = msg.get("content")
            if isinstance(c, str):
                digest_lines.append(f"{msg.get('role')}: {c[:500]}")
            else:
                digest_lines.append(f"{msg.get('role')}: (multimodal/omitted)")
        prompt = (
            "Summarize these prior game-agent turns for future decisions. "
            "Keep: goals, level/score progress, stuck patterns, forbidden repeats, "
            "useful opens/guide outcomes. Drop chatter. Reply plain text under 900 chars.\n\n"
            + "\n".join(digest_lines[-80:])
        )
        try:
            summary = (self.llm([{"role": "user", "content": prompt}]) or "").strip()
        except Exception as exc:
            self.trace.emit("context_condense_error", {"error": str(exc)[:240]})
            return None
        if not summary:
            return None
        summary = summary[:1200]
        condensed = [{
            "role": "user",
            "content": f"# Condensed prior turns\n{summary}",
        }]
        after = estimate_tokens(condensed[0]["content"])
        # Replace any prior REPLACE segments fully covered by this lineage.
        replaces = tuple(
            s.condense_id
            for s in self.store.segments
            if s.kind == SegmentKind.REPLACE and s.raw_lineage.issubset(lineage)
        )
        merged = set(lineage)
        for s in self.store.segments:
            if s.condense_id in replaces:
                merged |= set(s.raw_lineage)
        seg = self.store.add_segment(
            condensed_messages=condensed,
            raw_lineage=merged,
            kind=SegmentKind.REPLACE,
            replaces=replaces,
            token_before=before,
            token_after=after,
            ui_start_idx=0,
            ui_end_idx=max(0, len(targets) - 1),
        )
        return {"op": "llm_replace", "condense_id": seg.condense_id, "chars": len(summary)}

    def _fallback_replace(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        targets = self._collect_replace_targets(messages)
        if len(targets) < 4:
            return None
        lineage = {str(m["_msg_uuid"]) for m in targets}
        guides = []
        for msg in targets:
            c = msg.get("content") if isinstance(msg.get("content"), str) else ""
            m = re.search(r'"guide":\s*"([^"]+)"', c)
            if m:
                guides.append(m.group(1))
        summary = (
            f"Truncated {len(targets)} older turns under soft budget. "
            f"Recent guides sample: {','.join(guides[-12:]) or 'n/a'}."
        )
        condensed = [{"role": "user", "content": f"# Condensed prior turns\n{summary}"}]
        replaces = tuple(
            s.condense_id
            for s in self.store.segments
            if s.kind == SegmentKind.REPLACE and s.raw_lineage.issubset(lineage)
        )
        merged = set(lineage)
        for s in self.store.segments:
            if s.condense_id in replaces:
                merged |= set(s.raw_lineage)
        before = estimate_messages_tokens(targets)
        after = estimate_tokens(summary)
        seg = self.store.add_segment(
            condensed_messages=condensed,
            raw_lineage=merged,
            kind=SegmentKind.REPLACE,
            replaces=replaces,
            token_before=before,
            token_after=after,
        )
        return {"op": "fallback_replace", "condense_id": seg.condense_id, "n": len(targets)}
