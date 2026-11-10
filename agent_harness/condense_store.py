"""Non-destructive message compression store (in-tree).

history itself is never rewritten. Each compress adds a Segment that is replayed
when building the effective LLM view:

- REPLACE: continuous raw lineage → summary message(s)
- MAP: one-to-one transform of selected raw messages

Identity truth is ``raw_lineage`` (set of ``_msg_uuid``), not list indices.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, FrozenSet, Optional


class SegmentKind(str, Enum):
    REPLACE = "replace"
    MAP = "map"


@dataclass
class CondenseSegment:
    condense_id: str
    start_idx: int
    end_idx: int
    condensed_messages: list[dict[str, Any]]
    kind: SegmentKind = SegmentKind.REPLACE
    raw_lineage: FrozenSet[str] = field(default_factory=frozenset)
    replaces: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    token_before: int = 0
    token_after: int = 0


TraceEmitter = Callable[[str, CondenseSegment, dict[str, Any]], None]


class CondenseStore:
    def __init__(self, trace_emitter: Optional[TraceEmitter] = None) -> None:
        self._segments: list[CondenseSegment] = []
        self._trace_emitter = trace_emitter
        self._version = 0

    def set_trace_emitter(self, emitter: Optional[TraceEmitter]) -> None:
        self._trace_emitter = emitter

    def _notify(self, event: str, segment: CondenseSegment, extra: dict | None = None) -> None:
        if not self._trace_emitter:
            return
        try:
            self._trace_emitter(event, segment, extra or {})
        except Exception:
            pass

    @property
    def version(self) -> int:
        return self._version

    @property
    def has_condensed(self) -> bool:
        return bool(self._segments)

    @property
    def segments(self) -> list[CondenseSegment]:
        return sorted(self._segments, key=lambda s: s.created_at)

    def clear(self) -> None:
        self._segments.clear()
        self._version += 1

    def add_segment(
        self,
        *,
        condensed_messages: list[dict[str, Any]],
        raw_lineage: FrozenSet[str] | set[str],
        kind: SegmentKind = SegmentKind.REPLACE,
        replaces: tuple[str, ...] = (),
        token_before: int = 0,
        token_after: int = 0,
        ui_start_idx: int = 0,
        ui_end_idx: int = 0,
        condense_id: str | None = None,
    ) -> CondenseSegment:
        lineage = frozenset(raw_lineage)
        if not lineage:
            raise ValueError("raw_lineage cannot be empty")
        if kind == SegmentKind.MAP and len(condensed_messages) != len(lineage):
            raise ValueError("MAP condensed_messages length must equal lineage size")

        replaces_ids: list[str] = []
        for old_id in replaces:
            old = next((s for s in self._segments if s.condense_id == old_id), None)
            if old is None:
                raise ValueError(f"replaces id missing: {old_id}")
            if not old.raw_lineage.issubset(lineage):
                raise ValueError(f"replaces lineage not subset: {old_id}")
            replaces_ids.append(old_id)
        for old_id in replaces_ids:
            self._segments = [s for s in self._segments if s.condense_id != old_id]

        cid = condense_id or uuid.uuid4().hex[:12]
        if any(s.condense_id == cid for s in self._segments):
            raise ValueError(f"condense_id collision: {cid}")

        for msg in condensed_messages:
            if isinstance(msg, Mapping):
                msg["_condense_parent"] = cid
                if not msg.get("_msg_uuid"):
                    msg["_msg_uuid"] = f"c-{cid}-{uuid.uuid4().hex[:8]}"

        seg = CondenseSegment(
            condense_id=cid,
            start_idx=ui_start_idx,
            end_idx=ui_end_idx,
            condensed_messages=list(condensed_messages),
            kind=kind,
            raw_lineage=lineage,
            replaces=tuple(replaces_ids),
            token_before=token_before,
            token_after=token_after,
        )
        self._segments.append(seg)
        self._version += 1
        self._notify("add", seg, {"replaces": list(replaces_ids)})
        return seg

    def construct_effective_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        skip_map: bool = False,
    ) -> list[dict[str, Any]]:
        view, _ = self.construct_effective_messages_with_lineage(messages, skip_map=skip_map)
        return view

    def construct_effective_messages_with_lineage(
        self,
        messages: list[dict[str, Any]],
        *,
        skip_map: bool = False,
    ) -> tuple[list[dict[str, Any]], list[FrozenSet[str]]]:
        view = [dict(m) for m in messages]
        lineages: list[FrozenSet[str]] = [
            frozenset({m["_msg_uuid"]}) if m.get("_msg_uuid") else frozenset()
            for m in view
        ]
        ordered = list(self.segments)
        if skip_map:
            ordered = [s for s in ordered if s.kind != SegmentKind.MAP]
        for seg in ordered:
            view, lineages = self._apply(view, lineages, seg)
        return view, lineages

    def _apply(
        self,
        view: list[dict[str, Any]],
        lineages: list[FrozenSet[str]],
        seg: CondenseSegment,
    ) -> tuple[list[dict[str, Any]], list[FrozenSet[str]]]:
        positions = [i for i, m in enumerate(view) if m.get("_msg_uuid") in seg.raw_lineage]
        if not positions:
            return view, lineages

        if seg.kind == SegmentKind.MAP:
            # One-to-one: replace each hit with corresponding condensed message
            # Condensed list order follows ascending position order.
            condensed = list(seg.condensed_messages)
            if len(condensed) != len(positions):
                return view, lineages
            new_view = list(view)
            new_lineages = list(lineages)
            for pos, cm in zip(positions, condensed):
                row = dict(cm)
                row["_condense_parent"] = seg.condense_id
                new_view[pos] = row
                new_lineages[pos] = frozenset({view[pos].get("_msg_uuid")}) if view[pos].get("_msg_uuid") else frozenset()
            return new_view, new_lineages

        # REPLACE: require contiguous span (with condensed-gap rescue).
        lo, hi = positions[0], positions[-1]
        expected = list(range(lo, hi + 1))
        if positions != expected:
            gaps = [i for i in expected if i not in positions]
            if not all(isinstance(view[i], Mapping) and view[i].get("_condense_parent") for i in gaps):
                return view, lineages
            merged = set(seg.raw_lineage)
            for i in expected:
                merged |= lineages[i]
            out = [dict(cm) for cm in seg.condensed_messages]
            for cm in out:
                cm["_condense_parent"] = seg.condense_id
            return (
                view[:lo] + out + view[hi + 1 :],
                lineages[:lo] + [frozenset(merged)] * len(out) + lineages[hi + 1 :],
            )

        out = [dict(cm) for cm in seg.condensed_messages]
        for cm in out:
            cm["_condense_parent"] = seg.condense_id
        return (
            view[:lo] + out + view[hi + 1 :],
            lineages[:lo] + [seg.raw_lineage] * len(out) + lineages[hi + 1 :],
        )
