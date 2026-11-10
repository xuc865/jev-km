"""Structured span/trace (in-tree).

Keeps an in-memory event log the supervisor/UI can project. Event types:
context_condensed / context_truncated / skill_loaded / master_invoke / step.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    id: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    parent_id: str = ""


class TraceStore:
    def __init__(self, *, limit: int = 400) -> None:
        self.limit = max(50, int(limit))
        self.spans: list[Span] = []

    def emit(self, type: str, payload: dict[str, Any] | None = None, *, parent_id: str = "") -> Span:
        span = Span(
            id=uuid.uuid4().hex[:12],
            type=str(type),
            payload=dict(payload or {}),
            parent_id=parent_id or "",
        )
        self.spans.append(span)
        if len(self.spans) > self.limit:
            self.spans = self.spans[-self.limit :]
        return span

    def recent(self, n: int = 40) -> list[dict[str, Any]]:
        rows = self.spans[-max(1, n) :]
        return [
            {
                "id": s.id,
                "type": s.type,
                "payload": dict(s.payload),
                "created_at": s.created_at,
                "parent_id": s.parent_id,
            }
            for s in rows
        ]

    def clear(self) -> None:
        self.spans.clear()
