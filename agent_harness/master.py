"""Master outer loop that invokes an inner episode (in-tree).

Master does not pick keys. It:
1. owns the task / recovery / stall review
2. invokes the inner observe→compile→act episode
3. manages context pressure on the shared Thread between episodes
4. emits master_invoke / master_review spans

The existing Supervisor keeps HUD watch; this module is the invoke contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .thread import Thread
from .trace import TraceStore


InnerRunner = Callable[[], dict[str, Any]]


@dataclass
class MasterInvokeResult:
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    compress: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class MasterRuntime:
    def __init__(self, thread: Thread, trace: TraceStore | None = None) -> None:
        self.thread = thread
        self.trace = trace or thread.trace
        self.invokes = 0
        self.last: MasterInvokeResult | None = None

    def invoke_inner(self, runner: InnerRunner, *, reason: str = "step") -> MasterInvokeResult:
        """One Master→inner episode. Inner owns keys; Master owns context after."""
        self.invokes += 1
        parent = self.trace.emit(
            "master_invoke",
            {"reason": reason, "invoke": self.invokes, "phase": "start"},
        )
        try:
            result = runner() or {}
            compress = self.thread.maybe_compress()
            out = MasterInvokeResult(ok=not bool(result.get("error")), result=result, compress=compress)
            self.trace.emit(
                "master_invoke",
                {
                    "reason": reason,
                    "invoke": self.invokes,
                    "phase": "end",
                    "ok": out.ok,
                    "pressed": result.get("pressed"),
                    "compress_actions": list((compress or {}).get("actions") or []),
                    "tokens": ((compress or {}).get("pressure") or {}).get("tokens"),
                },
                parent_id=parent.id,
            )
            self.last = out
            return out
        except Exception as exc:
            out = MasterInvokeResult(ok=False, error=str(exc)[:400])
            self.trace.emit(
                "master_invoke",
                {"reason": reason, "invoke": self.invokes, "phase": "error", "error": out.error},
                parent_id=parent.id,
            )
            self.last = out
            return out

    def review(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self.trace.emit("master_review", {"kind": kind, **(payload or {})})
