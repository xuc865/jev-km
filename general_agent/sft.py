"""Distillation records for the two-role design.

The previous writer wrapped the fast scorer's ``score_fn`` and monkey-patched the
planner's ``__call__`` to capture whatever they happened to send. That worked
while the two tiers used different endpoints, and it broke down as soon as both
tiers share one client: the wrapper no longer sits between the caller and the
network, so it cannot see the request.

So recording moves to the place that actually knows: ``ModelClient`` keeps every
``ModelCall`` in wire form, and this module turns those into records. Two
consequences:

1. The record contains the exact message list that was sent. A student trained on
   it sees the same tokens as a live run, which is the only way the prompt can be
   changed later without silently invalidating the dataset.
2. The label is the parsed contract object, not a probability table. For the
   executor that is one action id plus its evidence; for the planner it is the
   plan delta. This matches what the two roles are now asked to produce, and it
   means a bad label is detectable (unparseable) rather than merely low-scoring.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from .client import ModelCall

ROOT = Path(__file__).resolve().parent.parent
SFT_ROOT = Path(os.environ.get("JEVGAMER_SFT_DIR") or (ROOT / "runs" / "sft"))


class RoleRecorder:
    """Turns ModelClient calls into role-specific training records.

    Kept deliberately thin: parsing and validation already happened in the
    policies, so this class only decides whether a call is worth recording and
    attaches the episode context. Any logic beyond that would duplicate the
    policies and drift from them.

    ``label_for`` is an optional override for callers that need to supply a label
    after the fact. The default path reads ``call.label``, which the policies fill
    in at the moment they accept a reply -- that is the only point where "this
    reply passed validation" is known to be true, so re-deciding it here would let
    the writer and the runtime disagree about the same call.
    """

    def __init__(
        self,
        game: str,
        run_id: str,
        goal: str = "",
        root: Path | None = None,
        label_for: Callable[[ModelCall], dict | None] | None = None,
    ) -> None:
        self.game = game
        self.run_id = run_id
        self.goal = goal
        self.folder = (root or SFT_ROOT) / game / run_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "roles.jsonl"
        self.label_for = label_for
        self.counts = {"executor": 0, "planner": 0}
        self.failed = 0
        self.unproven = 0
        self.progress: dict = {}
        self.step = 0
        self.plan_revision = 0
        self.trigger = ""

    def set_progress(self, record: dict | None) -> None:
        self.progress = dict(record) if isinstance(record, dict) else {}

    def _progress_known(self) -> bool:
        """Whether the game had published a numeric metric at this step.

        Steps taken before the game reports anything (title screen, loading) have
        nothing to attach an action to. They are counted separately rather than
        written, because an unverifiable row is worse than a missing one: it looks
        like supervision without being supervision.
        """
        value = (self.progress or {}).get("value")
        return isinstance(value, (int, float))

    def record(self, call: ModelCall, label: dict | None = None) -> bool:
        """Write one record; return whether it was written."""
        if call.error:
            # The call never produced a reply. There is no target to learn from,
            # and a record whose label is "error" teaches nothing.
            self.failed += 1
            return False
        resolved = label if label is not None else (call.label or None)
        if not resolved and self.label_for is not None:
            resolved = self.label_for(call)
        if not resolved:
            # A reply that arrived but failed contract validation is a wrong
            # answer, not a missing one. Counting it apart from transport errors
            # is what makes a prompt regression visible as a validation rate.
            self.failed += 1
            return False
        if not self._progress_known():
            self.unproven += 1
            return False
        record = {
            "schema": call.schema,
            "role": call.role,
            "game": self.game,
            "run_id": self.run_id,
            "goal": self.goal,
            "step": self.step,
            "plan_revision": self.plan_revision,
            "trigger": self.trigger,
            "model": call.model,
            "at": round(time.time(), 3),
            "ms": call.ms,
            "usage": call.usage,
            "messages": call.messages,
            "label": resolved,
            "raw_reply": call.text,
            "progress": self.progress,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.counts[call.role] = self.counts.get(call.role, 0) + 1
        return True

    def drain(self, client: Any) -> int:
        """Record every call the client has made since the last drain."""
        pending = client.drain_calls()
        return sum(1 for call in pending if self.record(call))

    def summary(self) -> dict:
        return {
            "game": self.game,
            "run_id": self.run_id,
            "folder": str(self.folder),
            "executor_records": self.counts.get("executor", 0),
            "planner_records": self.counts.get("planner", 0),
            "failed": self.failed,
            "unproven": self.unproven,
            "progress": self.progress or {},
        }
