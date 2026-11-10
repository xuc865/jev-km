"""Distillation records for the two-role design.

The previous writer wrapped the fast scorer's ``score_fn`` and monkey-patched the
planner's ``__call__`` to capture whatever they happened to send. That worked
while the two tiers used different endpoints, and it broke down as soon as both
tiers share one client: the wrapper no longer sits between the caller and the
network, so it cannot see the request.

So recording moves to the place that actually knows: ``ModelClient`` keeps every
``ModelCall`` in wire form, and this module turns those into records.

Three principles, each of which was a bug first:

1. The record contains the exact message list that was sent, plus the hash of the
   frame that was attached. A student trained on it sees what a live run saw, so
   the prompt can be changed later without silently invalidating the dataset.

2. A call that arrived but failed contract validation is kept, tagged
   ``proven: false``. An earlier version dropped it to keep the dataset clean,
   which turned out to be the wrong trade: a rejected reply is the most
   informative sample about where the prompt fails, and dropping it hides the
   failure rate instead of recording it.

3. Steps taken before the game publishes a metric are kept too, for the same
   reason the user gave for wanting whole trajectories: a filtered dataset can be
   derived from a complete one, never the reverse.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from .client import ModelCall

ROOT = Path(__file__).resolve().parent.parent
SFT_ROOT = Path(os.environ.get("JEVGAMER_SFT_DIR") or (ROOT / "runs" / "sft"))


def save_frames(folder: Path, data_url: str, seen: set[str]) -> str:
    """Persist one frame as ``frames/<sha>.png`` and return its hash.

    Frames are deduplicated by content because consecutive steps of a menu screen
    are byte-identical, and storing each one would make the images outweigh every
    label in the dataset. The hash is computed by the client for the call record,
    so this only writes when a new picture appears.
    """
    if not data_url:
        return ""
    header, _, body = data_url.partition("base64,")
    if not body:
        return ""
    import hashlib

    digest = hashlib.sha1(body.encode("ascii", "ignore")).hexdigest()[:16]
    if digest in seen:
        return digest
    suffix = ".png" if "image/png" in header else ".jpg"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        (folder / (digest + suffix)).write_bytes(base64.b64decode(body))
    except Exception:
        return digest
    seen.add(digest)
    return digest


class RoleRecorder:
    """Turns ModelClient calls into role-specific training records.

    Kept deliberately thin: parsing and validation already happened in the
    policies, so this class only attaches the episode context and decides whether
    a row is *provable* rather than whether it is *worth keeping*.

    ``label_for`` is an optional override for callers that need to supply a label
    after the fact. The default path reads ``call.label``, which the policies fill
    in at the moment they accept a reply -- that is the only point where "this
    reply passed validation" is known to be true.
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
        self.frame_dir = self.folder / "frames"
        self.label_for = label_for
        self.counts = {"executor": 0, "planner": 0}
        self.failed = 0
        self.invalid = 0
        self.unproven = 0
        self.reasons: dict[str, int] = {}
        self.progress: dict = {}
        self.step = 0
        self.plan_revision = 0
        self.trigger = ""

    def set_progress(self, record: dict | None) -> None:
        self.progress = dict(record) if isinstance(record, dict) else {}

    def _progress_known(self) -> bool:
        value = (self.progress or {}).get("value")
        return isinstance(value, (int, float))

    def record(self, call: ModelCall, label: dict | None = None) -> bool:
        """Write one record; return whether it carries a usable label.

        The row is written either way. ``proven`` records whether the game had
        published a numeric metric at this step; a planner revision issued on the
        title screen legitimately has neither.
        """
        if call.error:
            # A transport failure has no reply at all. It is still written, with
            # an empty label, so a run can be audited for how much of its budget
            # was spent on timeouts rather than on the game.
            self.failed += 1
            self._write(call, {}, proven=False, error=call.error)
            return False
        resolved = label if label is not None else (call.label or None)
        if not resolved and self.label_for is not None:
            resolved = self.label_for(call)
        proven = self._progress_known()
        if not resolved:
            # A reply that arrived but failed contract validation. Kept because it
            # is the sample that shows where the prompt breaks; tagged so a
            # curated dataset can exclude it. Two causes are told apart on purpose:
            # a reasoning-only reply means the model never wrote an answer (a
            # budget problem), which is a different fix from a malformed answer (a
            # contract problem).
            self.invalid += 1
            reason = "reasoning only" if call.from_reasoning else "label rejected"
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            self._write(call, {}, proven=proven, error=reason)
            return False
        self._write(call, resolved, proven=proven, error="")
        self.counts[call.role] = self.counts.get(call.role, 0) + 1
        if not proven:
            self.unproven += 1
        return True

    def _write(self, call: ModelCall, label: dict, proven: bool, error: str) -> None:
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
            "attempts": call.attempts,
            "messages": call.messages,
            "label": label,
            "raw_reply": call.text,
            "frame_sha": call.frame_sha,
            "progress": self.progress,
            # `proven` is about the *game*, not the reply: it says a metric was
            # readable this step, so the label can be scored against something.
            "proven": proven,
            # `raw_reply` is whatever text the call produced; this flag says
            # whether that text was the model's answer or a reasoning draft that
            # happened to be returned because `content` came back empty. Without
            # it, a draft looks like a malformed answer and the prompt gets blamed
            # for a budget problem.
            "from_reasoning": bool(getattr(call, "from_reasoning", False)),
            "error": error,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

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
            "invalid": self.invalid,
            "reasons": dict(self.reasons),
            "unproven": self.unproven,
            "progress": self.progress or {},
        }
