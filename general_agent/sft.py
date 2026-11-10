"""Collect SFT samples from live runs.

Two recorders share one JSONL:

* FastRecorder wraps ``JevPolicy.score_fn`` and captures the exact payload sent
  to WeMM-JEV (world text, per-action criteria, images) plus the probability
  distribution returned. This is the distillation target for a small JEV.
* SlowRecorder wraps the outer escalation callable and captures the slow model
  prompt/response, so the planner half of the pair is auditable too.

Samples are written immediately per step, so a crash never loses the run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
SFT_ROOT = Path(os.environ.get("JEVGAMER_SFT_DIR") or (ROOT / "runs" / "sft"))

# jev_scores() prepends one of these two sentences to `world` depending on
# whether images were attached, and sets one fixed per-track instruction. They
# are mirrored here so a distilled student trains on exactly the token
# sequence the teacher saw, not just the caller-supplied memory string.
FAST_WORLD_WITH_IMAGE = (
    "Use the supplied images and current public state to pick the option whose condition is satisfied."
)
FAST_WORLD_NO_IMAGE = (
    "Use the current public state and slow-planner context. No image is attached. "
    "Pick the option whose condition is satisfied."
)
FAST_CRITERIA_INSTRUCTIONS = "Pick the single option whose usage condition is satisfied. Using `world`."


def fast_world(memory: str = "", image_count: int = 0) -> str:
    """Rebuild the exact `state.world` string jev_scores sends to SystemOne."""
    world = FAST_WORLD_WITH_IMAGE if image_count else FAST_WORLD_NO_IMAGE
    if memory:
        world += " " + memory
    return world


def _now() -> float:
    return round(time.time(), 3)


class FastRecorder:
    """Wrap a score_fn so every JEV request/response becomes one SFT sample."""

    def __init__(self, inner: Callable[..., dict], writer: "SampleWriter", meta: dict | None = None,
                 progress_getter: Callable[[], dict] | None = None) -> None:
        self.inner = inner
        self.writer = writer
        self.meta = dict(meta or {})
        # Samples are written from inside score_fn, which runs before the run
        # loop's on_step hook. Without pulling the value here, the first sample
        # of every episode is written with no progress at all.
        self.progress_getter = progress_getter
        self.calls = 0

    def __call__(self, criteria: dict, token: str, tracks=None, memory: str = "", images=None) -> dict:
        images = list(images or [])
        grouped = tracks or {"action": tuple((criteria or {}).keys())}
        per_track = {}
        for track, names in grouped.items():
            subset = {name: criteria[name] for name in names if name in (criteria or {})}
            if subset:
                per_track[track] = {
                    "type": "choice",
                    "instructions": FAST_CRITERIA_INSTRUCTIONS,
                    "criteria": subset,
                }
        request = {
            "world": memory or "",
            "full_world": fast_world(memory or "", len(images)),
            "criteria": dict(criteria or {}),
            "criteria_instructions": FAST_CRITERIA_INSTRUCTIONS,
            "questions": per_track,
            "tracks": {k: list(v) for k, v in grouped.items()},
            "image_count": len(images),
        }
        started = time.time()
        result = self.inner(criteria, token, tracks, memory=memory, images=images) or {}
        elapsed_ms = int((time.time() - started) * 1000)
        self.calls += 1
        if self.progress_getter is not None:
            try:
                self.writer.set_progress(self.progress_getter())
            except Exception:
                pass
        self.writer.write_fast(
            request=request,
            result=result,
            images=images or [],
            meta=dict(self.meta, fast_call=self.calls),
            elapsed_ms=elapsed_ms,
        )
        return result


class SlowRecorder:
    """Record planner exchanges by patching the live escalation callable.

    A proxy wrapper is unsafe here: OuterEscalation reads and writes its own
    attributes (calls / error / hint) during __call__, and forwarding those
    through __getattr__ breaks the state machine. Instead this subclass-free
    helper monkey-patches the single instance.
    """

    def __init__(self, inner: Any, writer: "SampleWriter", meta: dict | None = None) -> None:
        self.inner = inner
        self.writer = writer
        self.meta = dict(meta or {})
        self._original = inner.__call__ if callable(inner) else None
        self.calls = 0

    def install(self) -> Any:
        inner = self.inner
        writer = self.writer
        meta = self.meta
        recorder = self
        original_call = type(inner).__call__

        def patched(instance, obs, goal, decision=None, recent=None):
            before = len(getattr(instance, "calls", []) or [])
            try:
                action = original_call(instance, obs, goal, decision, recent)
            except TypeError:
                action = original_call(instance, obs, goal, decision)
            calls = getattr(instance, "calls", []) or []
            for call in calls[before:]:
                writer.write_slow(call=call, action=action, meta=meta)
                recorder.calls += 1
            return action

        inner.__class__ = type(f"Recorded{type(inner).__name__}", (type(inner),), {"__call__": patched})
        return inner


class SampleWriter:
    """Append-only writer for fast/slow SFT samples of a single run."""

    def __init__(self, game: str, run_id: str, goal: str = "", root: Path | None = None) -> None:
        self.game = game
        self.run_id = run_id
        self.goal = goal
        self.folder = (root or SFT_ROOT) / game / run_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fast_path = self.folder / "fast.jsonl"
        self.slow_path = self.folder / "slow.jsonl"
        self.fast_count = 0
        self.slow_count = 0
        self.fast_failed = 0
        # Steps skipped because the game had not published its metric yet. They
        # are not failures — the call succeeded — they just carry no evidence of
        # progress, so they cannot be counted as verified data.
        self.fast_unproven = 0
        # Filled by the run loop before each sample is written. Kept as a plain
        # attribute so the writer stays independent of adapter internals.
        self.progress: dict | None = None
        self.progress_best = None
        self.progress_metric = ""
        self.progress_series: list = []

    def set_progress(self, record: dict | None) -> None:
        """Attach the current progress snapshot to subsequent samples.

        Also accumulates the series so an episode can be scanned for the step at
        which progress stopped, which is what tells a plateau apart from noise.
        """
        self.progress = dict(record) if isinstance(record, dict) else None
        if not self.progress:
            return
        best = self.progress.get("episode_best")
        if isinstance(best, (int, float)):
            self.progress_best = best
        name = str(self.progress.get("metric") or "")
        if name:
            self.progress_metric = name
        value = self.progress.get("value")
        if isinstance(value, (int, float)):
            self.progress_series.append(value)

    def _write(self, path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_fast(self, request: dict, result: dict, images: list, meta: dict, elapsed_ms: int) -> None:
        probabilities = result.get("probabilities") or {}
        ranked = sorted(
            ((str(name), float(value)) for name, value in probabilities.items() if isinstance(value, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            # SystemOne can time out (3 x 30s) and return an empty distribution.
            # Such a step carries no teacher signal: there is no label to distill
            # from, and the caller silently fell back to the slow contract. Writing
            # it would let a student learn from a fictional uniform-ish batch, so
            # count it as a failed call instead of a sample.
            self.fast_failed += 1
            return
        # A step taken before the game has published its metric has nothing to
        # attach the action to: it is the title screen, not progress. Writing it
        # left a single unverifiable row at the head of otherwise clean episodes
        # and, worse, made progress_series start with None. Skipping it here is
        # the honest version — the step happened, it just cannot be verified, so
        # it must not be counted as verified data.
        if not isinstance((self.progress or {}).get("value"), (int, float)):
            self.fast_unproven += 1
            return
        record = {
            "kind": "fast",
            "game": self.game,
            "run_id": self.run_id,
            "goal": self.goal,
            "at": _now(),
            "elapsed_ms": elapsed_ms,
            "model": result.get("model") or "",
            # The progress snapshot at this step, taken from the game's own
            # published metric. It is stored per sample rather than only per
            # episode so an episode can be split at any point and still know
            # whether the state was advancing.
            "progress": self.progress,
            # The full model input: the exact instruction prefix jev_scores
            # prepends, the real world text, every per-action criterion, and the
            # images that were actually attached. Storing the rendered string
            # (not just the parts) means a distilled student trains on the same
            # token sequence the teacher saw.
            "prompt": {
                "world": request.get("world") or "",
                "full_world": request.get("full_world") or request.get("world") or "",
                "criteria": request.get("criteria") or {},
                "criteria_instructions": request.get("criteria_instructions") or "",
                "questions": request.get("questions") or {},
                "tracks": request.get("tracks") or {},
            },
            "images": list(images),
            "image_count": int(result.get("image_count") or len(images)),
            "target": {
                "probabilities": dict(ranked),
                "top": ranked[0][0] if ranked else "",
                "top_p": ranked[0][1] if ranked else 0.0,
            },
            "usage": result.get("usage") or {},
            "error": str(result.get("error") or "")[:400],
            "meta": meta,
        }
        self._write(self.fast_path, record)
        self.fast_count += 1

    def write_slow(self, call: dict, action: str | None, meta: dict) -> None:
        # The planner can be consulted before the first observation, when no
        # metric has been sampled yet. Recording that as an empty progress field
        # makes the row indistinguishable from "this game has no metric", which
        # is what pushed a handful of rows out of the verified tier. The explicit
        # pending marker keeps the distinction.
        progress = self.progress
        if not (progress or {}).get("metric"):
            progress = {"metric": "", "sense": "high", "value": None, "best": None,
                        "source": "pre-observation",
                        "note": "planner consulted before the first observation"}
        record = {
            "kind": "slow",
            "game": self.game,
            "run_id": self.run_id,
            "goal": self.goal,
            "at": _now(),
            "model": call.get("model") or call.get("requested_model") or "",
            # The same per-step progress snapshot the fast samples carry. Slow
            # planner exchanges are samples too, and leaving the field out meant
            # every slow record in every run looked like a trajectory with no
            # measurable advance, so those 695 rows could not be verified at all.
            "progress": progress,
            "prompt": {
                "state": call.get("screen"),
                "recent": call.get("recent"),
                "image_count": call.get("image_count"),
                "generation": call.get("generation"),
            },
            "response": call.get("response") or "",
            "action": action or "",
            "usage": call.get("usage") or {},
            "ms": call.get("ms"),
            "meta": meta,
        }
        self._write(self.slow_path, record)
        self.slow_count += 1

    def _advance_stats(self) -> dict:
        """Count real advances, not merely the number of distinct values.

        Two distinct values can mean the score climbed for two minutes, or it
        can mean it jumped from the idle value to 1 once and froze. The tier
        rule needs the second reading, so the series is summarised as: how many
        adjacent transitions happened, where the last one sits in the episode,
        and how far the value travelled above its first non-zero reading.
        """
        series = [v for v in self.progress_series if isinstance(v, (int, float))]
        if not series:
            return {"progress_changes": 0, "progress_tail_frac": 0.0,
                    "progress_span": 0, "progress_last": None}
        changes = sum(1 for a, b in zip(series, series[1:]) if a != b)
        last_change = 0
        for i in range(1, len(series)):
            if series[i] != series[i - 1]:
                last_change = i
        # Position of the final advance, 0 at the start and 1 at the end. An
        # episode that only moved at the very beginning has an almost zero
        # tail, whatever else it did afterwards.
        tail = round(last_change / max(len(series) - 1, 1), 3)
        nonzero = [v for v in series if v]
        base = nonzero[0] if nonzero else 0
        return {"progress_changes": changes, "progress_tail_frac": tail,
                "progress_span": max(series) - base, "progress_last": series[-1]}

    def summary(self) -> dict:
        return {
            "game": self.game,
            "run_id": self.run_id,
            "goal": self.goal,
            "fast_samples": self.fast_count,
            "fast_failed": self.fast_failed,
            "fast_unproven": self.fast_unproven,
            "slow_samples": self.slow_count,
            "folder": str(self.folder),
            # The episode-level progress verdict. `metric` empty with
            # source=none means the game publishes no metric at all, which is a
            # known gap rather than a silent zero.
            "progress": self.progress or {},
            "progress_metric": self.progress_metric,
            "progress_best": self.progress_best,
            "progress_series_len": len(self.progress_series),
            "progress_distinct": len(set(self.progress_series)),
            **self._advance_stats(),
        }


def attach(policy, escalate, writer: SampleWriter, game: str = "",
           progress_getter: Callable[[], dict] | None = None) -> tuple[FastRecorder, Any]:
    """Wrap a policy's score_fn and the escalation callable with recorders."""
    meta = {"game": game or writer.game, "run_id": writer.run_id}
    fast = FastRecorder(policy.score_fn, writer, meta, progress_getter=progress_getter)
    policy.score_fn = fast
    if escalate is None:
        return fast, None
    slow = SlowRecorder(escalate, writer, meta)
    slow.install()
    return fast, slow
