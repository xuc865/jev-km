"""Strict wire contract between the fast executor and the slow planner.

Both tiers now run the same model family, so the only thing separating them is
the role prompt plus this contract. Two consequences follow, and both are why
this module exists instead of the free-form dicts the old design passed around:

1. Nothing crosses the boundary that is not declared here. The old contract let
   the slow model ship an open-ended ``play_instruction`` string plus a
   per-action criteria map, so the fast model's input shape drifted with
   whatever prose the planner happened to write that review. A declared schema
   makes the exchange inspectable and reproducible.

2. Field names are training labels. Distillation reads the same keys the runtime
   uses, so a change here is a change to the dataset. That is why each field
   declares its lifetime instead of leaving it implied by where code clears it.

Lifetimes, fastest to slowest:

    briefing     whole episode      what the game is, controls, win condition
    stage_goal   one stage          objective of the current phase/level
    tactics      one stage          how to pursue the stage goal
    avoid        one stage          action ids ruled out for the current plan
    exclude      whole episode      input ids pruned from the keyboard universe
    situation    one review         one line about the present moment
    action       one step           the single executed control

``avoid`` and ``exclude`` both remove options, and the difference is duration plus
who enforces them. ``avoid`` is advice to the executor about the current plan and
is checked at execution time. ``exclude`` is a standing decision about the game --
"this key never does anything here" -- so the adapter applies it to the candidate
list itself, which also means it changes what counts as a legal action next step.

A reply that only refreshes ``situation`` therefore cannot silently invalidate
``stage_goal``, which is the failure the old design had to guard with an
"always publish a complete revision" rule.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "fs2"

# Review cadence the planner may request. A planner that asks for a very short
# interval turns the slow tier back into a per-step caller, which is the cost
# this architecture exists to avoid; one that asks for a very long interval
# reintroduces the stale-plan failure. The band keeps both bounded.
MIN_REVIEW_INTERVAL = 2
MAX_REVIEW_INTERVAL = 40

# Longest accepted text for each free-text field. These are not cosmetic: the
# planner is asked for a compact directive, and a 4000-character `tactics`
# block crowds the executor's observation out of its own context window.
TEXT_LIMITS = {
    "briefing": 1200,
    "stage_goal": 400,
    "tactics": 800,
    "situation": 300,
    "why": 300,
    "expect": 240,
    "belief": 400,
}

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def clamp_text(value: Any, limit: int) -> str:
    """Single-line, length-bounded text.

    Newlines are collapsed because every one of these fields is embedded in a
    JSON payload whose shape is part of the contract; a planner that answers
    with a multi-paragraph tactic would otherwise change the token layout the
    distilled student was trained on.
    """
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text[:limit]


def load_json_object(text: str) -> dict | None:
    """Extract the first JSON object from a model reply.

    Models wrap JSON in fences, prefix it with prose, or append a closing
    sentence. Searching for the outermost brace pair handles all three without
    a tolerant parser, and returning None (rather than a partial dict) keeps the
    caller's decision to escalate explicit.
    """
    if not text:
        return None
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(0.0, min(1.0, number))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


@dataclass(frozen=True)
class LegalAction:
    """One control the adapter will actually accept this step.

    Deliberately flat and already string-typed: the executor sees exactly these
    fields and nothing else, so an action it names is either in this list or
    rejected. The adapter keeps ownership of enumeration and pruning.
    """

    id: str
    label: str = ""
    kind: str = "hold"
    role: str = ""
    seconds: float = 0.15

    def describe(self) -> str:
        bits = [self.label or self.id]
        if self.role:
            bits.append(f"role={self.role}")
        if self.seconds:
            bits.append(f"hold={self.seconds:g}s")
        if self.kind:
            bits.append(f"kind={self.kind}")
        return " | ".join(bits)

    def to_wire(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "role": self.role,
            "seconds": round(float(self.seconds), 3),
        }


@dataclass(frozen=True)
class WorldView:
    """Environment facts both tiers are allowed to see.

    Built once per step and reused for both prompts so the two tiers cannot
    disagree about what the observation was. ``state`` is already projected by
    the adapter; this class does not decide what is public.
    """

    game: str
    goal: str
    level: int
    step: int
    phase: str
    actions: tuple[LegalAction, ...]
    state: dict = field(default_factory=dict)
    image_attached: bool = False
    progress: dict = field(default_factory=dict)

    def legal_ids(self) -> set[str]:
        return {action.id for action in self.actions}

    def to_wire(self) -> dict:
        # Key order is fixed on purpose: the rendered prompt is a training
        # artifact, and a reordered dict would produce a different token
        # sequence for an identical situation.
        return {
            "game": self.game,
            "goal": self.goal,
            "level": self.level,
            "step": self.step,
            "phase": self.phase,
            "image_attached": self.image_attached,
            "legal_actions": [action.to_wire() for action in self.actions],
            "state": self.state,
            "progress": self.progress,
        }


@dataclass(frozen=True)
class PlanDirective:
    """What the planner publishes; the only slow-tier state the executor reads.

    ``revision`` increases on every accepted planner reply. The executor stamps
    it onto each step report so a trajectory can be segmented into plan epochs
    after the fact, which is what lets distillation distinguish "the plan was
    wrong" from "the execution was wrong".
    """

    briefing: str = ""
    stage_goal: str = ""
    tactics: str = ""
    situation: str = ""
    avoid: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    review_in: int = 8
    revision: int = 0
    issued_at_step: int = 0
    schema: str = SCHEMA_VERSION

    def to_wire(self) -> dict:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "issued_at_step": self.issued_at_step,
            "briefing": self.briefing,
            "stage_goal": self.stage_goal,
            "tactics": self.tactics,
            "situation": self.situation,
            "avoid": list(self.avoid),
            "exclude": list(self.exclude),
            "review_in": self.review_in,
        }

    def age(self, step: int) -> int:
        return max(0, step - self.issued_at_step)

    def forbids(self, action_id: str) -> bool:
        return action_id in set(self.avoid) or action_id in set(self.exclude)


@dataclass(frozen=True)
class PlannerReply:
    """Parsed planner output.

    ``verdict`` is the planner's own read on whether the current approach works.
    It is a separate field from ``tactics`` because "keep going" and "this stage
    plan is wrong" need different handling in the runner: the second one files
    the episode for review even when steps are still being spent.
    """

    plan: PlanDirective
    verdict: str = "continue"  # continue | pivot | finish
    note: str = ""


@dataclass(frozen=True)
class ExecutorReply:
    """Parsed executor output: one action plus the evidence behind it.

    ``belief`` and ``expect`` exist for the slow tier, not for the runner. A
    planner that only sees the chosen action cannot tell a confident player from
    one that is guessing, so the executor states what it thinks is happening and
    what it expects the action to cause. The mismatch between ``expect`` and the
    next observation becomes the cheapest available signal that the plan is
    wrong, and it costs no extra model call.
    """

    action: str
    why: str = ""
    confidence: float = 0.0
    expect: str = ""
    belief: str = ""
    stuck: bool = False
    raw: str = ""

    def to_wire(self) -> dict:
        return {
            "action": self.action,
            "why": self.why,
            "confidence": round(self.confidence, 3),
            "expect": self.expect,
            "belief": self.belief,
            "stuck": self.stuck,
        }


@dataclass(frozen=True)
class ExecutorStep:
    """What the executor reports upward when a review is requested.

    This is the fast tier's entire contribution to the planner's context. It is
    a summary, not the transcript: sending the planner every frame would make
    the review cost grow with the episode length, which is what the old
    full-state payload did.
    """

    step: int
    action: str
    why: str = ""
    confidence: float = 0.0
    expect: str = ""
    belief: str = ""
    stuck: bool = False
    repeat_run: int = 0
    plan_revision: int = 0
    plan_age: int = 0
    outcome: str = ""

    def to_wire(self) -> dict:
        return {
            "step": self.step,
            "action": self.action,
            "why": self.why,
            "confidence": round(self.confidence, 3),
            "expect": self.expect,
            "belief": self.belief,
            "stuck": self.stuck,
            "repeat_run": self.repeat_run,
            "plan_revision": self.plan_revision,
            "plan_age": self.plan_age,
            "outcome": self.outcome,
        }


def parse_executor_reply(text: str, legal: set[str]) -> ExecutorReply | None:
    """Validate an executor reply against the live legal set.

    Returns None rather than a repaired value when the action is not legal. A
    fabricated id means the model misread its own action list, and executing the
    nearest match would attribute the mistake to the game instead of to the
    call, which is the harder failure to debug.
    """
    parsed = load_json_object(text)
    if not parsed:
        return None
    action = str(parsed.get("action") or "").strip()
    if action not in legal:
        return None
    return ExecutorReply(
        action=action,
        why=clamp_text(parsed.get("why"), TEXT_LIMITS["why"]),
        confidence=_as_float(parsed.get("confidence"), 0.0),
        expect=clamp_text(parsed.get("expect"), TEXT_LIMITS["expect"]),
        belief=clamp_text(parsed.get("belief"), TEXT_LIMITS["belief"]),
        stuck=_as_bool(parsed.get("stuck")),
        raw=text or "",
    )


def parse_planner_reply(
    text: str,
    previous: PlanDirective,
    step: int,
    legal: set[str],
) -> PlannerReply | None:
    """Validate a planner reply and merge it onto the previous directive.

    Every field is optional: a planner that only updates ``situation`` keeps the
    rest, which is what the declared lifetimes mean in practice. An empty reply
    is rejected outright so a blank answer cannot silently wipe a working plan.
    """
    parsed = load_json_object(text)
    if not parsed:
        return None

    # `avoid` may name only controls that exist this step. An id outside the
    # legal set is treated as a typo in a prohibition, which is cheap to ignore,
    # unlike an unknown action id in an executor reply which is expensive.
    raw_avoid = parsed.get("avoid")
    avoid: tuple[str, ...] = previous.avoid
    if isinstance(raw_avoid, list):
        avoid = tuple(
            sorted({str(item).strip() for item in raw_avoid if str(item).strip() in legal})
        )
    elif isinstance(raw_avoid, str) and raw_avoid.strip().lower() in {"none", "clear", ""}:
        avoid = ()

    # `exclude` names ids the adapter should stop offering at all. Validation is
    # deliberately the same shape as `avoid`: an unknown id is a typo in a
    # prohibition, so it is dropped rather than failing the whole reply.
    raw_exclude = parsed.get("exclude")
    exclude: tuple[str, ...] = previous.exclude
    if isinstance(raw_exclude, list):
        exclude = tuple(
            sorted({str(item).strip() for item in raw_exclude if str(item).strip()})
        )
    elif isinstance(raw_exclude, str) and raw_exclude.strip().lower() in {"none", "clear", ""}:
        exclude = ()

    try:
        review_in = int(parsed.get("review_in", previous.review_in))
    except (TypeError, ValueError):
        review_in = previous.review_in
    review_in = max(MIN_REVIEW_INTERVAL, min(MAX_REVIEW_INTERVAL, review_in))

    verdict = str(parsed.get("verdict") or "continue").strip().lower()
    if verdict not in {"continue", "pivot", "finish"}:
        verdict = "continue"

    plan = PlanDirective(
        briefing=clamp_text(parsed.get("briefing"), TEXT_LIMITS["briefing"]) or previous.briefing,
        stage_goal=clamp_text(parsed.get("stage_goal"), TEXT_LIMITS["stage_goal"]) or previous.stage_goal,
        tactics=clamp_text(parsed.get("tactics"), TEXT_LIMITS["tactics"]) or previous.tactics,
        situation=clamp_text(parsed.get("situation"), TEXT_LIMITS["situation"]) or previous.situation,
        avoid=avoid,
        exclude=exclude,
        review_in=review_in,
        revision=previous.revision + 1,
        issued_at_step=step,
    )
    if not (plan.briefing or plan.stage_goal or plan.tactics):
        # A reply that refreshed nothing is not a revision; accepting it would
        # reset the review clock and let the executor run an empty plan.
        return None
    return PlannerReply(
        plan=plan,
        verdict=verdict,
        note=clamp_text(parsed.get("note"), 200),
    )


# --- prompt rendering -------------------------------------------------------
#
# Both renderers are pure functions over declared fields. Nothing here reaches
# into the adapter or the runtime: if a future field is needed in a prompt it
# must be added to the dataclass first, which is the point of the contract.

EXECUTOR_SCHEMA_HINT = (
    '{"action":"<one id from legal_actions>","why":"<=300 chars","confidence":0.0-1.0,'
    '"expect":"<=240 chars what this action should cause","belief":"<=400 chars what you think is happening",'
    '"stuck":false}'
)

PLANNER_SCHEMA_HINT = (
    '{"briefing":"<=1200 chars, keep unless wrong","stage_goal":"<=400 chars",'
    '"tactics":"<=800 chars","situation":"<=300 chars","avoid":["action ids"],'
    '"exclude":["input ids to stop offering"],'
    '"review_in":8,"verdict":"continue|pivot|finish","note":"<=200 chars"}'
)


def render_executor_payload(
    world: WorldView,
    plan: PlanDirective,
    recent: list[str],
    repeat_run: int,
) -> dict:
    """The executor's user turn: environment, plan, history, nothing else."""
    return {
        "world": world.to_wire(),
        "plan": plan.to_wire(),
        "recent_actions": list(recent[-12:]),
        "repeat_run": repeat_run,
        "output_schema": EXECUTOR_SCHEMA_HINT,
    }


def render_planner_payload(
    world: WorldView,
    plan: PlanDirective,
    report: ExecutorStep | None,
    recent: list[str],
    trigger: str,
) -> dict:
    """The planner's user turn: the same world plus the executor's report."""
    return {
        "world": world.to_wire(),
        "current_plan": plan.to_wire(),
        "executor_report": report.to_wire() if report else None,
        "recent_actions": list(recent[-12:]),
        "review_trigger": trigger,
        "output_schema": PLANNER_SCHEMA_HINT,
    }


def encode(payload: Any) -> str:
    """Canonical JSON used for every model turn and every SFT record.

    Separators are pinned and key order is preserved from the declared dicts, so
    two identical situations always encode to identical bytes. ``ensure_ascii``
    is off because the goals are Chinese and escaping them triples the token
    count of every prompt.
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
