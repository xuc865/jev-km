"""Generic episode loop: observe -> policy -> act -> repeat."""

from __future__ import annotations

import time
from typing import Callable

from .adapter import GameAdapter
from .policy import Escalate, JevPolicy
from .types import Decision, EpisodeResult, Observation

StepHook = Callable[[int, Observation, Decision], None]


def run_episode(
    adapter: GameAdapter,
    policy: JevPolicy,
    goal: str,
    *,
    max_steps: int = 500,
    max_seconds: float = 300.0,
    should_stop: Callable[[], bool] | None = None,
    on_step: StepHook | None = None,
    escalate: Escalate | None = None,
    hint: str = "",
) -> EpisodeResult:
    """Run one episode; success comes only from the adapter's observation."""
    started = time.time()
    decisions: list[Decision] = []
    recent: list[str] = []
    try:
        adapter.reset()
        for step in range(max_steps):
            if should_stop is not None and should_stop():
                return EpisodeResult(False, step, time.time() - started, decisions, stopped=True)
            if time.time() - started > max_seconds:
                return EpisodeResult(False, step, time.time() - started, decisions, error="max seconds")
            obs = adapter.observe()
            if obs.done:
                return EpisodeResult(bool(obs.success), step, time.time() - started, decisions)
            decision = policy.choose(obs, goal, recent=recent, hint=hint, escalate=escalate)
            decisions.append(decision)
            if on_step is not None:
                on_step(step, obs, decision)
            if decision.by == "stop" or not decision.action_id:
                return EpisodeResult(
                    False,
                    step,
                    time.time() - started,
                    decisions,
                    error=decision.error or decision.note,
                )
            action = next((item for item in obs.actions if item.id == decision.action_id), None)
            if action is None:
                return EpisodeResult(
                    False,
                    step,
                    time.time() - started,
                    decisions,
                    error=f"policy chose unknown action {decision.action_id}",
                )
            if should_stop is not None and should_stop():
                return EpisodeResult(False, step, time.time() - started, decisions, stopped=True)
            if hasattr(adapter, "show_decision"):
                adapter.show_decision(decision, {"stage_goal": getattr(policy, "outer_stage_goal", ""),
                    "play_instruction": getattr(policy, "outer_play_instruction", ""),
                    "outer_model": (getattr(escalate, "calls", [])[-1].get("model", "")
                                    if getattr(escalate, "calls", []) else getattr(escalate, "model", ""))})
            adapter.act(action)
            if hasattr(policy, "record_execution"):
                policy.record_execution(obs, decision, action, getattr(adapter, "last_execution", {}))
            recent.append(decision.action_id)
            recent = recent[-12:]
        return EpisodeResult(False, max_steps, time.time() - started, decisions, error="max steps")
    finally:
        adapter.close()
