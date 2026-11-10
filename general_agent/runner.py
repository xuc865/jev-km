"""Generic episode loop: observe -> executor -> act -> repeat.

The loop itself barely changed, and that is intentional: the two-tier split is a
policy concern, not a loop concern. What did change is the information the loop
carries. It no longer passes a ``hint`` string down to the policy (loose hints
were how stale strategy bypassed the plan's declared lifetimes), and it no longer
reads plan fields off the policy by attribute name for the adapter -- it asks the
policy for its directive, which is a declared object.
"""

from __future__ import annotations

import time
from typing import Callable

from .adapter import GameAdapter
from .contract import PlanDirective
from .policy import ExecutorPolicy
from .types import Decision, EpisodeResult, Observation

StepHook = Callable[[int, Observation, Decision], None]


def _directive(policy) -> PlanDirective:
    """Read the policy's current plan without depending on its concrete type.

    A duck-typed accessor rather than an isinstance check, because tests and
    alternative policies (a scripted executor, a replay policy) legitimately hold
    a plan without being an ExecutorPolicy.
    """
    plan = getattr(policy, "plan", None)
    return plan if isinstance(plan, PlanDirective) else PlanDirective()


def run_episode(
    adapter: GameAdapter,
    policy: ExecutorPolicy,
    goal: str,
    *,
    max_steps: int = 500,
    max_seconds: float = 300.0,
    should_stop: Callable[[], bool] | None = None,
    on_step: StepHook | None = None,
    escalate=None,
    hint: str = "",
) -> EpisodeResult:
    """Run one episode; success comes only from the adapter's observation.

    ``hint`` is retained for signature compatibility and ignored: the planner
    communicates through the plan contract, and a side-channel hint would be a
    third, undeclared path between the tiers.
    """
    del hint
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
            decision = policy.choose(obs, goal, recent=recent, escalate=escalate)
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
                plan = _directive(policy)
                adapter.show_decision(
                    decision,
                    {
                        "stage_goal": plan.stage_goal,
                        "tactics": plan.tactics,
                        "plan_revision": plan.revision,
                        "planner_model": getattr(escalate, "model", "") if escalate else "",
                    },
                )
            adapter.act(action)
            if hasattr(policy, "record_execution"):
                policy.record_execution(obs, decision, action, getattr(adapter, "last_execution", {}))
            recent.append(decision.action_id)
            recent = recent[-12:]
        return EpisodeResult(False, max_steps, time.time() - started, decisions, error="max steps")
    finally:
        adapter.close()
