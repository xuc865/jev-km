"""The slow tier: a planner that never presses a key.

The previous slow tier had three jobs bolted together: publish a layered plan,
optionally author a usage condition for every legal action, and optionally return
one concrete action to unblock the fast tier. The third job is what made the
architecture expensive. Because the fast model could not be trusted to choose
well, the planner was called on a low top-probability and answered with a single
action -- so on a hard step the two tiers effectively became one, at the
planner's latency and price.

Under the new contract the planner publishes a ``PlanDirective`` and nothing
else. Its output is validated field by field, and the only way it can influence a
specific frame is ``avoid``, which is a prohibition rather than a choice. This is
a real restriction, not a stylistic one: it means a review costs the same whether
the current situation is easy or hard, and it means the executor's per-step
latency is independent of the planner's.

What the planner gains in exchange: a structured report from the executor
(``why``, ``expect``, ``belief``, ``stuck``, ``outcome``) instead of a raw
probability table, so it can tell "the plan is hard" from "the executor is
confused" without a follow-up call.
"""

from __future__ import annotations

import os

from .client import ModelClient
from .contract import (
    ExecutorStep,
    LegalAction,
    PlanDirective,
    PlannerReply,
    WorldView,
    parse_planner_reply,
    render_planner_payload,
)
from .prompts import planner_system
from .types import Decision, Observation

VERDICT_KEEP = "continue"
VERDICT_PIVOT = "pivot"
VERDICT_FINISH = "finish"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


class PlannerPolicy:
    """Publishes plan revisions; exposes them as ``.plan`` for the executor.

    Callable so it can be passed straight to ``run_episode(escalate=...)``. It
    returns no action id by design -- the return value is always None -- because
    the executor owns action selection and a planner-supplied action would bypass
    that boundary.
    """

    def __init__(
        self,
        client: ModelClient,
        model: str = "",
        identity: str = "autonomous game player",
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model or os.environ.get("JEVGAMER_PLANNER_MODEL", "deepseek-v4.1-flash")
        self.identity = identity
        # A plan revision carries briefing (<=1200 chars) plus stage_goal, tactics
        # and situation. 512 output tokens truncated every single review into an
        # unterminated JSON object, which is why the planner had a 0% label rate.
        # The ceiling is deliberately generous because the cost of a rejected
        # review is a whole stale plan interval, not just one answer.
        self.max_tokens = (
            int(max_tokens)
            if max_tokens is not None
            else _env_int("JEVGAMER_PLANNER_MAX_TOKENS", 2048)
        )
        self.plan = PlanDirective()
        self.locked_goal: str | None = None
        self.error = ""
        self.verdict = VERDICT_KEEP
        self.revisions = 0
        self.last_trigger = ""
        self.history: list[PlannerReply] = []

    def __call__(
        self,
        obs: Observation,
        goal: str,
        decision: Decision | None = None,
        recent: list[str] | None = None,
        trigger: str = "",
        report: ExecutorStep | None = None,
        step: int = 0,
    ) -> None:
        """Produce one plan revision.

        Returns None unconditionally. The executor reads ``self.plan`` after the
        call, which is also how a rejected reply stays distinguishable from an
        accepted one: a rejection leaves the previous revision in place.

        ``step`` is the executor's own counter, passed in rather than read from the
        observation: ``issued_at_step`` is compared against that same counter by
        ``PlanDirective.age``, so the two must share one clock. Reading
        ``state["step"]`` here would give a second, adapter-dependent counter and
        every plan would look older than it is.
        """
        del decision
        if self.locked_goal is None:
            self.locked_goal = goal
        goal = self.locked_goal
        self.error = ""
        self.last_trigger = trigger

        world = self._world(obs, goal, step if step else (report.step if report else 0))
        payload = render_planner_payload(
            world,
            self.plan,
            report,
            list(recent or []),
            trigger or "unspecified",
        )
        call = self.client.complete(
            role="planner",
            model=self.model,
            system=planner_system(self.identity, goal),
            user_payload=payload,
            image=obs.image or "",
            max_tokens=self.max_tokens,
        )
        if call.error:
            self.error = call.error
            return None

        reply = parse_planner_reply(call.text, self.plan, world.step, world.legal_ids())
        if reply is None:
            # Keep the previous plan rather than publishing a blank one. A planner
            # that answered with unparseable text has said nothing about whether
            # the current plan is wrong, so discarding it would lose a working
            # directive on a formatting mistake.
            self.error = "planner reply unusable"
            return None

        self.plan = reply.plan
        self.verdict = reply.verdict
        self.revisions += 1
        self.history.append(reply)
        # The planner's label is the directive delta it actually published, so a
        # distilled planner learns to revise plans rather than to emit one.
        call.label = {
            "plan": reply.plan.to_wire(),
            "verdict": reply.verdict,
            "note": reply.note,
        }
        return None

    def _world(self, obs: Observation, goal: str, step: int = 0) -> WorldView:
        """Planner view of the world: same facts, plus the input universe.

        The keyboard universe is shown here because narrowing it is a legitimate
        standing decision about a game. It is withheld from the executor so the
        executor cannot name a key that is not live this step.
        """
        state = dict(obs.state or {})
        # Same resolution order as the executor's build_world. The two tiers must
        # agree about what the game is, or the plan and the action are conditioned
        # on different worlds while the dataset records a single row.
        named = (
            str(getattr(obs, "game", "") or "")
            or str(getattr(obs, "name", "") or "")
            or str(state.get("game") or "")
        )
        return WorldView(
            game=named,
            goal=goal,
            level=int(obs.level),
            step=int(step),
            phase=str(state.get("phase") or ""),
            actions=tuple(
                LegalAction(
                    id=action.id,
                    label=action.label or action.id,
                    kind=action.kind,
                    role=str(action.params.get("semantic_role") or ""),
                    seconds=float(action.seconds or 0.0),
                )
                for action in obs.actions
            ),
            state=state,
            image_attached=bool(obs.image),
            progress=dict(state.get("progress") or {}),
        )


# The old module exposed OuterEscalation with an ``action_criteria`` payload and a
# ``control_exclude`` verdict; both are gone. Aliasing the new class keeps imports
# in runner.py, live.py and the tests working while making the removal explicit
# rather than leaving dead attributes behind.
OuterEscalation = PlannerPolicy
