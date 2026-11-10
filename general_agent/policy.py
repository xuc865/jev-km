"""The fast tier: an executor that acts every step.

This replaces the previous probability-scoring policy. The difference is not
cosmetic, and it is worth stating why the old shape was abandoned.

The old fast tier was a scoring endpoint: it received a map of action id to
hand-written usage condition and returned a probability for each. That forced
three things on the architecture. The candidate list had to be fully enumerated
and described every step, because the scorer could not name anything it was not
given. The confidence threshold became the escalation mechanism, because a
probability table has no other way to say "I am unsure". And the planner had to
author per-action conditions to make the map meaningful, which is exactly the
burden the planner was supposed to be free of.

A chat model with a role prompt does not need any of that. It can report its own
confidence, state what it expects to happen, and name one action directly. So this
policy does the following instead:

- it holds a ``PlanDirective`` and nothing else about strategy;
- it renders one contract payload and asks for one action plus evidence;
- it escalates on evidence, not on a threshold: low self-reported confidence, an
  explicit ``stuck`` flag, a run of identical actions, a plan that has aged out,
  or a phase change;
- it never sees the keyboard universe, the win condition or the budget.

What it keeps from the old design, deliberately: the adapter owns candidate
enumeration; a failure to obtain an action does not halt the episode on its own;
and every executed step is fed back through ``record_execution`` so both tiers'
histories describe what actually happened.
"""

from __future__ import annotations

import os
import time

from .client import ModelClient
from .contract import (
    ExecutorReply,
    ExecutorStep,
    LegalAction,
    PlanDirective,
    WorldView,
    parse_executor_reply,
    render_executor_payload,
)
from .prompts import executor_system
from .types import Action, Decision, Observation

# Triggers, in the order they are checked. Kept as constants because the SFT
# records carry the trigger string, and a typo would silently create a new class.
TRIGGER_FIRST = "first_step"
TRIGGER_AGE = "plan_age"
TRIGGER_LOW_CONF = "low_confidence"
TRIGGER_NO_CHANGE = "no_change"
TRIGGER_PHASE = "phase_change"
TRIGGER_STUCK = "executor_stuck"
TRIGGER_ERROR = "executor_error"

# A run this long of the same action is a local loop. The executor is told to
# report `stuck` before reaching it, so hitting this means the self-report was
# not trusted or did not arrive.
REPEAT_LIMIT = 3


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


class ExecutorPolicy:
    """Acts every step under a planner-issued directive."""

    def __init__(
        self,
        client: ModelClient,
        model: str = "",
        identity: str = "autonomous game player",
        max_candidates: int = 32,
        confidence: float | None = None,
        review_every: int = 8,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model or os.environ.get("JEVGAMER_EXECUTOR_MODEL", "deepseek-v4.1-flash")
        self.identity = identity
        self.max_candidates = max(2, int(max_candidates))
        # The executor's reply is one action plus five short fields, but a verbose
        # model will spend hundreds of tokens restating the plan before answering.
        # A ceiling that is too tight does not shorten the answer, it cuts the JSON
        # mid-object and turns a usable label into a rejected one.
        self.max_tokens = (
            int(max_tokens)
            if max_tokens is not None
            else _env_int("JEVGAMER_EXECUTOR_MAX_TOKENS", 1024)
        )
        # Self-reported confidence below this escalates. Unlike the old gate this
        # is the model's own number about its own choice, not a softmax top mass,
        # so it is comparable across candidate counts.
        self.confidence = (
            float(confidence)
            if confidence is not None
            else _env_float("JEVGAMER_EXECUTOR_CONFIDENCE", 0.45)
        )
        self.review_every = max(2, int(review_every))

        self.plan = PlanDirective()
        self.locked_goal: str | None = None
        self.steps = 0
        self.last_review_step = 0
        self.steps_since_review = 0
        self.last_phase: tuple = ()
        self.last_level = 0
        self.last_trigger = ""

        self.recent: list[str] = []
        self.last_reply: ExecutorReply | None = None
        self.last_step: ExecutorStep | None = None
        self.failures = 0
        self.escalations = 0
        self.history: list[ExecutorStep] = []
        # Set by the panel wiring when the adapter can prune its own candidate
        # list. Left as None for adapters that enumerate a fixed action set, which
        # is why `_apply_exclude` must tolerate its absence.
        self.subset_sink = None
        self.exclude_note = ""

    # -- plan handling -------------------------------------------------------

    def apply_plan(self, plan: PlanDirective) -> None:
        """Adopt a new directive.

        The review clock resets here rather than at request time, so a review that
        returned nothing usable does not buy the executor another full interval on
        a stale plan.
        """
        self.plan = plan
        self.last_review_step = self.steps
        self.steps_since_review = 0
        self.failures = 0

    def plan_interval(self) -> int:
        """Effective review interval: the plan's request, bounded by config.

        The planner may ask for a longer interval when a plan is working, but it
        cannot ask for shorter than the configured floor, which is what stops the
        slow tier from becoming a per-step caller.
        """
        requested = int(self.plan.review_in or self.review_every)
        return max(2, min(max(requested, self.review_every), 40))

    # -- world construction --------------------------------------------------

    def build_world(self, obs: Observation, goal: str) -> WorldView:
        actions = tuple(
            LegalAction(
                id=action.id,
                label=action.label or action.id,
                kind=action.kind,
                role=str(action.params.get("semantic_role") or ""),
                seconds=float(action.seconds or 0.0),
            )
            for action in obs.actions
        )
        # The input universe and its pruning history belong to the planner's view
        # only. The executor must not be able to name a key that is not legal this
        # step, and showing it a wider set invites exactly that.
        state = {
            key: value
            for key, value in (obs.state or {}).items()
            if key not in {"input_universe", "input_excluded", "input_excluded_note",
                           "input_subset_history", "recurrence", "hint", "instructions"}
        }
        return WorldView(
            game=str(getattr(obs, "name", "") or ""),
            goal=goal,
            level=int(obs.level),
            step=self.steps,
            phase=str(state.get("phase") or ""),
            actions=actions,
            state=state,
            image_attached=bool(obs.image),
            progress=dict(state.get("progress") or {}),
        )

    def repeat_run(self) -> int:
        """Length of the current run of identical actions at the tail."""
        if not self.recent:
            return 0
        last = self.recent[-1]
        run = 0
        for action in reversed(self.recent):
            if action != last:
                break
            run += 1
        return run

    # -- triggers ------------------------------------------------------------

    def review_trigger(self, obs: Observation, world: WorldView) -> str:
        """Why a review is due, or an empty string when the plan still stands.

        Ordered by how much information the review would add. ``first_step`` wins
        because no plan exists; a phase change outranks aging because the cached
        tactics describe a situation that no longer exists.
        """
        phase = (world.phase, tuple(sorted(world.legal_ids())))
        if not self.plan.stage_goal:
            return TRIGGER_FIRST
        if self.failures >= 2:
            return TRIGGER_ERROR
        if self.last_reply is not None and self.last_reply.stuck:
            return TRIGGER_STUCK
        if self.last_phase and phase != self.last_phase:
            return TRIGGER_PHASE
        if world.level != self.last_level and self.last_level:
            return TRIGGER_PHASE
        if self.repeat_run() >= REPEAT_LIMIT:
            return TRIGGER_NO_CHANGE
        if self.last_reply is not None and self.last_reply.confidence < self.confidence:
            return TRIGGER_LOW_CONF
        if self.steps_since_review >= self.plan_interval():
            return TRIGGER_AGE
        return ""

    # -- execution -----------------------------------------------------------

    def choose(
        self,
        obs: Observation,
        goal: str,
        recent: list[str] | None = None,
        hint: str = "",
        escalate=None,
    ) -> Decision:
        """Choose one action, escalating to the planner when a trigger fires.

        The signature keeps ``hint`` and ``escalate`` for runner compatibility.
        ``hint`` is unused: under the contract, loose hints were the mechanism by
        which stale strategy leaked past the plan's own lifetime, so there is no
        place for one here.
        """
        del hint
        if self.locked_goal is None:
            self.locked_goal = goal
        goal = self.locked_goal

        if obs.done:
            return Decision("", by="stop", note="episode already done")
        if not obs.actions:
            return Decision("", by="stop", note="no legal actions")
        if len(obs.actions) > self.max_candidates:
            return Decision(
                "",
                by="stop",
                note=f"{len(obs.actions)} candidates > {self.max_candidates}; adapter must prune",
            )

        self.steps += 1
        if recent:
            self.recent = list(recent[-12:])
        world = self.build_world(obs, goal)

        trigger = self.review_trigger(obs, world) if escalate is not None else ""
        self.last_trigger = trigger
        if trigger:
            if not self._review(obs, world, goal, trigger, escalate):
                # A failed review is an optimisation lost, not a reason to stop.
                # The old design learned this the expensive way: with the review as
                # a precondition, one empty reply discarded a whole 37-step
                # trajectory at step 4. Only a sustained outage ends the run.
                self.failures += 1
                if self.failures >= 6:
                    detail = self.client.last_error or "no reply"
                    return Decision(
                        "",
                        by="stop",
                        note=f"planner unavailable: {detail}",
                        error=f"planner unavailable: {detail}",
                    )

        return self._execute(obs, world, goal)

    def _review(self, obs: Observation, world: WorldView, goal: str, trigger: str, escalate) -> bool:
        """Ask the planner for a directive; return True when one was adopted."""
        report = self.last_step
        action = escalate(obs, goal, Decision("", by="planner", probabilities={}),
                          recent=list(self.recent), trigger=trigger, report=report,
                          step=self.steps)
        plan = getattr(escalate, "plan", None)
        if getattr(escalate, "error", ""):
            self.last_review_step = self.steps
            return False
        if not isinstance(plan, PlanDirective):
            self.last_review_step = self.steps
            return False
        if plan.revision <= self.plan.revision and plan.stage_goal == self.plan.stage_goal:
            # An identical revision is not a decision; counting it would reset the
            # clock without changing anything.
            self.last_review_step = self.steps
            return False
        self.apply_plan(plan)
        self.last_phase = (world.phase, tuple(sorted(world.legal_ids())))
        self.last_level = world.level
        self._apply_exclude(plan)
        del action
        return True

    def _apply_exclude(self, plan: PlanDirective) -> None:
        """Hand the planner's standing pruning decision to the adapter.

        The adapter owns candidate enumeration, so it is also the only place that
        can honour "this key never does anything here". A rejection is recorded
        rather than retried: the adapter refuses when the request would empty the
        keyboard, and resubmitting the same list every step would spend the
        planner's budget on a decision already proven wrong.
        """
        if self.subset_sink is None or not plan.exclude:
            return
        try:
            result = self.subset_sink(plan.exclude, note=plan.situation)
        except Exception as exc:  # adapter is optional; never break a step
            self.exclude_note = f"sink failed: {type(exc).__name__}: {str(exc)[:120]}"
            return
        if isinstance(result, dict):
            self.exclude_note = (
                "" if result.get("ok") else str(result.get("reason") or "rejected")[:200]
            )

    def last_pick(self) -> str:
        return self.recent[-1] if self.recent else ""

    def _execute(self, obs: Observation, world: WorldView, goal: str) -> Decision:
        """One executor call, then validate and record its reply."""
        payload = render_executor_payload(world, self.plan, self.recent, self.repeat_run())
        call = self.client.complete(
            role="executor",
            model=self.model,
            system=executor_system(self.identity, goal),
            user_payload=payload,
            image=obs.image or "",
            max_tokens=self.max_tokens,
        )
        if call.error:
            return self._fallback(obs, world, f"executor call failed: {call.error}")

        legal = world.legal_ids()
        reply = parse_executor_reply(call.text, legal)
        if reply is None:
            return self._fallback(obs, world, "executor reply unusable")

        # A forbidden action is a contract violation, not a preference. Falling
        # back rather than executing it keeps `avoid` meaningful.
        if self.plan.forbids(reply.action):
            return self._fallback(obs, world, f"executor chose forbidden {reply.action}")

        self.last_reply = reply
        # The label is the accepted contract object itself, not a re-derivation of
        # it: the recorder must write what validation approved, and only this
        # method knows that. Fields the model omitted stay empty here, which is
        # what makes a partially-specified reply trainable as-is.
        call.label = reply.to_wire()
        self.steps_since_review += 1
        self.recent = (self.recent + [reply.action])[-12:]
        step = ExecutorStep(
            step=self.steps,
            action=reply.action,
            why=reply.why,
            confidence=reply.confidence,
            expect=reply.expect,
            belief=reply.belief,
            stuck=reply.stuck,
            repeat_run=self.repeat_run(),
            plan_revision=self.plan.revision,
            plan_age=self.plan.age(self.steps),
        )
        self.last_step = step
        self.history.append(step)
        return Decision(
            reply.action,
            by="executor",
            model=call.model,
            ms=call.ms,
            usage=call.usage,
            note=f"{reply.why[:120]} | conf {reply.confidence:.2f}",
        )

    def _fallback(self, obs: Observation, world: WorldView, note: str) -> Decision:
        """Pick without the model: a non-forbidden, non-wait action.

        Deliberately not a game heuristic. ``wait`` is excluded because in a real
        game it is usually offered during a transition, and taking it on failure
        converts a decision problem into an indefinite stall. If nothing qualifies
        the episode stops, which is the honest outcome: the harness has no legal
        way to act and should say so rather than guess.
        """
        candidates = [
            action.id
            for action in obs.actions
            if action.id != "wait" and not self.plan.forbids(action.id)
        ]
        if not candidates:
            return Decision("", by="stop", note=note, error=note)
        pick = candidates[0]
        self.steps_since_review += 1
        self.last_step = ExecutorStep(
            step=self.steps,
            action=pick,
            why=f"fallback: {note}",
            confidence=0.0,
            plan_revision=self.plan.revision,
            plan_age=self.plan.age(self.steps),
            outcome="fallback",
        )
        return Decision(pick, by="fallback", note=note[:200], error=note[:200])

    def record_execution(self, obs, decision, action: Action, execution) -> None:
        """Attach the observed outcome of the executed step.

        The outcome is compared against what the executor said it expected. That
        comparison is the cheapest available signal that the plan is wrong, and it
        costs no model call, so it is recorded per step and surfaced to the
        planner through ``executor_report.outcome``.
        """
        if self.last_step is None:
            return
        outcome = ""
        if isinstance(execution, dict):
            changed = execution.get("changed")
            if changed is None:
                changed = execution.get("state_changed")
            if isinstance(changed, bool):
                outcome = "state_changed" if changed else "no_visible_change"
            elif execution.get("error"):
                outcome = "error: " + str(execution["error"])[:120]
        if not outcome:
            outcome = str(execution.get("note") or "")[:120] if isinstance(execution, dict) else ""
        self.last_step = ExecutorStep(
            step=self.last_step.step,
            action=self.last_step.action,
            why=self.last_step.why,
            confidence=self.last_step.confidence,
            expect=self.last_step.expect,
            belief=self.last_step.belief,
            stuck=self.last_step.stuck,
            repeat_run=self.last_step.repeat_run,
            plan_revision=self.last_step.plan_revision,
            plan_age=self.last_step.plan_age,
            outcome=outcome,
        )
        if self.history and self.history[-1].step == self.last_step.step:
            self.history[-1] = self.last_step


# Backwards-compatible alias. The runner and live panel import JevPolicy by name,
# and renaming it everywhere in one commit would obscure what actually changed:
# the tier's responsibility, not its symbol.
JevPolicy = ExecutorPolicy
