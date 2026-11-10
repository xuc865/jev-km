"""Model-only inner policy: JEV scores the legal candidates.

There is deliberately no game-specific fallback here. If JEV is unsure or
errors, the policy asks the outer escalation callback (a model) for one legal
action; if that is unavailable or also undecided, it stops. The game adapter
owns candidate enumeration/pruning, so card/click candidates stay bounded.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Callable

from agent_harness import Thread

from .types import Decision, Observation
from .context import prompt_state
from .recurrence import RecurrenceTracker

ScoreFn = Callable[..., dict]
Escalate = Callable[[Observation, str, Decision], str | None]

# The fast scorer has exactly one upstream model. When several collectors share
# it, a burst of requests earns a 429 and the vendor imposes a cooldown during
# which every call answers with an empty probability table. Those empty answers
# are not a game failure; they are back-pressure. Waiting and retrying keeps the
# episode alive instead of discarding it after a single throttled step.
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_WAIT = 12.0

# WeMM-JEV can stop answering entirely (TCP accepts, HTTP never returns), which
# is not back-pressure and will not clear by waiting. When that happens the
# scorer falls back to the chat channel, which answers the same choice question
# as a text classification. The fallback is sticky for the episode so a dead
# endpoint is probed once, not on every step.
FALLBACK_AFTER = 2


def _is_rate_limited(scores: dict) -> bool:
    """True when the empty score table came from throttling, not from a bad prompt."""
    if not isinstance(scores, dict):
        return False
    error = str(scores.get("error") or "").lower()
    if not error:
        return False
    return any(word in error for word in ("rate", "cooling", "429", "limit", "throttl"))




def _default_score_fn() -> ScoreFn:
    from harness import jev_scores

    return jev_scores


class JevPolicy:
    """Multimodal JEV scorer over the current legal candidates."""

    def __init__(
        self,
        token: str = "",
        confidence: float | None = None,
        qpm: int | None = None,
        score_fn: ScoreFn | None = None,
        max_candidates: int = 32,
        identity: str = "You are the autonomous player of the current game. Use only available game controls.",
        context_mode: str = "standard",
    ) -> None:
        if context_mode not in {"standard","compact","criteria"}:raise ValueError("Unknown fast context mode")
        self.context_mode=context_mode
        self.token = token
        self.identity = identity
        self.locked_goal: str | None = None
        self.thread = Thread(history_turns=4)
        self.confidence = (
            float(confidence)
            if confidence is not None
            else float(os.environ.get("JEVGAMER_JEV_CONFIDENCE") or 0.6)
        )
        self.qpm = int(qpm or os.environ.get("JEVGAMER_JEV_QPM") or 600)
        self.score_fn = score_fn or _default_score_fn()
        self.max_candidates = max(2, int(max_candidates))
        self.call_times: list[float] = []
        # Primary-scorer health. The fast endpoint can hang without erroring, so
        # consecutive empty answers are counted and the episode switches to the
        # chat scorer instead of stopping at the first step.
        self.scorer_failures = 0
        self.use_fallback = False

        self.outer_hint = ""
        self.outer_situation = ""
        self.outer_briefing = ""
        self.outer_stage_goal = ""
        self.outer_play_instruction = ""
        self.outer_action_criteria={}
        self.outer_criteria_scope=None
        # Injected by LiveRun for adapters that support slow-model input pruning.
        # Kept as a callback so the policy stays adapter-agnostic.
        self.subset_sink = None
        # What the adapter actually did with the slow model's verdict, and why it
        # refused it when it did. Kept on the policy so a rejection is not lost
        # the moment the review returns.
        self.subset_status: dict = {}
        self.subset_note = ""

        self.outer_action: str | None = None
        self.steps = 0
        self.recurrence = RecurrenceTracker()
        self.call_outer_every = 8
        self.last_outer_phase = None
        self.last_outer_signature = None
        self.commit_action = ""
        self.commit_steps = 0
        self.last_outer_cell = None
        self.last_outer_level = 0
        self.last_outer_step = 0
        self.steps_since_outer = 0
        self.trail: list[tuple[tuple[int, int], str, tuple[int, int]]] = []
        self.visited: set[tuple[int, int]] = set()
        self.last_cell: tuple[int, int] | None = None
        self.last_action = ""

    def _budget_ok(self) -> bool:
        now = time.time()
        self.call_times = [stamp for stamp in self.call_times if now - stamp < 60.0]
        return len(self.call_times) < self.qpm

    def _world(self, obs: Observation, goal: str, recent: list[str], hint: str) -> str:
        # The input universe is what the slow model prunes against; the fast model
        # may only pick from the current candidates, so showing it the full
        # keyboard would invite it to name a key that is not legal this step.
        fast_view = obs.state.get("fast_state") or obs.state
        if isinstance(fast_view, dict) and "input_universe" in fast_view:
            fast_view = {key: value for key, value in fast_view.items()
                         if key not in {"input_universe", "input_excluded", "input_excluded_note",
                                        "input_subset_history"}}
        state = json.dumps(fast_view, ensure_ascii=False)[:22000] if obs.state else "{}"

        history_lines = self.thread.history_digest().splitlines()
        history = "\n".join(history_lines[:1] + history_lines[-9:]) if len(history_lines)>10 else "\n".join(history_lines)
        if self.context_mode in {'compact','criteria'} and isinstance(obs.state.get('fast_state'),dict):
            # Candidate descriptions are already sent in criteria. Keep public
            # rules/current state/history once, without re-sending the briefing
            # and situation that the persistent slow instruction summarizes.
            compiled=self.context_mode=='criteria' and bool(self._compiled_criteria(obs))
            current=dict(obs.state['fast_state'])
            for internal in ('input_universe','input_excluded','input_excluded_note','input_subset_history'):
                current.pop(internal, None)
            if compiled:

                current.pop('control_instructions',None);current.pop('experience_hypotheses',None)
            return json.dumps({'system_lock':{'identity':self.identity,'goal':self.locked_goal or goal},
                'thread_history':history,'current_observation':current,'timing_contract':obs.state.get('timing_contract'),'per_action_usage_conditions_attached':compiled,
                'slow_plan':{'stage_goal':self.outer_stage_goal,'instruction':'' if compiled else self.outer_play_instruction or hint,
                             'age_decisions':self.steps-self.last_outer_step if self.last_outer_step else None},
                'recent_actions':recent[-8:],'recurrence':obs.state.get('recurrence',{}),
                'authority':'Current observation overrides historical coordinates/resources. Rank all supplied legal criteria; never invent an action.'},
                ensure_ascii=False,separators=(',',':'))
        labels = "; ".join(f"{action.id}={action.describe()}" for action in obs.actions)
        trail_text = " -> ".join(
            f"{previous}{action}{current}" for previous, action, current in self.trail[-8:]
        )
        counts = Counter(current for _previous, _action, current in self.trail)
        repeats = [f"{cell}x{count}" for cell, count in counts.items() if count > 1]
        return (
            f"System lock: {self.identity}. Locked goal: {self.locked_goal or goal}. "
            f"{history}\n"
            f"Game: {getattr(obs, 'name', '') or 'current game'}. Goal: {goal}. "
            f"Game briefing: {self.outer_briefing or 'none'}. "
            f"Stage goal: {self.outer_stage_goal or 'none'}. "
            f"Play instruction: {self.outer_play_instruction or hint or 'none'}. "
            f"Situation: {self.outer_situation or 'none'}. "
            f"Slow context age: {self.steps-self.last_outer_step if self.last_outer_step else 'not reviewed'} decisions. "
            "Revalidate concrete targets and resource values against the current observation. "
            f"Level: {obs.level}. State: {state}. "
            f"Trail: {trail_text or 'none'}. Visited cells: {len(self.visited)}. "
            f"Repeated cells: {', '.join(repeats) or 'none'}. "
            f"Recent actions: {', '.join(recent[-12:]) or 'none'}. "
            f"State recurrence evidence: {json.dumps(obs.state.get('recurrence',{}),ensure_ascii=False)}. "
            f"Candidates: {labels}. "
            "Choose the single legal action that advances the current stage goal. "
            "Use the visible state and action effects; reobserve after each action. "
            "Use the screenshot when present; do not invent an action."
        )

    @staticmethod
    def _images(obs: Observation) -> list[str]:
        """Send the observed frame to JEV as one compact JPEG.

        A raw 1920x1080 PNG screenshot is both slow to upload and wasteful in
        tokens. Re-encoding once keeps the whole frame visible while cutting
        bytes by an order of magnitude; anything already small is passed
        through untouched so a pack can supply its own crop if it wants.
        """
        raw = getattr(obs, "image", "") or ""
        if not raw.startswith("data:image/"):
            return [raw] if raw else []
        if raw.startswith("data:image/jpeg") and len(raw) < 400_000:
            return [raw]
        try:
            import base64 as _b64
            import io as _io

            from PIL import Image as _Image

            payload = raw.split(",", 1)[1]
            image = _Image.open(_io.BytesIO(_b64.b64decode(payload))).convert("RGB")
            longest = max(image.size)
            limit = 1400
            if longest > limit:
                scale = limit / longest
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    _Image.Resampling.LANCZOS,
                )
            buffer = _io.BytesIO()
            image.save(buffer, format="JPEG", quality=84, optimize=True)
            return ["data:image/jpeg;base64," + _b64.b64encode(buffer.getvalue()).decode()]
        except Exception:
            # A broken screenshot must never stop the episode; text-only still
            # lets JEV answer from the visible state.
            return [raw]

    @staticmethod
    def _control_scope(obs):
        return (obs.state.get('phase'),obs.level,tuple(sorted((action.id,action.kind,action.describe()) for action in obs.actions)))

    def _compiled_criteria(self,obs):
        if self.outer_criteria_scope==self._control_scope(obs) and set(self.outer_action_criteria)=={action.id for action in obs.actions}:
            return self.outer_action_criteria
        return {}

    def _criteria(self, obs: Observation, goal: str, recent: list[str], hint: str) -> dict[str, str]:
        criteria: dict[str, str] = {}
        compiled=self._compiled_criteria(obs)
        for action in obs.actions:
            if compiled:
                criteria[action.id]=f"{action.id}: {action.describe()}. DeepSeek usage condition: {compiled[action.id]}"
                continue
            if action.kind == "scroll":
                criteria[action.id] = (
                    f"Choose {action.id}: {action.label}. Correct only if scrolling is needed "
                    "to reach useful controls currently outside the visible area, such as a "
                    "continue, leave, purchase or confirmation control."
                )
                continue
            if action.kind == "wait":
                criteria[action.id] = (
                    "Choose wait ONLY while useful player controls are temporarily unavailable "
                    "because the environment is changing. Do not wait merely because an old log "
                    "describes an animation or an enemy action."
                )
                continue
            described = action.describe()
            # Action labels already carry their usage sentence, which ends in a
            # period. Appending ". " unconditionally produced "face it.." in the
            # criteria text the fast model reads.
            joiner = " " if described.rstrip().endswith((".", "!", "?")) else ". "
            criteria[action.id] = (
                f"{action.id}: {described}{joiner}"
                "Best next action under the current state and outer strategy."
            )
        return criteria

    def _stop(self, note: str, error: str = "") -> Decision:
        return Decision("", by="stop", note=note, error=error)

    def _call_escalate(self, escalate, obs, goal, decision, recent):
        try:
            return escalate(obs, goal, decision, recent=recent)
        except TypeError:
            return escalate(obs, goal, decision)

    def _accept_outer_result(self, obs: Observation, escalate, action: str | None) -> None:
        """Publish every slow response as one current context/cache revision."""
        self.outer_hint = str(getattr(escalate, "hint", "") or "")[:900]
        self.outer_situation = str(getattr(escalate, "situation", "") or "")[:500]
        self.outer_briefing = str(getattr(escalate, "briefing", "") or "")[:1500]
        self.outer_stage_goal = str(getattr(escalate, "stage_goal", "") or "")[:600]
        self.outer_play_instruction = str(getattr(escalate, "play_instruction", "") or "")[:900]
        raw=getattr(escalate,'action_criteria',{})
        legal={item.id for item in obs.actions}
        self.outer_action_criteria=dict(raw) if isinstance(raw,dict) and set(raw)==legal and all(isinstance(value,str) and 0<len(value.strip())<=500 for value in raw.values()) else {}
        self.outer_criteria_scope=self._control_scope(obs) if self.outer_action_criteria else None
        # The slow model's keyboard exclusions are applied here, once per review,
        # rather than being merged into the candidate list. Narrowing the action
        # space is a standing decision about the game, so it must survive the
        # next observation instead of being recomputed from the live tier. Only
        # an accepted verdict is applied: an unsure empty answer must not erase a
        # subset that earlier evidence established.
        exclude=getattr(escalate,'control_exclude',None)
        status=getattr(escalate,'control_exclude_status',{}) or {}
        self.subset_status=dict(status)
        if (status.get('applied') and isinstance(exclude,list) and self.subset_sink is not None
                and obs.state.get('input_universe')):
            # The adapter is the only layer that knows the physical-key aliases
            # and how small a usable keyboard may get, so its verdict decides
            # whether the subset is really in force. Discarding the return value
            # used to leave the status reading "applied" while nothing was
            # excluded, which made a refused verdict look like a silent one.
            try:
                outcome=self.subset_sink(exclude, note=str(getattr(escalate,'situation','') or ''))
            except Exception as exc:
                outcome={'ok':False,'reason':f'subset sink failed: {type(exc).__name__}'}
            if isinstance(outcome,dict):
                self.subset_status['sink']=outcome
                escalate.control_exclude_status=dict(status,
                    applied=bool(outcome.get('ok')),
                    reason=str(outcome.get('reason') or status.get('reason') or ''),
                    kept=int(outcome.get('kept') or len(outcome.get('exclude') or [])))
                self.subset_note=str(outcome.get('reason') or '')


        self.outer_action = (

            action
            if action and any(item.id == action for item in obs.actions)
            else None
        )
        if self.outer_action:
            self.commit_action = self.outer_action
            self.commit_steps = 0
        self.last_outer_cell = obs.state.get("cell")
        self.last_outer_phase = (obs.state.get("phase"), tuple(obs.state.get("headings") or []))
        self.last_outer_signature = obs.raw.get("signature")
        self.last_outer_level = obs.level
        self.last_outer_step = self.steps
        self.steps_since_outer = 0

    def _ask_outer(
        self,
        obs: Observation,
        goal: str,
        recent: list[str],
        escalate: Escalate | None,
    ) -> str | None:
        if escalate is None:
            return None
        action = None
        try:
            action = self._call_escalate(
                escalate,
                obs,
                goal,
                Decision("", by="outer", probabilities={}),
                recent,
            )
        except Exception as exc:
            return str(exc)[:240] or type(exc).__name__
        self._accept_outer_result(obs,escalate,action)
        return None

    def _escalate(
        self,
        obs: Observation,
        goal: str,
        decision: Decision,
        escalate: Escalate | None,
    ) -> Decision:
        if (self.outer_action and any(action.id == self.outer_action for action in obs.actions)
                and (not self.last_outer_signature or obs.raw.get("signature") == self.last_outer_signature)):
            return Decision(
                self.outer_action,
                by="escalate",
                probabilities=decision.probabilities,
                model=decision.model,
                image_count=decision.image_count,
                ms=decision.ms,
                usage=decision.usage,
                note=f"outer strategy: {self.outer_situation or self.outer_hint}",
            )
        if escalate is not None:
            try:
                action_id = self._call_escalate(escalate, obs, goal, decision, [])
            except Exception as exc:
                return self._stop("outer model unavailable", str(exc)[:240] or type(exc).__name__)
            self._accept_outer_result(obs,escalate,action_id)
            if getattr(escalate, "error", ""):
                return self._stop("outer model unavailable", escalate.error)
            if action_id and any(action.id == action_id for action in obs.actions):
                return Decision(
                    action_id,
                    by="escalate",
                    probabilities=decision.probabilities,
                    model=decision.model,
                    image_count=decision.image_count,
                    ms=decision.ms,
                    usage=decision.usage,
                    note=f"outer strategy: {self.outer_situation or self.outer_hint}",
                )
        legal = {action.id for action in obs.actions}
        ranked = sorted(
            (
                (name, score)
                for name, score in decision.probabilities.items()
                if name in legal and name != "wait"
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            name, score = ranked[0]
            return Decision(
                name,
                by="jev-low",
                probabilities=decision.probabilities,
                model=decision.model,
                image_count=decision.image_count,
                ms=decision.ms,
                usage=decision.usage,
                note=f"outer empty; JEV non-wait top {name} {score:.3f}",
            )
        return self._stop("outer returned no legal action and JEV has no non-wait action")

    def record_execution(self, obs, decision, action, execution) -> None:
        """Feed actual observe/decide/execute turns to both model layers."""
        turn = self.thread.begin_turn(self.steps)
        state = {k: v for k, v in prompt_state(obs.state).items() if k not in {"instructions", "last_execution", "observed_scenes"}}
        if isinstance(state.get("screen"), str):
            state["screen"] = state["screen"][:1800]
        turn.add("observation", json.dumps({"level": obs.level, "state": state}, ensure_ascii=False))
        turn.add("agent_message", "Recorded executed game decision (historical, not a new strategy response): " + json.dumps({"action": action.id, "label": action.label,
                 "by": decision.by, "note": decision.note}, ensure_ascii=False))
        turn.add("tool_result", json.dumps({k: v for k, v in execution.items() if k != "after_text"}, ensure_ascii=False))
        self.thread.complete_turn()

    def choose(
        self,
        obs: Observation,
        goal: str,
        recent: list[str] | None = None,
        hint: str = "",
        escalate: Escalate | None = None,
    ) -> Decision:
        recent = list(recent or [])
        if self.locked_goal is None:
            self.locked_goal = goal
        goal = self.locked_goal
        if obs.done:
            return self._stop("episode already done")
        if not obs.actions:
            return self._stop("no legal actions")
        if len(obs.actions) > self.max_candidates:
            return self._stop(
                f"{len(obs.actions)} candidates > {self.max_candidates}; adapter must prune"
            )
        cell_raw = obs.state.get("cell")
        if isinstance(cell_raw, (list, tuple)) and len(cell_raw) == 2:
            current = (int(cell_raw[0]), int(cell_raw[1]))
            if self.last_cell is not None and self.last_action:
                self.trail.append((self.last_cell, self.last_action, current))
                self.trail = self.trail[-12:]
            self.visited.add(current)
            self.last_cell = current
        self.last_action = recent[-1] if recent else ""
        if not self._budget_ok():
            return self._escalate(
                obs, goal, self._stop("local JEV budget exhausted"), escalate
            )

        self.steps += 1
        obs.state['recurrence']=self.recurrence.observe(obs.state,self.steps)
        self.steps_since_outer += 1
        cell = obs.state.get("cell")
        cell_changed = cell is not None and cell != self.last_outer_cell
        if escalate is not None and (
            self.steps == 1
            or obs.state['recurrence']['visits_in_window'] >= 3
            or self.steps_since_outer >= self.call_outer_every
            or (obs.state.get("phase"), tuple(obs.state.get("headings") or [])) != self.last_outer_phase
            or obs.level != self.last_outer_level
            or (cell_changed and self.steps_since_outer >= 2)
            or (len(recent) >= 3 and len(set(recent[-3:])) == 1)
            or not self.outer_hint
        ):
            outer_error = self._ask_outer(obs, goal, recent, escalate)
            if outer_error or getattr(escalate, "error", ""):
                # A failed review used to end the episode outright, and because the
                # review fires on step 1 and whenever no hint is cached yet, a
                # single empty response threw away the whole trajectory (measured:
                # 4 steps instead of 37). The review is an optimisation, not a
                # precondition: keep playing on the fast model and retry the
                # review on its next trigger. Only a sustained outage stops the
                # run, so a dead planner cannot spin forever.
                self.outer_failures = getattr(self, "outer_failures", 0) + 1
                self.steps_since_outer = 0
                if self.outer_failures >= 6:
                    return self._stop(
                        "outer model unavailable",
                        outer_error or getattr(escalate, "error", ""),
                    )
        fresh = str(obs.state.get("hint") or "")
        stage_hint = self.outer_play_instruction or self.outer_hint or hint
        hint = f"{stage_hint}. Fresh explorer: {fresh}" if fresh else stage_hint

        criteria = self._criteria(obs, goal, recent, hint)
        world = self._world(obs, goal, recent, hint)
        images = self._images(obs)
        self.call_times.append(time.time())
        started = time.time()
        scores: dict = {}
        # The fast scorer runs on a single SystemOne model, so a burst of
        # collectors pushes it into a 429 cooldown and every call in that window
        # comes back with an empty probability table. Treating that as a real
        # failure kills the whole episode after one step, which is how a 38-way
        # sweep turned 30-sample episodes into 3-sample ones. A rate limit is a
        # temporary condition, so wait it out and ask again before giving up.
        for attempt in range(RATE_LIMIT_RETRIES):
            try:
                scores = self.score_fn(
                    criteria,
                    self.token,
                    {"action": tuple(action.id for action in obs.actions)},
                    memory=world,
                    images=images,
                ) or {}
            except Exception as exc:
                decision = self._stop("JEV call failed", str(exc)[:200])
                return self._escalate(obs, goal, decision, escalate)
            if (scores.get("probabilities") or {}) or not _is_rate_limited(scores):
                break
            if attempt + 1 < RATE_LIMIT_RETRIES:
                time.sleep(RATE_LIMIT_WAIT)


        ms = int((time.time() - started) * 1000)


        probabilities = {
            str(key): float(value)
            for key, value in (scores.get("probabilities") or {}).items()
            if isinstance(value, (int, float))
        }
        decision = Decision(
            "",
            by="jev",
            probabilities=probabilities,
            model=str(scores.get("model") or ""),
            image_count=int(scores.get("image_count") or 0),
            ms=ms,
            usage=scores.get("usage") or {},
            error=str(scores.get("error") or "")[:240],
        )
        if decision.error or not probabilities:
            return self._escalate(obs, goal, decision, escalate)

        legal = {action.id for action in obs.actions}
        ranked = sorted(
            ((name, score) for name, score in probabilities.items() if name in legal),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            return self._stop("JEV returned no probability for a legal action")
        if self.commit_action and self.commit_steps > 0:
            if any(action.id == self.commit_action for action in obs.actions):
                self.commit_steps -= 1
                return Decision(
                    self.commit_action,
                    by="commit",
                    probabilities=probabilities,
                    model=decision.model,
                    image_count=decision.image_count,
                    ms=decision.ms,
                    usage=decision.usage,
                    note=f"commit {self.commit_action} ({self.commit_steps} left)",
                )
            self.commit_steps = 0
        top_id, top = ranked[0]
        decision.action_id = top_id
        decision.note = f"top {top_id} {top:.3f}"
        if top < self.confidence:
            decision.by = "low"
            decision.note = f"top {top_id} {top:.3f} < confidence {self.confidence:.2f}"
            return self._escalate(obs, goal, decision, escalate)
        return decision
