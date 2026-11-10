"""DeepSeek outer escalation for the general JEV harness.

The JEV is the fast multimodal inner. When it is unsure, the outer model sees
the current observation, legal candidates and state, then returns either one
legal action or a short hint. If it cannot decide, the harness stops; there is
no hand-coded game fallback.
"""

from __future__ import annotations

import json
import os
import re
import time

from agent_harness import Thread

from .types import Decision, Observation
from .context import prompt_state

SYSTEM = """You are the outer strategist of a game agent. The inner multimodal JEV
chooses one legal action per decision; you receive the full legal action list,
the current public state, recent actions and the JEV probabilities. A screenshot
is attached only when image_attached is true; otherwise reason from the supplied
public observation and do not assume an unseen image.
Before the JEV acts, give it a compact layered context:
1. briefing: what this game is, its purpose/background, controls/gameplay, and
   the win condition. Keep it stable across reviews.
2. stage_goal: the objective of the current phase/level.
3. play_instruction: concrete steps for the JEV for this stage.
4. situation: one sentence on the current state/risk.
5. action: optional exactly one legal action id, or empty.
On later reviews, update stage_goal and play_instruction from the current
observation and recent actions. The JEV still picks each legal action under your
instruction. Never invent an action id.
For persistent play_instruction, prefer conditions referring to current observation
fields; put a one-time decision in action. Concrete coordinates and resource values
from this review may be stale on later fast steps and must be revalidated.
The current visible observation and currently available controls are authoritative.
observed_scenes contains only historical public observations, never future scenes.
Its resources, available flags and battle estimates were true when observed and
may be stale under current resources. Use it as factual planning context; only
current legal_actions can be executed. Do not assume revisiting a scene resets it.
A visible navigation target may be resource_eligible yet not executable_now because
other interactions block its path. Distinguish map visibility from current reachability;
do not assume changing floors refreshes legal paths or bypasses blockers.
Control roles distinguish selection, confirmation, switching and cancellation.
A delivered input does not imply the intended game effect happened; compare the new state.
progress_evidence tracks only pack-declared public outcome fields. If these remain unchanged across many decisions despite successful inputs, reconsider whether the current stage plan advances the locked root goal and compare alternatives. An unchanged score can still be a legitimate long rally; reassess rather than automatically abandoning it.
Earlier strategy, resources and event logs may be outdated; revise them whenever
current evidence differs. Do not mistake a historical event log for the current phase.
Reconstruct resources and playable options from the current observation on every review.
When recurrence evidence shows matching public states after repeated actions, reevaluate
the assumed effects and candidate tradeoffs. A delivered input or a return to an earlier
state does not establish that a new route opened. Recurrence alone is not a reason to
forbid backtracking, waiting or any otherwise legal action.
Do not copy previously listed cards, energy, health or floor numbers into the new strategy.
Only recommend controls present in the current legal_actions. If no useful control is
available during a transition, wait and inspect the next observation.
Return JSON only:
{"briefing":"...", "stage_goal":"...", "play_instruction":"...",
 "situation":"...", "action":"optional legal id or empty"}
"""


class OuterEscalation:
    """Callable escalation for JevPolicy; returns one legal action id or None."""

    def __init__(self, token: str = "", model: str | None = None, chat_fn=None,
                 thread: Thread | None = None, identity: str = "Autonomous game player", generation: dict | None = None,
                 generate_criteria: bool = False) -> None:
        if not token:
            try:
                import harness

                token = str(getattr(harness, "TOKEN", "") or "")
            except Exception:
                token = ""
            token = token or (os.environ.get("JEVGAMER_TOKEN") or "")
        if not isinstance(generate_criteria,bool):raise ValueError("generate_criteria must be boolean")
        self.generate_criteria=generate_criteria
        self.action_criteria={}
        self.criteria_status={}
        # Slow model's claim about which keyboard inputs this game ignores. It is
        # validated against the observed universe before the adapter applies it.
        self.control_exclude: list[str] = []
        self.control_exclude_status: dict = {}

        self.token = token
        self.generation = dict(generation or {})
        if set(self.generation)-{'thinking','max_output_tokens'}:
            raise ValueError("Unsupported slow-model generation setting")
        # The default slow model is a reasoning model. Left on, it spent 4785
        # characters of hidden reasoning and ~1250 completion tokens on a decision
        # that only needs a small JSON object, which cost 10.6s per review and
        # occasionally consumed the whole budget so `content` came back empty and
        # the episode died with "outer returned no valid JSON". Turning thinking
        # off measured 142 completion tokens in ~1s with the JSON still valid, so
        # it is the default here; JEVGAMER_SLOW_THINKING=1 restores the old
        # behaviour for models that need to reason.
        if "thinking" not in self.generation:
            configured = (os.environ.get("JEVGAMER_SLOW_THINKING") or "").strip()
            if configured in {"0", "1"}:
                self.generation["thinking"] = configured == "1"
            else:
                self.generation["thinking"] = False
        self.thread = thread or Thread(history_turns=4)
        self._own_thread = thread is None
        self.identity = identity
        self.locked_goal: str | None = None
        self.model = model or os.environ.get("JEVGAMER_SUPERVISOR") or "deepseek-v4.1-flash"
        self.hint = ""
        self.situation = ""
        self.briefing = ""
        self.stage_goal = ""
        self.play_instruction = ""
        self._chat = chat_fn
        self.calls: list[dict] = []
        self.error = ""

    def _chat_fn(self):
        if self._chat is not None:
            return self._chat
        from harness import chat

        return chat

    @staticmethod
    def _parse(text: str) -> dict:
        match = re.search(r"\{.*\}", text or "", re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def __call__(
        self,
        obs: Observation,
        goal: str,
        decision: Decision | None = None,
        recent: list[str] | None = None,
    ) -> str | None:
        if self.generate_criteria:
            self.action_criteria={}
            self.criteria_status={'complete':False,'reason':'awaiting valid response'}
        if self.locked_goal is None:
            self.locked_goal = goal
        goal = self.locked_goal
        actions = [
            {
                "id": action.id,
                "label": action.label or action.id,
                "kind": action.kind,
                "role": action.params.get("semantic_role", ""),
                "seconds": action.seconds,
            }
            for action in obs.actions
        ]
        payload = {
            "previous_strategy": {"briefing": self.briefing, "stage_goal": self.stage_goal,
                                  "play_instruction": self.play_instruction},
            "goal": goal,
            "level": obs.level,
            "image_attached": bool(obs.image),
            "state": prompt_state(obs.state),
            "legal_actions": actions,
            "jev_top": decision.action_id if decision else "",
            "jev_note": decision.note if decision else "",
            "jev_probabilities": decision.probabilities if decision else {},
            "recent_actions": list(recent or []),
        }
        content: list[dict] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        if obs.image:
            content.append({"type": "image_url", "image_url": {"url": obs.image}})
        system = SYSTEM + "\nImmutable player identity: " + self.identity + "\nLocked root goal: " + goal
        if self.generate_criteria:
            system += """
Also return action_criteria: an object mapping EVERY current legal action ID to a reusable usage condition (1-500 characters each). Conditions should refer to CURRENT public observation fields and declared input semantics, not fixed coordinates/resource values from this review. The fast model will match these conditions against later observations of the same control schema. Preserve the root goal; never invent controls, hidden information or automatic game policies. Include all IDs, even currently unattractive choices; do not prune candidates. Explain when each candidate is appropriate rather than marking only this instant's chosen action as correct. Distinguish alternatives, including duration variants, by their expected contribution to the locked goal rather than merely repeating key bindings or maintaining activity. This is model context, not executable code.
When state.input_universe is present, also return control_exclude: an array of input IDs from that universe which this game demonstrably does not use, judged from the whole screenshot, visible text and what the recent actions failed to change. The universe lists only IDs; move_up/move_down/move_left/move_right are the four arrow keys, the *_alt ones are WASD, confirm is Space, enter is Enter, escape is Escape, restart is R, key_<letter> is that letter key, digit_<n> is that number key, key_tab and key_backspace are the menu keys. The currently offered legal_actions are only the tier that happens to be live, so an exclusion must be a claim about this game, not about the current tier. Exclude a key only when the evidence says the game ignores it: the key is described as a menu/traversal input on a screen with no menu, or the game's own visible instructions name other keys and this one is unrelated. Never exclude a key merely because it has not been tried yet, and never exclude an input that the visible instructions or on-screen hints ask the player to press. Always keep the keys the game is currently responding to. If the evidence is insufficient, return an empty array: an unnecessary exclusion costs more than a missing one, because the fast model is left unable to press the key the game needs. Do not exclude click_* or DOM controls; this field is keyboard-only. Keep the total exclusion small and justified, and if a previously excluded key now looks required, report the corrected full exclusion list instead of leaving it out. When state.input_excluded is non-empty and you now believe that pruning was wrong, return control_exclude as an empty array together with "control_exclude_clear": true; an empty array alone is read as "not enough evidence yet" and the cached subset is kept."""

        system += ("\nExcluding one id also excludes any other id that sends the same "
                   "physical key, because otherwise the game would still receive it "
                   "through the alias."
                   " state.input_subset_history shows the last verdicts and whether the harness accepted them: an entry with \"rejected\": true carries the reason in \"note\". Read it before answering. A rejected verdict changed nothing, so do not resubmit the same list; either narrow it to the keys the evidence actually rules out, or return an empty array. A game may legitimately need only one or two keys (holding Space to shield, a single rotation key): that is a valid verdict and must be reported, not padded with keys kept for appearance.")
        messages = self.thread.messages_for_prompt(system, content)
        started = time.monotonic()
        response = self._chat_fn()(messages, self.token, model=self.model, **self.generation)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self.error = str(response.get("_error") or "")
        if self.error:
            self.calls.append({"model":response.get("model") or "", "requested_model":self.model,
                "goal":goal,"generation":dict(self.generation),"image_count":int(bool(obs.image)),"screen":payload["state"],"ms":elapsed_ms,"recent":list(recent or []),
                "error":self.error,"status":response.get("_status"),
                "transport_attempts":response.get("_transport_attempts",1),
                "transport_failures":response.get("_transport_failures",[])})
            raise RuntimeError("outer model failed: " + self.error[:200])
        choices = response.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "")
        if self._own_thread:
            turn = self.thread.begin_turn(len(self.calls) + 1)
            turn.add("observation", json.dumps(payload, ensure_ascii=False))
            turn.add("agent_message", text)
            self.thread.complete_turn()
        parsed = self._parse(text)
        self.calls.append({"model": response.get("model") or self.model, "goal": goal,
                           "screen": payload["state"], "image_count":int(bool(obs.image)), "generation":dict(self.generation), "ms": elapsed_ms, "recent": list(recent or []),
                           "response": text, "usage": response.get("usage") or {},
                           "transport_attempts":response.get("_transport_attempts",1),
                           "transport_failures":response.get("_transport_failures",[])})
        if not parsed:
            self.error = "outer returned no valid JSON"
            return None
        self.action_criteria={}
        self.control_exclude=[]
        self.control_exclude_status={}
        if self.generate_criteria:
            legal={action.id for action in obs.actions}
            raw=parsed.get('action_criteria')
            valid={key:value.strip() for key,value in raw.items() if key in legal
                   and isinstance(value,str) and 0<len(value.strip())<=500} if isinstance(raw,dict) else {}
            complete=bool(legal) and set(valid)==legal
            self.action_criteria=valid if complete else {}
            self.criteria_status={'complete':complete,'missing_ids':sorted(legal-set(valid)),
                                  'ignored_ids':sorted(set(raw)-legal) if isinstance(raw,dict) else []}
            self.calls[-1]['criteria_status']=dict(self.criteria_status)
            universe=[str(item) for item in (obs.state.get('input_universe') or [])
                      if isinstance(item,str) and item]
            requested=parsed.get('control_exclude')
            clear=parsed.get('control_exclude_clear') is True
            if not isinstance(requested,list):
                self.control_exclude_status={'applied':False,'reason':'no control_exclude list'}
            else:
                wanted=[str(item) for item in requested]
                known={item for item in wanted if item in universe}
                unknown=[item for item in wanted if item not in universe]
                # An exclusion outside the universe is a model error, but it is
                # also the only signal that the model misread the input list, so
                # the whole response is refused instead of silently dropping the
                # bad entries and applying a half-checked subset.
                if unknown:
                    self.control_exclude=[]
                    self.control_exclude_status={'applied':False,'requested':len(wanted),
                                                 'kept':0,'unknown_ids':unknown[:12],
                                                 'reason':'ids outside input_universe'}
                elif known:
                    self.control_exclude=sorted(known)
                    self.control_exclude_status={'applied':True,'requested':len(wanted),
                                                 'kept':len(self.control_exclude),
                                                 'unknown_ids':[],'reason':''}
                elif clear:
                    # Only an explicit clear flag removes a previously verified
                    # subset. An empty array is indistinguishable from "not enough
                    # evidence yet", and erasing a cached answer on that signal
                    # would throw away the game-level decision on every review
                    # that happened to be unsure.
                    self.control_exclude=[]
                    self.control_exclude_status={'applied':True,'requested':0,'kept':0,
                                                 'unknown_ids':[],'reason':'cleared by model'}
                else:
                    self.control_exclude=[]
                    self.control_exclude_status={'applied':False,'requested':0,'kept':0,
                                                 'unknown_ids':[],
                                                 'reason':'empty array without clear flag; previous subset retained'}
            self.calls[-1]['control_exclude_status']=dict(self.control_exclude_status)


        self.briefing = str(parsed.get("briefing") or self.briefing or "")[:1500]
        self.stage_goal = str(parsed.get("stage_goal") or self.stage_goal or "")[:600]
        self.play_instruction = str(parsed.get("play_instruction") or parsed.get("note") or self.play_instruction or "")[:900]
        self.situation = str(parsed.get("situation") or parsed.get("note") or self.situation or "")[:500]
        self.hint = str(parsed.get("hint") or self.play_instruction or "")[:900]
        action = str(parsed.get("action") or "").strip()
        return action or None
