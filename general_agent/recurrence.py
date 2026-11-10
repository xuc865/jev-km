"""Bounded evidence of state recurrence; never an action selector."""
from __future__ import annotations
from collections import deque
import hashlib
import json
from .context import prompt_state


class RecurrenceTracker:
    def __init__(self, window=256):
        self.observations=deque(maxlen=window)

    def observe(self, state, step):
        visible=state.get('fast_state') or prompt_state(state)
        # `last_action` / `last_execution` describe what the agent just did, not
        # what the screen shows. Hashing them makes every step look like a brand
        # new state, so recurrence never fires and the model is never told it has
        # been staring at the same frame for ten turns.
        visible={key:value for key,value in visible.items()
                 if key not in {'instructions','control_instructions','experience_hypotheses',
                                'last_execution','last_action','recurrence','progress_evidence'}}
        key=hashlib.sha256(json.dumps(visible,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
        previous=[old_step for old_key,old_step in self.observations if old_key==key]
        self.observations.append((key,step))
        return {'observation_key':key,'visits_in_window':len(previous)+1,
                'matching_steps':previous[-7:]+[step],
                'note':'Matching public states are recurrence evidence. They do not by themselves prove failure or choose an action.'}
