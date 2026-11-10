"""Remember changes in pack-declared public outcome fields; never choose actions."""
from collections import deque
from copy import deepcopy


class ObservedProgress:
    def __init__(self, fields=None):
        self.fields=list(fields or [])
        self.changes=deque(maxlen=16)
        self.current=None
        self.last_change_step=None

    def observe(self, state, step):
        values={}
        for path in self.fields:
            value=state
            for key in path.split('.'):
                if not isinstance(value,dict) or key not in value:break
                value=value[key]
            else:values[path]=deepcopy(value)
        if not values:return {}
        if values!=self.current:
            self.current=values
            self.last_change_step=step
            self.changes.append({'step':step,'values':deepcopy(values)})
        return {'current':deepcopy(values),'last_change_step':self.last_change_step,
                'decisions_since_change':max(0,step-self.last_change_step),
                'recent_changes':deepcopy(list(self.changes)[-8:]),
                'note':'Public outcome-field changes only. Unchanged outcomes can include a legitimate long rally or transition; they neither prove failure nor choose a strategy.'}
