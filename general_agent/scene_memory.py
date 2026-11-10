"""Previously observed public scenes, retained as facts rather than a planner."""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
import json


def read_field(value,path):
    for part in path.split('.'):
        if not isinstance(value,dict):return None
        value=value.get(part)
    return value


class ObservedScenes:
    def __init__(self,config=None):
        self.config=config or {}
        self.scenes=OrderedDict()

    def remember(self,state,step):
        fields=self.config.get('scene_fields',[])
        if not fields:return []
        scene={name:read_field(state,name) for name in fields}
        if any(value is None for value in scene.values()):return []
        key=json.dumps(scene,sort_keys=True,ensure_ascii=False)
        required=self.config.get('require_fields',[])
        if all(read_field(state,name) is not None for name in required):
            snapshot={name:deepcopy(read_field(state,name)) for name in self.config.get('snapshot_fields',[])}
            self.scenes[key]={'scene':scene,'observed_at_step':step,'snapshot':snapshot,
                             'status':'Historical observation; resources, availability and estimates may be stale. Current observation is authoritative.'}
            self.scenes.move_to_end(key)
            while len(self.scenes)>max(1,int(self.config.get('max_scenes',32))):self.scenes.popitem(last=False)
        return deepcopy([item for old_key,item in self.scenes.items() if old_key!=key])
