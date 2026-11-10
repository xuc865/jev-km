from copy import deepcopy
import json
import unittest
from general_agent.policy import JevPolicy
from general_agent.types import Action, Observation, Decision


class CompactFastContextTests(unittest.TestCase):
    def test_compact_context_keeps_rules_current_state_thread_history_and_locked_goal(self):
        policy=JevPolicy(context_mode='compact',score_fn=lambda *args,**kwargs:{})
        policy.locked_goal='original root';policy.outer_stage_goal='stage';policy.outer_play_instruction='conditional current plan'
        historical=Observation(actions=[],state={'phase':'play','fast_state':{'hp':10}},level=1)
        action=Action('move','Move normally',keys=('ArrowRight',))
        policy.record_execution(historical,Decision('move',by='jev'),action,{'ok':True,'changed':True})
        current=Observation(actions=[action],state={'phase':'play','fast_state':{'hp':4,'control_instructions':'Original public control rules'}},level=1)
        before=deepcopy(current.state)
        world=json.loads(policy._world(current,'replacement goal',[],'ignored'))
        self.assertEqual(world['system_lock']['goal'],'original root')
        self.assertEqual(world['current_observation']['hp'],4)
        self.assertIn('Original public control rules',world['current_observation']['control_instructions'])
        self.assertIn('move',world['thread_history'])
        self.assertEqual(current.state,before)

    def test_compact_mode_does_not_prune_or_duplicate_legal_candidates(self):
        policy=JevPolicy(context_mode='compact',score_fn=lambda *args,**kwargs:{})
        actions=[Action('a'+str(i),'Control '+str(i))for i in range(27)]
        observation=Observation(actions=actions,state={'fast_state':{'phase':'play'}},level=1)
        self.assertEqual(set(policy._criteria(observation,'goal',[],'')),{action.id for action in actions})
        world=json.loads(policy._world(observation,'goal',[],''))
        self.assertNotIn('candidates',world)
