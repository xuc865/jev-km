import json
from types import SimpleNamespace
import unittest
from general_agent.outer import OuterEscalation
from general_agent.policy import JevPolicy
from general_agent.types import Action,Observation,Decision

class ModelCriteriaTests(unittest.TestCase):
    def test_outer_accepts_complete_model_map_and_clears_it_on_a_new_incomplete_revision(self):
        answers=[{'action':'a','action_criteria':{'a':'When current value is positive','b':'When current value is zero','unknown':'Never execute invented controls'}},
                 {'action':'a','action_criteria':{'a':'new condition'}},
                 {'action':'a','action_criteria':{'a':'A','b':'B'}},'not json']
        messages=[]
        def chat(prompt,token,**kwargs):
            messages.append(prompt)
            return {'choices':[{'message':{'content':json.dumps(answers.pop(0))}}]}
        outer=OuterEscalation(chat_fn=chat,generate_criteria=True)
        obs=Observation(actions=[Action('a'),Action('b')],state={'phase':'play'})
        self.assertEqual(outer(obs,'locked goal'),'a')
        self.assertEqual(set(outer.action_criteria),{'a','b'})
        self.assertIn('locked goal',messages[0][0]['content'])
        self.assertTrue(outer.calls[-1]['criteria_status']['complete'])
        outer(obs,'replacement goal')
        self.assertEqual(outer.action_criteria,{})
        self.assertEqual(outer.locked_goal,'locked goal')
        outer(obs,'replacement goal');self.assertTrue(outer.action_criteria)
        outer(obs,'replacement goal');self.assertEqual(outer.action_criteria,{})

    def test_fast_criteria_keep_every_action_and_expire_when_control_meaning_changes(self):
        policy=JevPolicy(score_fn=lambda *args,**kwargs:{})
        obs=Observation(actions=[Action('a','Original choice'),Action('b','Other choice')],state={'phase':'play'})
        slow=SimpleNamespace(action_criteria={'a':'MODEL-A','b':'MODEL-B'})
        policy._accept_outer_result(obs,slow,None)
        criteria=policy._criteria(obs,'goal',[],'')
        self.assertEqual(set(criteria),{'a','b'})
        self.assertIn('MODEL-A',criteria['a'])
        changed=Observation(actions=[Action('a','Different choice'),Action('b','Other choice')],state={'phase':'play'})
        self.assertNotIn('MODEL-A',policy._criteria(changed,'goal',[],'')['a'])
        new_level=Observation(actions=obs.actions,state=obs.state,level=2)
        self.assertNotIn('MODEL-A',policy._criteria(new_level,'goal',[],'')['a'])
        policy._accept_outer_result(obs,SimpleNamespace(action_criteria={}),None)
        self.assertNotIn('MODEL-A',policy._criteria(obs,'goal',[],'')['a'])

    def test_slim_matching_context_requires_complete_current_model_conditions(self):
        policy=JevPolicy(context_mode='criteria',score_fn=lambda *args,**kwargs:{})
        policy.locked_goal='root goal'
        obs=Observation(actions=[Action('a')],state={'phase':'play','fast_state':{'value':7,'control_instructions':'Full public control rules'}})
        before=json.loads(policy._world(obs,'other goal',[],''))
        self.assertIn('control_instructions',before['current_observation'])
        policy._accept_outer_result(obs,SimpleNamespace(action_criteria={'a':'Use current value 7'}),None)
        after=json.loads(policy._world(obs,'other goal',[],''))
        self.assertTrue(after['per_action_usage_conditions_attached'])
        self.assertEqual(after['current_observation']['value'],7)
        self.assertEqual(after['system_lock']['goal'],'root goal')
        self.assertEqual(set(policy._criteria(obs,'goal',[],'')),{'a'})
