import unittest
from general_agent.policy import JevPolicy
from general_agent.types import Action,Observation

class LayeredOuter:
    error=''
    def __init__(self):self.count=0
    def __call__(self,obs,goal,decision,**kwargs):
        self.count+=1
        self.hint='hint-'+str(self.count)
        self.briefing='brief-'+str(self.count)
        self.stage_goal='stage-'+str(self.count)
        self.play_instruction='instruction-'+str(self.count)
        self.situation='situation-'+str(self.count)
        return 'go' if self.count==1 else 'back'

class ScoreSequence:
    def __init__(self,values):self.values=iter(values);self.memories=[]
    def __call__(self,criteria,token,tracks,memory='',images=None):
        self.memories.append(memory);prob=next(self.values)
        return {'model':'test-jev','probabilities':{'go':prob,'back':1-prob}}

def observation(signature):
    return Observation(actions=[Action('go'),Action('back')],
                       state={'fast_state':{'position':signature}},raw={'signature':signature})

class SlowContextSyncTests(unittest.TestCase):
    def test_low_confidence_update_is_visible_to_the_next_fast_step(self):
        scorer=ScoreSequence([.95,.55,.95]);outer=LayeredOuter()
        policy=JevPolicy(score_fn=scorer);policy.call_outer_every=100
        policy.choose(observation('A'),'root goal',escalate=outer)
        second=policy.choose(observation('B'),'root goal',escalate=outer)
        self.assertEqual(second.action_id,'back')
        self.assertEqual(policy.outer_stage_goal,'stage-2')
        self.assertEqual(policy.outer_play_instruction,'instruction-2')
        self.assertEqual(policy.last_outer_signature,'B')
        self.assertEqual(policy.steps_since_outer,0)
        policy.choose(observation('C'),'root goal',escalate=outer)
        self.assertEqual(outer.count,2)
        self.assertIn('instruction-2',scorer.memories[-1])
        self.assertNotIn('instruction-1',scorer.memories[-1])
        self.assertIn('root goal',scorer.memories[-1])

    def test_returning_to_an_old_observation_cannot_reuse_a_superseded_directive(self):
        scorer=ScoreSequence([.95,.55,.55]);outer=LayeredOuter()
        policy=JevPolicy(score_fn=scorer);policy.call_outer_every=100
        policy.choose(observation('A'),'goal',escalate=outer)
        policy.choose(observation('B'),'goal',escalate=outer)
        decision=policy.choose(observation('A'),'goal',escalate=outer)
        self.assertEqual(outer.count,3)
        self.assertEqual(decision.action_id,'back')
        self.assertEqual(policy.outer_play_instruction,'instruction-3')
