import unittest
from general_agent.runner import run_episode
from general_agent.policy import JevPolicy
from general_agent.types import Action,Observation

class StopTests(unittest.TestCase):
    def test_stop_during_inference_does_not_execute_the_planned_input(self):
        stop=[False];actions=[];closed=[]
        class Adapter:
            def reset(self):pass
            def observe(self):return Observation(actions=[Action('play')])
            def act(self,action):actions.append(action.id)
            def close(self):closed.append(True)
        def score(*args,**kwargs):stop[0]=True;return {'probabilities':{'play':1.0}}
        result=run_episode(Adapter(),JevPolicy(score_fn=score),'win',should_stop=lambda:stop[0])
        self.assertTrue(result.stopped)
        self.assertEqual(actions,[])
        self.assertTrue(closed)
