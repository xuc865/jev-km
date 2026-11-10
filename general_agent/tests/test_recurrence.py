import unittest
from general_agent.recurrence import RecurrenceTracker

class RecurrenceTests(unittest.TestCase):
    def test_alternating_states_report_returns_but_changed_resources_are_distinct(self):
        tracker=RecurrenceTracker()
        a={'fast_state':{'floor':3,'player':{'x':5,'hp':100}}}
        b={'fast_state':{'floor':2,'player':{'x':5,'hp':100}}}
        self.assertEqual(tracker.observe(a,1)['visits_in_window'],1)
        tracker.observe(b,2)
        tracker.observe(a,3)
        tracker.observe(b,4)
        recurrence=tracker.observe(a,5)
        self.assertEqual(recurrence['matching_steps'],[1,3,5])
        self.assertEqual(recurrence['visits_in_window'],3)
        self.assertNotIn('action',recurrence)
        changed={'fast_state':{'floor':3,'player':{'x':5,'hp':99}}}
        self.assertEqual(tracker.observe(changed,6)['visits_in_window'],1)

    def test_transport_feedback_does_not_hide_recurrence_and_window_is_bounded(self):
        tracker=RecurrenceTracker(window=2)
        tracker.observe({'value':1,'last_execution':{'action':'a'}},1)
        report=tracker.observe({'value':1,'last_execution':{'action':'b'}},2)
        self.assertEqual(report['visits_in_window'],2)
        tracker.observe({'value':2},3)
        tracker.observe({'value':2},4)
        self.assertEqual(tracker.observe({'value':1},5)['visits_in_window'],1)


    def test_recurrence_requests_model_review_without_selecting_an_action(self):
        from general_agent.policy import JevPolicy
        from general_agent.types import Action, Observation
        memories=[]
        def score(criteria,token,tracks,memory='',images=None):
            memories.append(memory)
            return {'probabilities':{'advance':.95,'back':.05},'model':'test-jev'}
        class Outer:
            hint='Inspect current evidence.'
            error=''
            def __init__(self):self.reviews=[]
            def __call__(self,obs,goal,decision,**kwargs):
                self.reviews.append(obs.state['recurrence']['visits_in_window'])
                return 'back'
        outer=Outer();policy=JevPolicy(score_fn=score);policy.call_outer_every=100
        decisions=[]
        for value in ['A','B','A','B','A']:
            observation=Observation(actions=[Action('advance'),Action('back')],
                                    state={'fast_state':{'position':value}})
            decisions.append(policy.choose(observation,'finish',escalate=outer))
        self.assertEqual(outer.reviews,[1,3])
        self.assertEqual([decision.action_id for decision in decisions],['advance']*5)
        self.assertIn('"matching_steps": [1, 3, 5]',memories[-1])
