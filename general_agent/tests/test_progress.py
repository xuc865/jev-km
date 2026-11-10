import unittest
from general_agent.progress import ObservedProgress
from general_agent.recurrence import RecurrenceTracker

class ObservedProgressTests(unittest.TestCase):
    def test_motion_does_not_hide_unchanged_score_and_new_score_resets_age(self):
        tracker=ObservedProgress(['score'])
        tracker.observe({'score':{'a':0,'b':0},'position':1},0)
        value=tracker.observe({'score':{'a':0,'b':0},'position':999},100)
        self.assertEqual(value['decisions_since_change'],100)
        changed=tracker.observe({'score':{'a':1,'b':0}},101)
        self.assertEqual(changed['decisions_since_change'],0)
        self.assertEqual(changed['recent_changes'][0]['values']['score'],{'a':0,'b':0})

    def test_repeated_observation_is_idempotent_and_results_cannot_mutate_memory(self):
        tracker=ObservedProgress(['score'])
        first=tracker.observe({'score':0},5)
        second=tracker.observe({'score':0},5)
        self.assertEqual(first,second)
        second['recent_changes'][0]['values']['score']=999
        self.assertEqual(tracker.observe({'score':0},6)['recent_changes'][0]['values']['score'],0)
        self.assertEqual(ObservedProgress().observe({'score':5},0),{})
        self.assertEqual(ObservedProgress(['score']).observe({'score':5},0)['decisions_since_change'],0)

    def test_progress_age_does_not_destroy_state_recurrence(self):
        tracker=RecurrenceTracker()
        first=tracker.observe({'fast_state':{'score':0,'progress_evidence':{'decisions_since_change':0}}},1)
        second=tracker.observe({'fast_state':{'score':0,'progress_evidence':{'decisions_since_change':1}}},2)
        self.assertEqual(first['observation_key'],second['observation_key'])
        self.assertEqual(second['visits_in_window'],2)
