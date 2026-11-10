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


    def test_recurring_state_is_reported_to_the_planner_not_turned_into_an_action(self):
        """Recurrence is evidence for the planner, never an action selector.

        The tracker itself is unchanged; what changed is who reads it. Under the
        old design the fast scorer got the recurrence block as part of its prompt
        and the planner was called as a side effect. Now the block travels in the
        planner's world state and the executor never sees it.
        """
        from general_agent.outer import PlannerPolicy
        from general_agent.policy import ExecutorPolicy
        from general_agent.types import Action, Observation

        from .fake_models import ScriptedClient, executor_reply, planner_reply

        tracker = RecurrenceTracker()
        client = ScriptedClient(
            executor=[executor_reply("advance")],
            planner=[planner_reply(stage_goal="advance")],
        )
        policy = ExecutorPolicy(client=client, model="fake")
        planner = PlannerPolicy(client=client, model="fake-planner")
        decisions = []
        for index, value in enumerate(["A", "B", "A", "B", "A"]):
            recurrence = tracker.observe({"fast_state": {"position": value}}, index + 1)
            observation = Observation(
                actions=[Action("advance"), Action("back")],
                state={"recurrence": recurrence},
            )
            decisions.append(policy.choose(observation, "finish", escalate=planner))

        self.assertEqual([decision.action_id for decision in decisions], ["advance"] * 5)
        self.assertEqual([decision.by for decision in decisions], ["executor"] * 5)
        # The planner was reviewed once (first step) and saw the recurrence block.
        payload = client.payloads("planner")[0]
        self.assertEqual(payload["world"]["state"]["recurrence"]["visits_in_window"], 1)
        self.assertNotIn("recurrence", client.payloads("executor")[0]["world"]["state"])
