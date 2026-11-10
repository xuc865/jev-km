"""Tests for the planner: role boundary, plan merging, failure handling."""

from __future__ import annotations

import unittest

from general_agent.contract import ExecutorStep, PlanDirective, render_planner_payload
from general_agent.outer import PlannerPolicy
from general_agent.types import Action, Observation

from .fake_models import ScriptedClient, planner_reply


def observation(actions=None, level=1, state=None, image="") -> Observation:
    return Observation(
        actions=actions if actions is not None else [Action("right"), Action("wait")],
        level=level,
        state=state or {},
        image=image,
    )


def build(client: ScriptedClient, **kwargs) -> PlannerPolicy:
    return PlannerPolicy(client=client, model="fake-planner", **kwargs)


class PlannerBoundaryTests(unittest.TestCase):
    def test_planner_never_returns_an_action(self) -> None:
        # The return value is always None by design: a planner-supplied action
        # would let the slow tier bypass the executor's ownership of selection.
        client = ScriptedClient(planner=[planner_reply(stage_goal="go east")])
        planner = build(client)
        self.assertIsNone(planner(observation(), "goal"))

    def test_plan_is_published_through_the_attribute(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="go east", tactics="hold right")])
        planner = build(client)
        planner(observation(), "goal")
        self.assertEqual(planner.plan.stage_goal, "go east")
        self.assertEqual(planner.plan.tactics, "hold right")
        self.assertEqual(planner.revisions, 1)

    def test_planner_sees_the_input_universe_and_the_report(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="g")])
        planner = build(client)
        report = ExecutorStep(step=7, action="right", why="east", confidence=0.3,
                              plan_age=4, repeat_run=2, outcome="no_visible_change")
        planner(
            observation(state={"input_universe": ["a"], "phase": "play"}),
            "goal",
            trigger="low_confidence",
            report=report,
        )
        payload = client.payloads("planner")[0]
        self.assertEqual(payload["world"]["state"]["input_universe"], ["a"])
        self.assertEqual(payload["review_trigger"], "low_confidence")
        self.assertEqual(payload["executor_report"]["confidence"], 0.3)
        self.assertEqual(payload["executor_report"]["outcome"], "no_visible_change")

    def test_planner_system_prompt_forbids_naming_a_control(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="g")])
        build(client)(observation(), "goal")
        system = client.systems("planner")[0]
        self.assertIn("Never name a single control", system)
        self.assertIn("avoid", system)
        self.assertIn("executor's job", system)

    def test_goal_is_locked_across_calls(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="g")])
        planner = build(client)
        planner(observation(), "original goal")
        planner(observation(), "replacement goal")
        systems = client.systems("planner")
        self.assertIn("original goal", systems[0])
        self.assertIn("original goal", systems[1])
        self.assertNotIn("replacement goal", systems[1])


class PlanMergeTests(unittest.TestCase):
    def test_partial_reply_preserves_the_untouched_fields(self) -> None:
        client = ScriptedClient(
            planner=[
                planner_reply(briefing="rules", stage_goal="survive", tactics="block"),
                planner_reply(stage_goal=None, tactics=None, situation="under attack"),
            ]
        )
        planner = build(client)
        planner(observation(), "goal")
        planner(observation(), "goal")
        self.assertEqual(planner.plan.briefing, "rules")
        self.assertEqual(planner.plan.tactics, "block")
        self.assertEqual(planner.plan.situation, "under attack")
        self.assertEqual(planner.plan.revision, 2)

    def test_unparseable_reply_keeps_the_previous_plan(self) -> None:
        client = ScriptedClient(
            planner=[planner_reply(stage_goal="survive", tactics="block"), "not json"]
        )
        planner = build(client)
        planner(observation(), "goal")
        planner(observation(), "goal")
        self.assertEqual(planner.plan.stage_goal, "survive")
        self.assertEqual(planner.plan.tactics, "block")
        self.assertEqual(planner.plan.revision, 1)
        self.assertTrue(planner.error)

    def test_transport_error_is_surfaced_and_plan_untouched(self) -> None:
        client = ScriptedClient(planner=[RuntimeError("endpoint down")])
        planner = build(client)
        planner(observation(), "goal")
        self.assertIn("endpoint down", planner.error)
        self.assertEqual(planner.plan.revision, 0)
        self.assertEqual(planner.revisions, 0)

    def test_empty_reply_is_rejected(self) -> None:
        client = ScriptedClient(planner=["{}", "{}"])
        planner = build(client)
        planner(observation(), "goal")
        self.assertEqual(planner.plan.revision, 0)
        self.assertEqual(planner.revisions, 0)
        self.assertTrue(planner.error)

    def test_accepted_reply_becomes_the_training_label(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="go east", verdict="pivot")])
        build(client)(observation(), "goal")
        label = client.calls[0].label
        self.assertEqual(label["plan"]["stage_goal"], "go east")
        self.assertEqual(label["verdict"], "pivot")
        self.assertEqual(label["plan"]["revision"], 1)

    def test_rejected_reply_leaves_no_label(self) -> None:
        client = ScriptedClient(planner=["not json"])
        build(client)(observation(), "goal")
        self.assertEqual(client.calls[0].label, {})

    def test_revision_counter_matches_accepted_plans(self) -> None:
        client = ScriptedClient(
            planner=[planner_reply(stage_goal="a"), "bad", planner_reply(stage_goal="b")]
        )
        planner = build(client)
        for _ in range(3):
            planner(observation(), "goal")
        self.assertEqual(planner.revisions, 2)
        self.assertEqual(planner.plan.revision, 2)

    def test_avoid_only_accepts_legal_ids(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="g", avoid=["wait", "teleport"])])
        planner = build(client)
        planner(observation(), "goal")
        self.assertEqual(planner.plan.avoid, ("wait",))


class PayloadTests(unittest.TestCase):
    def test_payload_shape_is_the_declared_one(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="g")])
        build(client)(observation(), "goal", trigger="first_step")
        payload = client.payloads("planner")[0]
        self.assertEqual(
            list(payload),
            ["output_schema", "world", "current_plan", "executor_report",
             "recent_actions", "review_trigger"],
        )

    def test_missing_trigger_gets_an_explicit_value(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="g")])
        build(client)(observation(), "goal")
        self.assertEqual(client.payloads("planner")[0]["review_trigger"], "unspecified")

    def test_recent_actions_are_carried_but_capped(self) -> None:
        client = ScriptedClient(planner=[planner_reply(stage_goal="g")])
        build(client)(observation(), "goal", recent=[f"a{i}" for i in range(30)])
        self.assertEqual(len(client.payloads("planner")[0]["recent_actions"]), 12)


if __name__ == "__main__":
    unittest.main()
