"""Tests for the executor's trigger logic and plan handling."""

from __future__ import annotations

import unittest

from general_agent.contract import PlanDirective
from general_agent.policy import (
    TRIGGER_AGE,
    TRIGGER_ERROR,
    TRIGGER_FIRST,
    TRIGGER_LOW_CONF,
    TRIGGER_NO_CHANGE,
    TRIGGER_PHASE,
    TRIGGER_STUCK,
    ExecutorPolicy,
)
from general_agent.types import Action, Observation

from .fake_models import ScriptedClient, executor_reply, planner_reply


def observation(actions=None, level=1, state=None, image="") -> Observation:
    return Observation(
        actions=actions if actions is not None else [Action("right"), Action("wait")],
        level=level,
        state=state or {},
        image=image,
    )


def build(client: ScriptedClient, **kwargs) -> ExecutorPolicy:
    return ExecutorPolicy(client=client, model="fake", **kwargs)


class TriggerTests(unittest.TestCase):
    def test_first_step_always_reviews_because_no_plan_exists(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]))
        world = policy.build_world(observation(), "goal")
        self.assertEqual(policy.review_trigger(observation(), world), TRIGGER_FIRST)

    def test_a_fresh_plan_with_good_confidence_does_not_review(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right", confidence=0.9)]))
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=8, revision=1))
        policy.steps = 3
        policy.steps_since_review = 2
        policy.last_reply = None
        world = policy.build_world(observation(), "goal")
        self.assertEqual(policy.review_trigger(observation(), world), "")

    def test_low_self_reported_confidence_reviews(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right", confidence=0.1)]))
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=20, revision=1))
        policy.steps_since_review = 1
        world = policy.build_world(observation(), "goal")
        policy.last_reply = __import__(
            "general_agent.contract", fromlist=["ExecutorReply"]
        ).ExecutorReply(action="right", confidence=0.1)
        self.assertEqual(policy.review_trigger(observation(), world), TRIGGER_LOW_CONF)

    def test_explicit_stuck_flag_reviews_before_other_signals(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right", stuck=True)]))
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=20, revision=1))
        policy.steps_since_review = 1
        from general_agent.contract import ExecutorReply

        policy.last_reply = ExecutorReply(action="right", confidence=0.9, stuck=True)
        world = policy.build_world(observation(), "goal")
        self.assertEqual(policy.review_trigger(observation(), world), TRIGGER_STUCK)

    def test_phase_change_outranks_aging(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]))
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=2, revision=1))
        policy.last_phase = ("menu", ("right", "wait"))
        policy.steps_since_review = 10
        world = policy.build_world(observation(state={"phase": "play"}), "goal")
        self.assertEqual(policy.review_trigger(observation(state={"phase": "play"}), world), TRIGGER_PHASE)

    def test_level_change_reviews(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]))
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=20, revision=1))
        policy.last_phase = ("", ("right", "wait"))
        policy.last_level = 1
        obs = observation(level=2)
        world = policy.build_world(obs, "goal")
        self.assertEqual(policy.review_trigger(obs, world), TRIGGER_PHASE)

    def test_repeated_identical_action_reviews(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]))
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=20, revision=1))
        policy.last_phase = ("", ("right", "wait"))
        policy.recent = ["right", "right", "right"]
        obs = observation()
        world = policy.build_world(obs, "goal")
        self.assertEqual(policy.review_trigger(obs, world), TRIGGER_NO_CHANGE)

    def test_plan_age_reviews_at_the_requested_interval(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]), review_every=2)
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=2, revision=1))
        policy.last_phase = ("", ("right", "wait"))
        policy.steps_since_review = 2
        obs = observation()
        world = policy.build_world(obs, "goal")
        self.assertEqual(policy.review_trigger(obs, world), TRIGGER_AGE)

    def test_repeated_failures_review(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]))
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=20, revision=1))
        policy.failures = 2
        obs = observation()
        world = policy.build_world(obs, "goal")
        self.assertEqual(policy.review_trigger(obs, world), TRIGGER_ERROR)

    def test_planner_cannot_shorten_the_configured_review_floor(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]), review_every=10)
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=2, revision=1))
        self.assertEqual(policy.plan_interval(), 10)

    def test_planner_can_lengthen_the_interval_within_the_ceiling(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]), review_every=4)
        policy.apply_plan(PlanDirective(stage_goal="g", review_in=30, revision=1))
        self.assertEqual(policy.plan_interval(), 30)


class ExecutionTests(unittest.TestCase):
    def test_executor_payload_carries_the_current_plan(self) -> None:
        client = ScriptedClient(executor=[executor_reply("right")])
        policy = build(client)
        policy.apply_plan(PlanDirective(stage_goal="go east", tactics="hold right", revision=4))
        policy.choose(observation(), "goal")
        payload = client.payloads("executor")[0]
        self.assertEqual(payload["plan"]["stage_goal"], "go east")
        self.assertEqual(payload["plan"]["revision"], 4)

    def test_accepted_reply_gets_stamped_as_the_training_label(self) -> None:
        client = ScriptedClient(executor=[executor_reply("right", why="east is open")])
        policy = build(client)
        policy.choose(observation(), "goal")
        label = client.calls[0].label
        self.assertEqual(label["action"], "right")
        self.assertEqual(label["why"], "east is open")
        self.assertIn("confidence", label)

    def test_rejected_reply_leaves_no_label(self) -> None:
        client = ScriptedClient(executor=[executor_reply("teleport")])
        policy = build(client)
        policy.choose(observation(), "goal")
        self.assertEqual(client.calls[0].label, {})

    def test_executor_never_sees_the_input_universe(self) -> None:
        client = ScriptedClient(executor=[executor_reply("right")])
        policy = build(client)
        policy.choose(
            observation(state={"input_universe": ["a"], "hp": 5, "hint": "advice"}), "goal"
        )
        state = client.payloads("executor")[0]["world"]["state"]
        self.assertEqual(state, {"hp": 5})

    def test_step_report_is_kept_for_the_planner(self) -> None:
        client = ScriptedClient(executor=[executor_reply("right", expect="move east")])
        policy = build(client)
        policy.apply_plan(PlanDirective(stage_goal="g", revision=2, issued_at_step=0))
        policy.choose(observation(), "goal")
        self.assertEqual(policy.last_step.action, "right")
        self.assertEqual(policy.last_step.expect, "move east")
        self.assertEqual(policy.last_step.plan_revision, 2)
        self.assertEqual(policy.last_step.plan_age, 1)
        self.assertEqual(policy.last_step.step, 1)

    def test_outcome_from_the_adapter_is_attached_to_the_step(self) -> None:
        client = ScriptedClient(executor=[executor_reply("right")])
        policy = build(client)
        obs = observation()
        decision = policy.choose(obs, "goal")
        policy.record_execution(obs, decision, Action("right"), {"changed": False})
        self.assertEqual(policy.last_step.outcome, "no_visible_change")
        self.assertEqual(policy.history[-1].outcome, "no_visible_change")

    def test_outcome_falls_back_to_a_note(self) -> None:
        client = ScriptedClient(executor=[executor_reply("right")])
        policy = build(client)
        obs = observation()
        decision = policy.choose(obs, "goal")
        policy.record_execution(obs, decision, Action("right"), {"note": "queued click"})
        self.assertEqual(policy.last_step.outcome, "queued click")

    def test_goal_is_locked_after_the_first_step(self) -> None:
        client = ScriptedClient(executor=[executor_reply("right")])
        policy = build(client)
        policy.choose(observation(), "root goal")
        policy.choose(observation(), "replacement goal")
        systems = client.systems("executor")
        self.assertIn("root goal", systems[0])
        self.assertIn("root goal", systems[1])
        self.assertNotIn("replacement goal", systems[1])

    def test_done_observation_stops_without_calling_the_model(self) -> None:
        client = ScriptedClient(executor=[executor_reply("right")])
        policy = build(client)
        decision = policy.choose(Observation(actions=[Action("right")], done=True), "goal")
        self.assertEqual(decision.by, "stop")
        self.assertEqual(client.calls, [])

    def test_fallback_skips_forbidden_and_wait_actions(self) -> None:
        client = ScriptedClient(executor=["bad json"])
        policy = build(client)
        policy.apply_plan(PlanDirective(stage_goal="g", avoid=("right",), revision=1))
        obs = observation(actions=[Action("right"), Action("wait"), Action("left")])
        decision = policy.choose(obs, "goal")
        self.assertEqual(decision.by, "fallback")
        self.assertEqual(decision.action_id, "left")

    def test_no_usable_fallback_stops_rather_than_guessing(self) -> None:
        client = ScriptedClient(executor=["bad json"])
        policy = build(client)
        policy.apply_plan(PlanDirective(stage_goal="g", avoid=("right",), revision=1))
        decision = policy.choose(observation(actions=[Action("right"), Action("wait")]), "goal")
        self.assertEqual(decision.by, "stop")
        self.assertIn("executor reply unusable", decision.error)


class ReviewTests(unittest.TestCase):
    def test_planner_is_called_when_a_trigger_fires(self) -> None:
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[planner_reply(stage_goal="go east")],
        )
        policy = build(client)
        from general_agent.outer import PlannerPolicy

        planner = PlannerPolicy(client=client, model="fake-planner")
        policy.choose(observation(), "goal", escalate=planner)
        self.assertEqual(len(client.payloads("planner")), 1)
        self.assertEqual(policy.plan.stage_goal, "go east")
        self.assertEqual(policy.plan.revision, 1)

    def test_adopting_a_plan_resets_the_review_clock(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]))
        policy.steps = 5
        policy.steps_since_review = 5
        policy.failures = 3
        policy.apply_plan(PlanDirective(stage_goal="g", revision=1))
        self.assertEqual(policy.steps_since_review, 0)
        self.assertEqual(policy.failures, 0)
        self.assertEqual(policy.last_review_step, 5)

    def test_failed_review_does_not_reset_the_clock(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]))

        class Failing:
            error = "down"
            plan = None

            def __call__(self, *args, **kwargs):
                return None

        policy.steps = 4
        policy.steps_since_review = 4
        adopted = policy._review(observation(), policy.build_world(observation(), "goal"),
                                 "goal", "plan_age", Failing())
        self.assertFalse(adopted)
        self.assertEqual(policy.steps_since_review, 4)

    def test_identical_revision_is_not_adopted(self) -> None:
        policy = build(ScriptedClient(executor=[executor_reply("right")]))
        policy.apply_plan(PlanDirective(stage_goal="same", revision=3))

        class Echo:
            error = ""
            plan = PlanDirective(stage_goal="same", revision=3)

            def __call__(self, *args, **kwargs):
                return None

        adopted = policy._review(
            observation(), policy.build_world(observation(), "goal"), "goal", "plan_age", Echo()
        )
        self.assertFalse(adopted)

    def test_trigger_is_recorded_for_the_dataset(self) -> None:
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[planner_reply(stage_goal="g")],
        )
        policy = build(client)
        from general_agent.outer import PlannerPolicy

        policy.choose(observation(), "goal", escalate=PlannerPolicy(client=client))
        self.assertEqual(policy.last_trigger, TRIGGER_FIRST)


if __name__ == "__main__":
    unittest.main()
