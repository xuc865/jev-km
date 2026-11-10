"""Tests for the episode loop under the two-role contract."""

from __future__ import annotations

import unittest

from general_agent import ExecutorPolicy, PlannerPolicy, run_episode
from general_agent.types import Action, Observation

from .fake_models import ScriptedClient, executor_reply, planner_reply


class FakeAdapter:
    name = "fake"

    def __init__(self, win_after: int = 2, extra_actions: int = 0) -> None:
        self.win_after = win_after
        self.extra_actions = extra_actions
        self.steps = 0
        self.closed = False
        self.last_execution: dict = {}

    def reset(self) -> None:
        self.steps = 0

    def observe(self) -> Observation:
        done = self.steps >= self.win_after
        actions = [] if done else [Action("right", "go right"), Action("wait", "wait")]
        actions.extend(Action(f"extra{i}", f"extra {i}") for i in range(self.extra_actions))
        return Observation(
            actions=actions,
            state={"steps": self.steps},
            done=done,
            success=done,
        )

    def act(self, action: Action) -> None:
        if action.id == "right":
            self.steps += 1

    def close(self) -> None:
        self.closed = True


def build(client: ScriptedClient, **kwargs):
    return (
        ExecutorPolicy(client=client, model="fake-executor", **kwargs),
        PlannerPolicy(client=client, model="fake-planner"),
    )


class RunnerTests(unittest.TestCase):
    def test_executor_action_is_executed_and_episode_completes(self) -> None:
        adapter = FakeAdapter(win_after=2)
        client = ScriptedClient(executor=[executor_reply("right")])
        result = run_episode(adapter, build(client)[0], "reach the exit", max_steps=10)
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 2)
        self.assertEqual([d.by for d in result.decisions], ["executor", "executor"])
        self.assertTrue(all(d.action_id == "right" for d in result.decisions))
        self.assertTrue(adapter.closed)

    def test_executor_sees_only_legal_actions_and_no_keyboard_universe(self) -> None:
        adapter = FakeAdapter(win_after=1)
        adapter.observe = lambda: Observation(  # type: ignore[method-assign]
            actions=[Action("right"), Action("wait")],
            state={"steps": 0, "input_universe": ["a", "b"], "hint": "secret"},
        )
        client = ScriptedClient(executor=[executor_reply("right")])
        run_episode(adapter, build(client)[0], "goal", max_steps=3)
        payload = client.payloads("executor")[0]
        self.assertEqual(
            [item["id"] for item in payload["world"]["legal_actions"]], ["right", "wait"]
        )
        self.assertNotIn("input_universe", payload["world"]["state"])
        self.assertNotIn("hint", payload["world"]["state"])

    def test_planner_sees_the_keyboard_universe(self) -> None:
        adapter = FakeAdapter(win_after=2)
        adapter.observe = lambda: Observation(  # type: ignore[method-assign]
            actions=[Action("right")],
            state={"steps": 0, "input_universe": ["a", "b"]},
        )
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[planner_reply()],
        )
        executor, planner = build(client)
        run_episode(adapter, executor, "goal", max_steps=2, escalate=planner)
        payload = client.payloads("planner")[0]
        self.assertIn("input_universe", payload["world"]["state"])

    def test_unparseable_executor_reply_falls_back_without_stopping(self) -> None:
        adapter = FakeAdapter(win_after=3)
        client = ScriptedClient(executor=["not json", executor_reply("right")])
        result = run_episode(adapter, build(client)[0], "goal", max_steps=4)
        self.assertEqual(result.decisions[0].by, "fallback")
        self.assertEqual(result.decisions[1].by, "executor")

    def test_illegal_action_id_falls_back_instead_of_executing(self) -> None:
        adapter = FakeAdapter(win_after=1)
        client = ScriptedClient(executor=[executor_reply("teleport")])
        result = run_episode(adapter, build(client)[0], "goal", max_steps=3)
        self.assertEqual(result.decisions[0].by, "fallback")
        self.assertNotEqual(result.decisions[0].action_id, "")

    def test_no_legal_actions_stops(self) -> None:
        adapter = FakeAdapter(win_after=10)
        adapter.observe = lambda: Observation(actions=[], state={}, done=False)  # type: ignore[method-assign]
        client = ScriptedClient(executor=[executor_reply("right")])
        result = run_episode(adapter, build(client)[0], "goal", max_steps=3)
        self.assertFalse(result.success)
        self.assertIn("no legal actions", result.error)

    def test_too_many_candidates_stops_without_pruning(self) -> None:
        adapter = FakeAdapter(win_after=10, extra_actions=3)
        client = ScriptedClient(executor=[executor_reply("right")])
        result = run_episode(
            adapter, build(client, max_candidates=3)[0], "goal", max_steps=3
        )
        self.assertFalse(result.success)
        self.assertIn("adapter must prune", result.error)

    def test_failed_reset_closes_partially_created_adapter(self) -> None:
        adapter = FakeAdapter()

        def failed_reset():
            raise RuntimeError("browser startup failed")

        adapter.reset = failed_reset  # type: ignore[method-assign]
        client = ScriptedClient(executor=[executor_reply("right")])
        with self.assertRaisesRegex(RuntimeError, "browser startup failed"):
            run_episode(adapter, build(client)[0], "goal")
        self.assertTrue(adapter.closed)

    def test_planner_is_consulted_before_the_first_action(self) -> None:
        adapter = FakeAdapter(win_after=2)
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[planner_reply(stage_goal="walk to the exit")],
        )
        executor, planner = build(client)
        result = run_episode(adapter, executor, "goal", max_steps=4, escalate=planner)
        self.assertTrue(result.success)
        payloads = client.payloads("planner")
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["review_trigger"], "first_step")
        self.assertIsNone(payloads[0]["executor_report"])
        self.assertEqual(payloads[0]["current_plan"]["revision"], 0)

    def test_plan_reaches_the_executor_on_the_same_step(self) -> None:
        adapter = FakeAdapter(win_after=1)
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[planner_reply(stage_goal="walk to the exit", tactics="hold right")],
        )
        executor, planner = build(client)
        run_episode(adapter, executor, "goal", max_steps=2, escalate=planner)
        payload = client.payloads("executor")[0]
        self.assertEqual(payload["plan"]["stage_goal"], "walk to the exit")
        self.assertEqual(payload["plan"]["tactics"], "hold right")
        self.assertEqual(payload["plan"]["revision"], 1)

    def test_planner_failure_does_not_block_the_executor(self) -> None:
        adapter = FakeAdapter(win_after=2)
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[RuntimeError("planner down")],
        )
        executor, planner = build(client)
        result = run_episode(adapter, executor, "goal", max_steps=4, escalate=planner)
        self.assertTrue(result.success)
        self.assertEqual(result.decisions[0].by, "executor")

    def test_sustained_planner_outage_stops_the_episode(self) -> None:
        adapter = FakeAdapter(win_after=100)
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[RuntimeError("planner down")],
        )
        executor, planner = build(client)
        result = run_episode(adapter, executor, "goal", max_steps=20, escalate=planner)
        self.assertFalse(result.success)
        self.assertIn("planner unavailable", result.error)

    def test_forbidden_action_is_not_executed(self) -> None:
        adapter = FakeAdapter(win_after=2)
        client = ScriptedClient(
            executor=[executor_reply("wait"), executor_reply("right")],
            planner=[planner_reply(avoid=["wait"])],
        )
        executor, planner = build(client)
        result = run_episode(adapter, executor, "goal", max_steps=4, escalate=planner)
        self.assertEqual(result.decisions[0].by, "fallback")
        self.assertEqual(result.decisions[0].action_id, "right")

    def test_planner_stamps_the_executor_step_so_ages_align(self) -> None:
        """`issued_at_step` must come from the executor's clock, not the adapter's.

        `PlanDirective.age(step)` subtracts that stamp from the executor's counter,
        so a plan stamped with a different counter would look arbitrarily old (or
        young) for its whole life.
        """
        adapter = FakeAdapter(win_after=3)
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[planner_reply(stage_goal="walk")],
        )
        executor, planner = build(client)
        run_episode(adapter, executor, "goal", max_steps=4, escalate=planner)
        self.assertEqual(executor.plan.issued_at_step, 1)
        self.assertEqual(executor.plan.age(executor.steps), executor.steps - 1)

    def test_show_decision_receives_the_declared_plan(self) -> None:
        adapter = FakeAdapter(win_after=1)
        seen: list[dict] = []
        adapter.show_decision = lambda decision, panel: seen.append(panel)  # type: ignore[attr-defined]
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[planner_reply(stage_goal="exit", tactics="right")],
        )
        executor, planner = build(client)
        run_episode(adapter, executor, "goal", max_steps=2, escalate=planner)
        self.assertEqual(seen[0]["stage_goal"], "exit")
        self.assertEqual(seen[0]["tactics"], "right")
        self.assertEqual(seen[0]["plan_revision"], 1)
        self.assertEqual(seen[0]["planner_model"], "fake-planner")


if __name__ == "__main__":
    unittest.main()
