"""Tests for the live panel's snapshot under the new contract.

The panel is the only consumer that reads policy internals by name, so it is also
the place where a renaming refactor breaks silently: nothing raises, the panel
just renders blanks. These tests pin the field names it publishes.
"""

from __future__ import annotations

import unittest

from general_agent.contract import ExecutorStep, PlanDirective
from general_agent.live import LiveRun
from general_agent.types import Action, Decision, Observation


class SnapshotTests(unittest.TestCase):
    def build(self) -> LiveRun:
        import threading

        run = object.__new__(LiveRun)
        run.game = "demo"
        run.goal = "reach the exit"
        run.run_id = "run"
        run.folder = "runs/panel/run"
        run.kind = "generic"
        run.running = True
        run.status = "执行者决策"
        run.steps = 12
        run.level = 2
        run.started_at = 0.0
        run.error = ""
        run.game_url = ""
        run.adapter = type("A", (), {"config": {}})()
        run.obs = Observation(actions=[Action("right", "go right")], level=2)
        run.decision = Decision("right", by="executor", model="executor-model", ms=120,
                                note="east is open")
        run.recent = [run.decision]
        run.policy = type("P", (), {})()
        run.policy.model = "executor-model"
        run.policy.steps = 12
        run.policy.plan = PlanDirective(
            briefing="rules", stage_goal="exit", tactics="hold right",
            situation="corridor", avoid=("wait",), exclude=("key_q",),
            review_in=6, revision=3, issued_at_step=9,
        )
        run.policy.last_step = ExecutorStep(
            step=12, action="right", why="east is open", confidence=0.82,
            expect="move east", belief="open corridor", stuck=False, repeat_run=1,
            plan_revision=3, plan_age=3, outcome="state_changed",
        )
        run.outer = type("O", (), {})()
        run.outer.model = "planner-model"
        run.outer.revisions = 3
        run.outer.error = ""
        run.outer.verdict = "continue"
        run.client = type("C", (), {"calls": [1, 2, 3]})()
        run._lock = threading.Lock()
        return run

    def test_plan_fields_are_published_under_their_contract_names(self) -> None:
        snap = self.build().snapshot()
        self.assertEqual(snap["briefing"], "rules")
        self.assertEqual(snap["stage_goal"], "exit")
        self.assertEqual(snap["tactics"], "hold right")
        self.assertEqual(snap["situation"], "corridor")
        self.assertEqual(snap["avoid"], ["wait"])
        self.assertEqual(snap["exclude"], ["key_q"])
        self.assertEqual(snap["review_in"], 6)
        self.assertEqual(snap["plan_revision"], 3)

    def test_plan_age_is_measured_from_issue_step(self) -> None:
        snap = self.build().snapshot()
        self.assertEqual(snap["plan_age"], 3)
        self.assertEqual(snap["stage_age"], 3)

    def test_executor_evidence_is_published_as_one_block(self) -> None:
        snap = self.build().snapshot()
        self.assertEqual(snap["executor"]["why"], "east is open")
        self.assertAlmostEqual(snap["executor"]["confidence"], 0.82)
        self.assertEqual(snap["executor"]["expect"], "move east")
        self.assertEqual(snap["executor"]["belief"], "open corridor")
        self.assertFalse(snap["executor"]["stuck"])
        self.assertEqual(snap["executor"]["outcome"], "state_changed")

    def test_both_model_names_are_reported_separately(self) -> None:
        snap = self.build().snapshot()
        self.assertEqual(snap["executor_model"], "executor-model")
        self.assertEqual(snap["planner_model"], "planner-model")
        self.assertEqual(snap["planner_calls"], 3)
        self.assertEqual(snap["model_calls"], 3)

    def test_no_removed_fields_are_published(self) -> None:
        snap = self.build().snapshot()
        for removed in ("probabilities", "image_count", "outer_calls", "outer_model",
                        "outer_situation", "outer_hint", "outer_stage_goal",
                        "play_instruction", "last_outer_step"):
            self.assertNotIn(removed, snap)

    def test_empty_policy_state_still_renders(self) -> None:
        run = self.build()
        run.policy.plan = PlanDirective()
        run.policy.last_step = None
        snap = run.snapshot()
        self.assertEqual(snap["stage_goal"], "")
        self.assertEqual(snap["executor"]["why"], "")
        # No plan has ever been issued, so there is no age to report.
        self.assertEqual(snap["plan_age"], 0)

    def test_recent_decisions_do_not_carry_probability_tables(self) -> None:
        snap = self.build().snapshot()
        self.assertEqual(snap["recent"][0]["choice"], "right")
        self.assertNotIn("probabilities", snap["recent"][0])


if __name__ == "__main__":
    unittest.main()
