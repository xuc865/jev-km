"""Keyboard pruning: the universe the planner sees, the subset it may exclude.

Split out from the old ``control_exclude`` tests because the mechanism moved. The
planner no longer returns an ad-hoc verdict dict keyed ``control_exclude``; it
declares ``exclude`` inside the plan contract, and the executor applies it to the
adapter when the plan is adopted. The adapter's own clamping rules are unchanged
and still tested here, because they are what keeps a bad exclusion from leaving
the game with no controls.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from general_agent.adapters import generic_html
from general_agent.adapters.generic_html import GenericHtmlAdapter
from general_agent.contract import PlanDirective, WorldView
from general_agent.policy import ExecutorPolicy
from general_agent.types import Action, Observation

from .fake_models import ScriptedClient, executor_reply, planner_reply


def make_adapter(tmp: str) -> GenericHtmlAdapter:
    adapter = GenericHtmlAdapter("test-game")
    adapter.artifacts = Path(tmp)
    return adapter


class InputUniverseTests(unittest.TestCase):
    def test_universe_covers_letters_digits_navigation_and_core_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            ids = [key for key, _label, _keys in adapter._universe_keys()]
        for needed in ("move_up", "confirm", "key_d", "key_p", "digit_0", "digit_9",
                       "key_tab", "key_backspace"):
            self.assertIn(needed, ids)
        self.assertEqual(len(ids), len(set(ids)), "input ids must be unique")

    def test_tier_zero_still_offers_only_the_original_movement_keys(self):
        """A game that is being played must not see a wider keyboard than before."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            adapter._no_progress_steps = 0
            self.assertEqual([item[0] for item in adapter._key_table()],
                             [item[0] for item in generic_html.MOVE_KEYS])

    def test_stalled_game_opens_the_whole_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            adapter._no_progress_steps = 10
            offered = {item[0] for item in adapter._key_table()}
            universe = {item[0] for item in adapter._universe_keys()}
        self.assertIn("key_q", offered)
        self.assertIn("digit_9", offered)
        self.assertIn("key_tab", offered)
        # A key the planner was shown but the executor cannot press would make a
        # correct exclusion look like an invented id.
        self.assertEqual(offered, universe)
        self.assertIn("escape", universe)
        self.assertIn("restart", universe)


class SubsetPersistenceTests(unittest.TestCase):
    def test_apply_subset_writes_a_per_game_file_and_reloads_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(generic_html, "SUBSET_ROOT", Path(tmp)):
                adapter = make_adapter(tmp)
                result = adapter.apply_subset(["digit_0", "key_tab"], note="no menu on screen")
                self.assertTrue(result["ok"])
                path = Path(tmp) / "test-game.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(record["exclude"], ["digit_0", "key_tab"])
                self.assertNotIn("digit_0", record["keep"])
                reloaded = make_adapter(tmp)
                self.assertEqual(reloaded.excluded, {"digit_0", "key_tab"})
                self.assertEqual(reloaded.excluded_note, "no menu on screen")

    def test_unknown_and_invented_ids_are_discarded_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(generic_html, "SUBSET_ROOT", Path(tmp)):
                Path(tmp, "test-game.json").write_text(
                    json.dumps({"exclude": ["key_q", "click_center", "invented_key"]}),
                    encoding="utf-8")
                adapter = make_adapter(tmp)
        # click_* is not a keyboard input; it must never be removed this way.
        self.assertEqual(adapter.excluded, {"key_q"})

    def test_a_corrupt_file_means_no_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(generic_html, "SUBSET_ROOT", Path(tmp)):
                Path(tmp, "test-game.json").write_text("{not json", encoding="utf-8")
                adapter = make_adapter(tmp)
        self.assertEqual(adapter.excluded, set())

    def test_subset_cannot_empty_the_keyboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(generic_html, "SUBSET_ROOT", Path(tmp)):
                adapter = make_adapter(tmp)
                everything = [key for key, _l, _k in adapter._universe_keys()]
                result = adapter.apply_subset(everything)
                self.assertFalse(result["ok"])
                self.assertEqual(adapter.excluded, set())
                self.assertFalse((Path(tmp) / "test-game.json").exists())


class SubsetEffectTests(unittest.TestCase):
    def test_excluded_keys_disappear_from_the_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(generic_html, "SUBSET_ROOT", Path(tmp)):
                adapter = make_adapter(tmp)
                adapter._no_progress_steps = 10
                before = {action.id for action in adapter._actions({"candidates": []}, "sig")}
                adapter.apply_subset(["key_q", "key_w", "digit_0"])
                after = {action.id for action in adapter._actions({"candidates": []}, "sig")}
        self.assertIn("key_q", before)
        self.assertNotIn("key_q", after)
        self.assertNotIn("digit_0", after)
        self.assertIn("key_d", after, "unrelated letters must survive")

    def test_pruning_never_removes_the_movement_keys_needed_to_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(generic_html, "SUBSET_ROOT", Path(tmp)):
                adapter = make_adapter(tmp)
                adapter._no_progress_steps = 10
                adapter.apply_subset(["key_d", "key_f", "key_q"])
                offered = {action.id for action in adapter._actions({"candidates": []}, "sig")}
        self.assertIn("move_up", offered)
        self.assertIn("confirm", offered)

    def test_excluding_one_id_also_excludes_its_physical_alias(self):
        """`key_w` and `move_up_alt` send the same key; both must go together."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(generic_html, "SUBSET_ROOT", Path(tmp)):
                adapter = make_adapter(tmp)
                result = adapter.apply_subset(["key_w"])
                self.assertTrue(result["ok"])
                self.assertIn("key_w", adapter.excluded)
                self.assertIn("move_up_alt", adapter.excluded)
                adapter._no_progress_steps = 10
                offered = {action.id for action in adapter._actions({"candidates": []}, "sig")}
        self.assertNotIn("key_w", offered)
        self.assertNotIn("move_up_alt", offered)
        self.assertIn("move_up", offered, "the arrow key is a different physical key")


class CandidateBudgetTests(unittest.TestCase):
    def test_keys_are_not_truncated_to_nothing_by_a_long_dom_candidate_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            adapter._no_progress_steps = 10
            frame = {"candidates": [
                {"id": f"dom_{index}", "selector": f"#c{index}", "label": f"button {index}"}
                for index in range(80)
            ]}
            actions = adapter._actions(frame, "sig")
        kinds = [action.kind for action in actions]
        self.assertGreaterEqual(sum(1 for kind in kinds if kind == "web_keys"),
                                adapter.MIN_KEY_SLOTS)
        self.assertGreaterEqual(sum(1 for kind in kinds if kind == "dom_click"),
                                generic_html.MIN_CLICK_SLOTS)
        self.assertLessEqual(len(actions), adapter.max_candidates)

    def test_click_driven_game_keeps_points_when_no_dom_candidates_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            adapter._no_progress_steps = 10
            actions = adapter._actions({"candidates": []}, "sig")
        self.assertGreaterEqual(sum(1 for a in actions if a.kind == "canvas_click"),
                                generic_html.MIN_CLICK_SLOTS)


class PlannerExcludeParsingTests(unittest.TestCase):
    """`exclude` is a plan field now, so it is parsed with the rest of the plan."""

    def plan(self, text: str, previous: PlanDirective | None = None, legal=None) -> PlanDirective:
        from general_agent.contract import parse_planner_reply

        reply = parse_planner_reply(
            text, previous or PlanDirective(), 3, legal or {"move_up", "key_d", "key_q"}
        )
        self.assertIsNotNone(reply)
        return reply.plan

    def test_exclusions_are_accepted(self):
        plan = self.plan(planner_reply(stage_goal="g", exclude=["key_q"]))
        self.assertEqual(plan.exclude, ("key_q",))

    def test_exclusions_are_sorted_and_deduplicated(self):
        plan = self.plan(
            '{"stage_goal":"g","exclude":["key_q","key_q","digit_0"]}',
            legal={"key_q", "digit_0"},
        )
        self.assertEqual(plan.exclude, ("digit_0", "key_q"))

    def test_an_omitted_exclude_preserves_the_previous_set(self):
        previous = PlanDirective(stage_goal="g", exclude=("key_q",))
        plan = self.plan(planner_reply(stage_goal=None, tactics=None, situation="calm"),
                         previous=previous)
        self.assertEqual(plan.exclude, ("key_q",))

    def test_clear_restores_the_full_keyboard(self):
        previous = PlanDirective(stage_goal="g", exclude=("key_q",))
        plan = self.plan('{"stage_goal":"g","exclude":"clear"}', previous=previous)
        self.assertEqual(plan.exclude, ())

    def test_a_non_list_exclude_is_ignored(self):
        previous = PlanDirective(stage_goal="g", exclude=("key_q",))
        plan = self.plan('{"stage_goal":"g","exclude":42}', previous=previous)
        self.assertEqual(plan.exclude, ("key_q",))

    def test_the_prompt_states_the_universe_caution(self):
        from general_agent.prompts import PLANNER_SYSTEM

        self.assertIn("exclude", PLANNER_SYSTEM)
        self.assertIn("cannot get back", PLANNER_SYSTEM)


class PolicySubsetSinkTests(unittest.TestCase):
    def build(self, sink=None) -> ExecutorPolicy:
        policy = ExecutorPolicy(
            client=ScriptedClient(executor=[executor_reply("move_up")]), model="fake"
        )
        if sink is not None:
            policy.subset_sink = sink
        return policy

    def obs(self) -> Observation:
        return Observation(
            actions=[Action("move_up")],
            state={"phase": "play", "input_universe": ["move_up", "key_q"]},
        )

    def test_adopting_a_plan_reaches_the_adapter_through_the_callback(self):
        seen = {}
        policy = self.build(
            lambda exclude, note="": seen.update({"exclude": list(exclude), "note": note})
        )
        policy._apply_exclude(PlanDirective(stage_goal="g", exclude=("key_q",), situation="rhythm"))
        self.assertEqual(seen["exclude"], ["key_q"])
        self.assertEqual(seen["note"], "rhythm")

    def test_a_rejected_pruning_is_recorded_not_retried(self):
        policy = self.build(lambda exclude, note="": {"ok": False, "reason": "no play key left"})
        policy._apply_exclude(PlanDirective(stage_goal="g", exclude=("key_q",)))
        self.assertIn("no play key left", policy.exclude_note)

    def test_no_callback_means_no_crash_for_other_adapters(self):
        policy = self.build()
        policy._apply_exclude(PlanDirective(stage_goal="g", exclude=("key_q",)))
        self.assertIsNone(policy.subset_sink)
        self.assertEqual(policy.exclude_note, "")

    def test_a_raising_callback_does_not_break_the_step(self):
        def boom(exclude, note=""):
            raise RuntimeError("adapter gone")

        policy = self.build(boom)
        policy._apply_exclude(PlanDirective(stage_goal="g", exclude=("key_q",)))
        self.assertIn("adapter gone", policy.exclude_note)

    def test_a_plan_without_exclusions_does_not_call_the_adapter(self):
        called = []
        policy = self.build(lambda exclude, note="": called.append(list(exclude)))
        policy._apply_exclude(PlanDirective(stage_goal="g"))
        self.assertEqual(called, [])

    def test_executor_never_receives_the_full_input_universe(self):
        """The executor may only name a legal action, so the universe stays out."""
        client = ScriptedClient(executor=[executor_reply("move_up")])
        policy = ExecutorPolicy(client=client, model="fake")
        policy.choose(
            Observation(
                actions=[Action("move_up")],
                state={"phase": "play", "input_universe": ["move_up", "key_q"],
                       "input_excluded": ["key_q"]},
            ),
            "goal",
        )
        payload = client.payloads("executor")[0]
        self.assertNotIn("key_q", json.dumps(payload["world"]["state"]))
        self.assertNotIn("input_universe", payload["world"]["state"])

    def test_excluded_ids_are_also_treated_as_forbidden(self):
        # The adapter applies the pruning, but the plan is adopted before the next
        # observation, so the executor must not pick an id it just excluded.
        policy = self.build()
        policy.apply_plan(PlanDirective(stage_goal="g", exclude=("move_up",)))
        self.assertTrue(policy.plan.forbids("move_up"))


if __name__ == "__main__":
    unittest.main()
