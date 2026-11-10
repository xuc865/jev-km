"""Slow-model input pruning: the universe it sees, and the subset it may keep."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from general_agent.adapters import generic_html
from general_agent.adapters.generic_html import GenericHtmlAdapter
from general_agent.outer import OuterEscalation
from general_agent.policy import JevPolicy
from general_agent.types import Action, Observation


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
        # A key the slow model was shown but the fast model cannot press would
        # make a correct exclusion look like an invented id.
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


class OuterSubsetParsingTests(unittest.TestCase):
    def _outer(self, answer: dict) -> OuterEscalation:
        def chat(prompt, token, **kwargs):
            return {"choices": [{"message": {"content": json.dumps(answer)}}]}
        return OuterEscalation(chat_fn=chat, generate_criteria=True)

    def _obs(self) -> Observation:
        return Observation(
            actions=[Action("move_up"), Action("key_d")],
            state={"phase": "play", "input_universe": ["move_up", "key_d", "key_q"]},
        )

    def test_exclusions_inside_the_universe_are_accepted(self):
        outer = self._outer({"action": "key_d",
                             "action_criteria": {"move_up": "when moving", "key_d": "when a beat drops"},
                             "control_exclude": ["key_q"]})
        outer(self._obs(), "goal")
        self.assertEqual(outer.control_exclude, ["key_q"])
        self.assertTrue(outer.control_exclude_status["applied"])

    def test_an_id_outside_the_universe_refuses_the_whole_subset(self):
        outer = self._outer({"action": "key_d",
                             "action_criteria": {"move_up": "x", "key_d": "y"},
                             "control_exclude": ["key_q", "invented_key"]})
        outer(self._obs(), "goal")
        self.assertEqual(outer.control_exclude, [])
        self.assertFalse(outer.control_exclude_status["applied"])
        self.assertIn("invented_key", outer.control_exclude_status["unknown_ids"])

    def test_a_non_list_subset_is_ignored(self):
        outer = self._outer({"action": "key_d",
                             "action_criteria": {"move_up": "x", "key_d": "y"},
                             "control_exclude": "key_q"})
        outer(self._obs(), "goal")
        self.assertEqual(outer.control_exclude, [])
        self.assertFalse(outer.control_exclude_status["applied"])

    def test_an_empty_array_without_a_clear_flag_keeps_the_cached_subset(self):
        """An unsure answer must not erase what earlier evidence established."""
        outer = self._outer({"action": "key_d",
                             "action_criteria": {"move_up": "x", "key_d": "y"},
                             "control_exclude": []})
        outer(self._obs(), "goal")
        self.assertFalse(outer.control_exclude_status["applied"])

    def test_an_explicit_clear_flag_removes_the_subset(self):
        outer = self._outer({"action": "key_d",
                             "action_criteria": {"move_up": "x", "key_d": "y"},
                             "control_exclude": [],
                             "control_exclude_clear": True})
        outer(self._obs(), "goal")
        self.assertTrue(outer.control_exclude_status["applied"])
        self.assertEqual(outer.control_exclude, [])

    def test_the_prompt_states_the_universe_and_the_caution(self):
        captured = {}

        def chat(prompt, token, **kwargs):
            captured["system"] = prompt[0]["content"]
            return {"choices": [{"message": {"content": json.dumps(
                {"action": "move_up", "action_criteria": {"move_up": "a", "key_d": "b"},
                 "control_exclude": []})}}]}

        outer = OuterEscalation(chat_fn=chat, generate_criteria=True)
        outer(self._obs(), "goal")
        self.assertIn("control_exclude", captured["system"])
        self.assertIn("state.input_universe", captured["system"])
        self.assertIn("empty array", captured["system"])


class PolicySubsetSinkTests(unittest.TestCase):
    def test_review_result_reaches_the_adapter_through_the_callback(self):
        seen = {}
        policy = JevPolicy(score_fn=lambda *args, **kwargs: {})
        policy.subset_sink = lambda exclude, note="": seen.update({"exclude": list(exclude), "note": note})
        obs = Observation(actions=[Action("move_up")],
                          state={"phase": "play", "input_universe": ["move_up"]})
        slow = SimpleNamespace(action_criteria={}, control_exclude=["key_q"], situation="rhythm game",
                               control_exclude_status={"applied": True})
        policy._accept_outer_result(obs, slow, None)
        self.assertEqual(seen["exclude"], ["key_q"])
        self.assertEqual(seen["note"], "rhythm game")

    def test_an_unaccepted_verdict_does_not_touch_the_adapter(self):
        policy = JevPolicy(score_fn=lambda *args, **kwargs: {})
        called = []
        policy.subset_sink = lambda exclude, note="": called.append(list(exclude))
        obs = Observation(actions=[Action("move_up")],
                          state={"phase": "play", "input_universe": ["move_up"]})
        slow = SimpleNamespace(action_criteria={}, control_exclude=[], situation="",
                               control_exclude_status={"applied": False})
        policy._accept_outer_result(obs, slow, None)
        self.assertEqual(called, [])

    def test_no_callback_means_no_crash_for_other_adapters(self):
        policy = JevPolicy(score_fn=lambda *args, **kwargs: {})
        obs = Observation(actions=[Action("move_up")],
                          state={"phase": "play", "input_universe": ["move_up"]})
        slow = SimpleNamespace(action_criteria={}, control_exclude=["key_q"], situation="",
                               control_exclude_status={"applied": True})
        policy._accept_outer_result(obs, slow, None)
        self.assertIsNone(policy.subset_sink)

    def test_fast_context_never_receives_the_full_input_universe(self):
        """The fast model may only name a legal action, so the universe stays out."""
        policy = JevPolicy(score_fn=lambda *args, **kwargs: {})
        policy.locked_goal = "goal"
        obs = Observation(actions=[Action("move_up")],
                          state={"phase": "play",
                                 "input_universe": ["move_up", "key_q"],
                                 "input_excluded": ["key_q"]})
        world = policy._world(obs, "goal", [], "")
        self.assertNotIn("key_q", world)
        self.assertNotIn("input_universe", world)


if __name__ == "__main__":
    unittest.main()
