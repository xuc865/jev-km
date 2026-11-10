"""Tests for the distillation writer.

The old writer wrapped a scorer and monkey-patched the planner's ``__call__``.
That seam no longer exists, so these tests check the replacement contract: the
policies stamp an accepted reply onto the ``ModelCall``, and the recorder writes
every exchange while marking which rows carry a validated label and which have a
readable game metric behind them.

Two of these tests encode a reversal. An earlier version of the writer dropped
rows it could not prove, on the theory that an unverifiable sample is worse than a
missing one. Whole-trajectory collection needs the opposite: a filtered dataset can
always be derived from a complete one, never the reverse.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from general_agent.client import ModelCall
from general_agent.contract import SCHEMA_VERSION
from general_agent.policy import ExecutorPolicy
from general_agent.sft import RoleRecorder, save_frames
from general_agent.types import Action, Observation

from .fake_models import ScriptedClient


def call(role="executor", text='{"action":"right"}', error="", label=None, frame_sha="") -> ModelCall:
    return ModelCall(
        role=role,
        model="model-x",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        text=text,
        ms=12,
        usage={"input_tokens": 3},
        error=error,
        schema=SCHEMA_VERSION,
        label=label or {},
        frame_sha=frame_sha,
    )


def rows(recorder: RoleRecorder) -> list[dict]:
    return [json.loads(line) for line in recorder.path.read_text().splitlines()]


class RecorderTests(unittest.TestCase):
    def build(self, folder, **kwargs) -> RoleRecorder:
        recorder = RoleRecorder("game", "run", goal="goal", root=Path(folder), **kwargs)
        recorder.set_progress({"metric": "score", "value": 10})
        return recorder

    def test_labelled_and_unlabelled_calls_are_both_written(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            self.assertTrue(recorder.record(call(label={"action": "right"})))
            self.assertFalse(recorder.record(call(label={})))
            written = rows(recorder)
        self.assertEqual(len(written), 2)
        self.assertEqual(written[0]["label"], {"action": "right"})
        self.assertEqual(written[0]["error"], "")
        self.assertEqual(written[1]["label"], {})
        self.assertEqual(written[1]["error"], "label rejected")

    def test_transport_failure_and_validation_failure_are_counted_apart(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.record(call(error="timeout"))
            recorder.record(call(label={}))
            summary = recorder.summary()
            written = rows(recorder)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["invalid"], 1)
        self.assertEqual(written[0]["error"], "timeout")
        self.assertEqual(written[1]["error"], "label rejected")

    def test_unknown_progress_is_written_but_marked_unproven(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = RoleRecorder("game", "run", goal="g", root=Path(folder))
            recorder.set_progress({"metric": "score", "value": None})
            self.assertTrue(recorder.record(call(label={"action": "right"})))
            summary = recorder.summary()
            written = rows(recorder)
        self.assertEqual(summary["unproven"], 1)
        self.assertFalse(written[0]["proven"])
        # The label is still valid; only the game's metric is missing.
        self.assertEqual(written[0]["label"], {"action": "right"})

    def test_proven_flag_tracks_the_metric_not_the_reply(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.record(call(label={"action": "right"}))
            recorder.set_progress({"metric": "score", "value": None})
            recorder.record(call(label={"action": "left"}))
            written = rows(recorder)
        self.assertTrue(written[0]["proven"])
        self.assertFalse(written[1]["proven"])

    def test_planner_records_are_counted_by_role(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.record(call(role="planner", label={"plan": {}}))
            recorder.record(call(role="executor", label={"action": "right"}))
            summary = recorder.summary()
        self.assertEqual(summary["planner_records"], 1)
        self.assertEqual(summary["executor_records"], 1)

    def test_episode_context_and_frame_hash_are_stamped_on_each_row(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.step = 7
            recorder.plan_revision = 3
            recorder.trigger = "low_confidence"
            recorder.record(call(label={"action": "right"}, frame_sha="abc123"))
            row = rows(recorder)[0]
        self.assertEqual(row["step"], 7)
        self.assertEqual(row["plan_revision"], 3)
        self.assertEqual(row["trigger"], "low_confidence")
        self.assertEqual(row["game"], "game")
        self.assertEqual(row["frame_sha"], "abc123")

    def test_drain_pulls_from_the_client_and_empties_it(self) -> None:
        class Client:
            def __init__(self):
                self.pending = [call(label={"action": "right"}), call(error="boom")]
                self.drained = 0

            def drain_calls(self):
                self.drained += 1
                pending = self.pending
                self.pending = []
                return pending

        with tempfile.TemporaryDirectory() as folder:
            client = Client()
            recorder = self.build(folder)
            self.assertEqual(recorder.drain(client), 1)
            self.assertEqual(len(rows(recorder)), 2)
            self.assertEqual(recorder.drain(client), 0)
            self.assertEqual(client.drained, 2)

    def test_explicit_label_overrides_the_carried_one(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.record(call(label={"action": "right"}), label={"action": "left"})
            self.assertEqual(rows(recorder)[0]["label"], {"action": "left"})

    def test_reasoning_only_reply_is_recorded_as_a_budget_problem(self) -> None:
        # The distinction matters for the fix: a reasoning-only reply means no
        # answer was ever written, so raising the budget helps, while a malformed
        # answer means the prompt or the contract is wrong.
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            draft = call(role="planner", text="let me think")
            draft.from_reasoning = True
            recorder.record(draft)
            summary = recorder.summary()
            row = rows(recorder)[0]
        self.assertEqual(row["error"], "reasoning only")
        self.assertTrue(row["from_reasoning"])
        self.assertEqual(summary["reasons"], {"reasoning only": 1})
        self.assertEqual(summary["invalid"], 1)

    def test_label_rejection_and_reasoning_only_are_counted_separately(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            draft = call(role="planner", text="thinking")
            draft.from_reasoning = True
            recorder.record(draft)
            recorder.record(call(role="planner", label={}))
            summary = recorder.summary()
        self.assertEqual(summary["invalid"], 2)
        self.assertEqual(summary["reasons"], {"reasoning only": 1, "label rejected": 1})

    def test_the_world_carries_the_game_name_from_the_observation(self) -> None:
        # A blank game name in the prompt while the row records one is a
        # train/serve mismatch: the student is asked to play an unnamed game and
        # then scored as if it had been told which one.
        obs = Observation(actions=[Action("right")], game="demo-1")
        world = ExecutorPolicy(client=ScriptedClient(), model="m").build_world(obs, "g")
        self.assertEqual(world.game, "demo-1")

    def test_state_game_is_used_when_the_field_is_absent(self) -> None:
        # Older adapters put the identifier in state rather than on the observation.
        obs = Observation(actions=[Action("right")], state={"game": "legacy"})
        world = ExecutorPolicy(client=ScriptedClient(), model="m").build_world(obs, "g")
        self.assertEqual(world.game, "legacy")

    def test_the_observation_field_wins_over_state(self) -> None:
        obs = Observation(actions=[Action("right")], game="explicit",
                          state={"game": "from-state"})
        world = ExecutorPolicy(client=ScriptedClient(), model="m").build_world(obs, "g")
        self.assertEqual(world.game, "explicit")

    def test_a_game_name_never_leaks_into_the_pruned_state(self) -> None:
        obs = Observation(actions=[Action("right")], game="g1", state={"hp": 3})
        world = ExecutorPolicy(client=ScriptedClient(), model="m").build_world(obs, "g")
        self.assertEqual(world.state, {"hp": 3})

    def test_messages_are_stored_exactly_as_sent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            original = call(label={"action": "right"})
            recorder.record(original)
            row = rows(recorder)[0]
        self.assertEqual(row["messages"], original.messages)
        self.assertEqual(row["raw_reply"], original.text)


class FrameStoreTests(unittest.TestCase):
    PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="

    def test_frames_are_written_once_per_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            seen: set[str] = set()
            first = save_frames(Path(folder), self.PNG, seen)
            second = save_frames(Path(folder), self.PNG, seen)
            files = list(Path(folder).glob("*.png"))
        self.assertEqual(first, second)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stem, first)

    def test_distinct_frames_get_distinct_files(self) -> None:
        other = self.PNG[:-8] + "AAAAAABJRU5ErkJggg=="
        with tempfile.TemporaryDirectory() as folder:
            seen: set[str] = set()
            first = save_frames(Path(folder), self.PNG, seen)
            second = save_frames(Path(folder), other, seen)
            files = list(Path(folder).glob("*.png"))
        self.assertNotEqual(first, second)
        self.assertEqual(len(files), 2)

    def test_empty_and_malformed_input_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            seen: set[str] = set()
            self.assertEqual(save_frames(Path(folder), "", seen), "")
            self.assertEqual(save_frames(Path(folder), "https://example.com/x.png", seen), "")
            self.assertEqual(list(Path(folder).glob("*")), [])

    def test_dedup_does_not_depend_on_the_folder_already_existing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "frames"
            self.assertFalse(target.exists())
            digest = save_frames(target, self.PNG, set())
            self.assertTrue((target / f"{digest}.png").is_file())


if __name__ == "__main__":
    unittest.main()
