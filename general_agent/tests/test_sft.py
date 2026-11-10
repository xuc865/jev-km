"""Tests for the distillation writer.

The old writer wrapped a scorer and monkey-patched the planner's ``__call__``.
That seam no longer exists, so these tests check the replacement contract: the
policies stamp an accepted reply onto the ``ModelCall``, and the recorder writes
exactly those calls, distinguishing transport failures from validation failures.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from general_agent.client import ModelCall
from general_agent.contract import SCHEMA_VERSION
from general_agent.sft import RoleRecorder


def call(role="executor", text='{"action":"right"}', error="", label=None) -> ModelCall:
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
    )


class RecorderTests(unittest.TestCase):
    def build(self, folder, **kwargs) -> RoleRecorder:
        recorder = RoleRecorder("game", "run", goal="goal", root=Path(folder), **kwargs)
        recorder.set_progress({"metric": "score", "value": 10})
        return recorder

    def test_only_labelled_calls_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            self.assertTrue(recorder.record(call(label={"action": "right"})))
            self.assertFalse(recorder.record(call(label={})))
            rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], {"action": "right"})
        self.assertEqual(rows[0]["schema"], SCHEMA_VERSION)
        self.assertEqual(recorder.counts["executor"], 1)

    def test_transport_failure_is_counted_separately_from_validation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.record(call(error="timeout"))
            recorder.record(call(label={}))
            summary = recorder.summary()
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["executor_records"], 0)

    def test_unknown_progress_is_recorded_as_unproven(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = RoleRecorder("game", "run", goal="g", root=Path(folder))
            recorder.set_progress({"metric": "score", "value": None})
            self.assertFalse(recorder.record(call(label={"action": "right"})))
            summary = recorder.summary()
            self.assertEqual(summary["unproven"], 1)
            self.assertEqual(summary["executor_records"], 0)

    def test_planner_records_are_counted_by_role(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.record(call(role="planner", label={"plan": {}}))
            recorder.record(call(role="executor", label={"action": "right"}))
            summary = recorder.summary()
        self.assertEqual(summary["planner_records"], 1)
        self.assertEqual(summary["executor_records"], 1)

    def test_episode_context_is_stamped_on_each_row(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.step = 7
            recorder.plan_revision = 3
            recorder.trigger = "low_confidence"
            recorder.record(call(label={"action": "right"}))
            row = json.loads(recorder.path.read_text().splitlines()[0])
        self.assertEqual(row["step"], 7)
        self.assertEqual(row["plan_revision"], 3)
        self.assertEqual(row["trigger"], "low_confidence")
        self.assertEqual(row["game"], "game")
        self.assertEqual(row["goal"], "goal")

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
            self.assertEqual(client.pending, [])
            self.assertEqual(recorder.drain(client), 0)
            self.assertEqual(client.drained, 2)

    def test_explicit_label_overrides_the_carried_one(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            recorder.record(call(label={"action": "right"}), label={"action": "left"})
            row = json.loads(recorder.path.read_text().splitlines()[0])
        self.assertEqual(row["label"], {"action": "left"})

    def test_messages_are_stored_exactly_as_sent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.build(folder)
            original = call(label={"action": "right"})
            recorder.record(original)
            row = json.loads(recorder.path.read_text().splitlines()[0])
        self.assertEqual(row["messages"], original.messages)
        self.assertEqual(row["raw_reply"], original.text)


if __name__ == "__main__":
    unittest.main()
