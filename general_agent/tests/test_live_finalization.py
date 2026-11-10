import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from general_agent.contract import PlanDirective
from general_agent.live import LiveRun
from general_agent.types import EpisodeResult


class LiveFinalizationTests(unittest.TestCase):
    def fixture(self, folder):
        run = object.__new__(LiveRun)
        run.game = "fixture"
        run.goal = "finish"
        run.run_id = "trial"
        run.folder = Path(folder)
        run.policy = SimpleNamespace(model="executor-model", plan=PlanDirective(stage_goal="g"))
        run._stop = False
        run.running = True
        run.error = ""
        run.adapter = SimpleNamespace(config={}, executions=[], _state={"done": False})
        run.outer = SimpleNamespace(
            model="planner-model", revisions=3, error="", plan=PlanDirective(stage_goal="g")
        )
        run.client = SimpleNamespace(calls=[object(), object()])
        run.sft = SimpleNamespace(summary=lambda: {"executor_records": 1})
        return run

    def test_running_flag_stays_true_until_summary_is_fully_written(self):
        original = Path.write_text
        with tempfile.TemporaryDirectory() as folder:
            run = self.fixture(folder)
            during = []

            def write(path, *args, **kwargs):
                during.append(run.running)
                return original(path, *args, **kwargs)

            with patch("general_agent.live.run_episode", return_value=EpisodeResult(False, 2, 3, stopped=True)), patch.object(Path, "write_text", write):
                run._run()
            self.assertEqual(during, [True])
            self.assertFalse(run.running)
            summary = json.loads((Path(folder) / "summary.json").read_text())
            self.assertTrue(summary["result"]["stopped"])
            self.assertEqual(summary["planner_revisions"], 3)
            self.assertEqual(summary["model_calls"], 2)
            self.assertFalse((Path(folder) / "summary.json.tmp").exists())

    def test_summary_io_failure_is_reported_without_leaving_running_stuck(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.fixture(folder)
            with patch("general_agent.live.run_episode", return_value=EpisodeResult(False, 0, 0)), patch.object(Path, "write_text", side_effect=OSError("disk unavailable")):
                run._run()
            self.assertFalse(run.running)
            self.assertIn("summary write failed", run.error)


if __name__ == "__main__":
    unittest.main()
