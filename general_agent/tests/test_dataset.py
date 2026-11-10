"""Dataset assembly: which rows are usable, and where the frames point.

The collector writes one folder per game so a crash cannot corrupt earlier games.
Training wants the opposite shape, so ``build_dataset`` merges the folders and
marks each row. These tests pin the two decisions that are easy to get wrong and
expensive to notice:

* the default clean set keeps every labelled row, including the ones whose step
  had no numeric metric. Dropping them would silently trim whole trajectories,
  which is the opposite of what whole-trajectory collection is for;
* a frame reference is only written when the file is actually on disk, so a
  consumer never gets a path that 404s mid-training.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_dataset import attach_frames, load_rows, summarise  # noqa: E402


def row(game="g", run="r", role="executor", at=1.0, label=None, proven=True,
        frame_sha="", error="") -> dict:
    return {
        "schema": "fs2",
        "role": role,
        "game": game,
        "run_id": run,
        "step": 1,
        "trigger": "first_step",
        "model": "deepseek-v4.1-flash",
        "at": at,
        "label": label if label is not None else {"action": "right"},
        "frame_sha": frame_sha,
        "proven": proven,
        "from_reasoning": False,
        "error": error,
    }


def write_run(root: Path, game: str, run: str, rows: list[dict]) -> Path:
    folder = root / game / run
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "roles.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
                    encoding="utf-8")
    return folder


class LoadTests(unittest.TestCase):
    def test_rows_are_ordered_by_game_then_time(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_run(root, "b", "r", [row(game="b", at=2.0), row(game="b", at=1.0)])
            write_run(root, "a", "r", [row(game="a", at=5.0)])
            rows = load_rows(root)
        self.assertEqual([item["game"] for item in rows], ["a", "b", "b"])
        self.assertEqual([item["at"] for item in rows], [5.0, 1.0, 2.0])

    def test_blank_and_malformed_lines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            folder_path = write_run(root, "g", "r", [row()])
            with (folder_path / "roles.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")
                handle.write("{not json\n")
            rows = load_rows(root)
        self.assertEqual(len(rows), 1)

    def test_an_empty_root_yields_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_rows(Path(folder)), [])


class FrameLinkTests(unittest.TestCase):
    def test_an_existing_frame_becomes_a_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = root / "g" / "r" / "frames"
            frames.mkdir(parents=True)
            (frames / "abc.png").write_bytes(b"x")
            rows = [row(frame_sha="abc")]
            missing = attach_frames(rows, root)
        self.assertEqual(missing, 0)
        self.assertEqual(rows[0]["frame_path"], "g/r/frames/abc.png")

    def test_a_jpeg_frame_is_found_too(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = root / "g" / "r" / "frames"
            frames.mkdir(parents=True)
            (frames / "abc.jpg").write_bytes(b"x")
            rows = [row(frame_sha="abc")]
            attach_frames(rows, root)
        self.assertTrue(rows[0]["frame_path"].endswith("abc.jpg"))

    def test_a_missing_frame_is_counted_and_left_blank(self) -> None:
        # A row that names a frame which was never written must not receive a path:
        # training would fail on the first read instead of at manifest review.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "g" / "r").mkdir(parents=True)
            rows = [row(frame_sha="gone")]
            missing = attach_frames(rows, root)
        self.assertEqual(missing, 1)
        self.assertEqual(rows[0]["frame_path"], "")

    def test_rows_without_a_frame_are_not_counted_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [row(frame_sha="")]
            missing = attach_frames(rows, root)
        self.assertEqual(missing, 0)
        self.assertEqual(rows[0]["frame_path"], "")


class SummaryTests(unittest.TestCase):
    def test_label_rate_counts_validation_not_metric_availability(self) -> None:
        rows = [row(), row(proven=False), row(label={}, error="label rejected")]
        summary = summarise(rows)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["labelled_rows"], 2)
        self.assertEqual(summary["label_rate"], 0.6667)
        self.assertEqual(summary["unproven_rows"], 1)

    def test_reasoning_drafts_are_reported_as_their_own_reason(self) -> None:
        rows = [row(label={}, error="reasoning only"), row(label={}, error="label rejected")]
        summary = summarise(rows)
        self.assertEqual(summary["by_error"],
                         {"reasoning only": 1, "label rejected": 1})

    def test_roles_triggers_and_games_are_counted(self) -> None:
        rows = [
            row(role="executor"),
            row(role="planner", label={"plan": {}}),
            row(game="h"),
        ]
        summary = summarise(rows)
        self.assertEqual(summary["by_role"], {"executor": 2, "planner": 1})
        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["by_trigger"], {"first_step": 3})

    def test_unique_frames_are_counted_once_each(self) -> None:
        rows = [row(frame_sha="a"), row(frame_sha="a"), row(frame_sha="b"), row(frame_sha="")]
        self.assertEqual(summarise(rows)["unique_frames"], 2)

    def test_schema_and_model_sets_are_reported(self) -> None:
        summary = summarise([row()])
        self.assertEqual(summary["schemas"], ["fs2"])
        self.assertEqual(summary["models"], ["deepseek-v4.1-flash"])


if __name__ == "__main__":
    unittest.main()
