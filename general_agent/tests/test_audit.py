"""An audit that cannot be checked is just an opinion.

The audit has two audiences: a batch that must pass before training, and a person
reading the report. Both need the checks themselves to be tested, because a
false positive here is expensive in one direction -- it blocks a good batch -- and
a false negative is expensive in the other -- it lets a corrupt batch through.

The tests use hand-built runs rather than the real collector, so each rule can be
violated in isolation.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_dataset import audit  # noqa: E402


def system_prompt(role: str) -> str:
    return f"You are the {role.upper()} of a game-playing agent."


def executor_row(step=0, action="right", game="g", run="r", reply=None,
                 label=None, proven=True, error="", legal=("right", "wait"),
                 world_game="g", world_step=None) -> dict:
    payload = {
        "output_schema": "{}",
        "world": {
            "game": world_game,
            "goal": f"play {game}",
            "level": 1,
            "step": (step + 1) if world_step is None else world_step,
            "phase": "play",
            "image_attached": True,
            "legal_actions": [{"id": item, "label": item, "kind": "hold",
                               "role": "", "seconds": 0.15} for item in legal],
            "state": {},
            "progress": {},
        },
        "plan": {"schema": "fs2", "revision": 1, "issued_at_step": 1,
                 "briefing": "b", "stage_goal": "s", "tactics": "t",
                 "situation": "", "avoid": [], "exclude": [], "review_in": 8},
        "recent_actions": [],
        "repeat_run": 0,
    }
    body = reply if reply is not None else json.dumps(
        {"action": action, "why": "w", "confidence": 0.5, "expect": "e",
         "belief": "b", "stuck": False}
    )
    if label is None and not error:
        label = {"action": action, "why": "w", "confidence": 0.5, "expect": "e",
                 "belief": "b", "stuck": False}
    return {
        "schema": "fs2", "role": "executor", "game": game, "run_id": run,
        "goal": f"play {game}", "step": step, "plan_revision": 1,
        "trigger": "first_step", "model": "m", "at": float(step),
        "ms": 1, "usage": {}, "attempts": 1,
        "messages": [{"role": "system", "content": system_prompt("executor")},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "label": label or {}, "raw_reply": body, "frame_sha": "",
        "progress": {}, "proven": proven, "from_reasoning": False, "error": error,
    }


def planner_row(step=0, game="g", run="r", reply=None, label=None, proven=True,
                error="", legal=("right", "wait"), previous_revision=0,
                world_game="g", plan_extra=None, text="t") -> dict:
    payload = {
        "output_schema": "{}",
        "world": {
            "game": world_game, "goal": f"play {game}", "level": 1,
            "step": step + 1, "phase": "play", "image_attached": True,
            "legal_actions": [{"id": item, "label": item, "kind": "hold",
                               "role": "", "seconds": 0.15} for item in legal],
            "state": {}, "progress": {},
        },
        "current_plan": {"schema": "fs2", "revision": previous_revision,
                         "issued_at_step": 0, "briefing": "b", "stage_goal": "",
                         "tactics": "", "situation": "", "avoid": [], "exclude": [],
                         "review_in": 8},
        "executor_report": None,
        "recent_actions": [],
        "review_trigger": "first_step",
    }
    interpreted = {"briefing": "b", "stage_goal": "s", "tactics": text}
    body = reply if reply is not None else json.dumps(interpreted)
    store = {"schema": "fs2", "revision": previous_revision + 1,
             "issued_at_step": step + 1, "briefing": "b", "stage_goal": "s",
             "tactics": text, "situation": "", "avoid": [], "exclude": [],
             "review_in": 8}
    if plan_extra:
        store.update(plan_extra)
    if label is None and not error:
        label = {"plan": store, "verdict": "continue", "note": ""}
    return {
        "schema": "fs2", "role": "planner", "game": game, "run_id": run,
        "goal": f"play {game}", "step": step,
        "plan_revision": previous_revision + 1, "trigger": "first_step",
        "model": "m", "at": float(step), "ms": 1, "usage": {}, "attempts": 1,
        "messages": [{"role": "system", "content": system_prompt("planner")},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "label": label or {}, "raw_reply": body, "frame_sha": "", "progress": {},
        "proven": proven, "from_reasoning": False, "error": error,
    }


def write(root: Path, game: str, run: str, rows: list[dict]) -> None:
    folder = root / game / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "roles.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class AuditTests(unittest.TestCase):
    def run_audit(self, rows: list[dict], frames: Path | None = None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "runs"
            write(root, rows[0]["game"], rows[0]["run_id"], rows)
            target = Path(folder) / "frames"
            target.mkdir(parents=True, exist_ok=True)
            return audit(root, frames or target)

    def kinds(self, report) -> list[str]:
        return [check for check, _ in report.blockers]

    def test_a_clean_run_passes(self) -> None:
        report = self.run_audit([planner_row(0), executor_row(0)])
        self.assertEqual(report.blockers, [])

    def test_an_action_outside_the_legal_set_blocks(self) -> None:
        report = self.run_audit([
            planner_row(0),
            executor_row(0, action="jump", legal=("right",)),
        ])
        self.assertIn("reproducible", self.kinds(report))

    def test_a_label_that_cannot_be_rebuilt_blocks(self) -> None:
        # The stored label claims "right" but the reply says "wait": the row would
        # teach an answer the model never gave.
        row = executor_row(0, action="right",
                           reply=json.dumps({"action": "wait"}))
        report = self.run_audit([planner_row(0), row])
        self.assertIn("reproducible", self.kinds(report))

    def test_an_executor_row_with_a_planner_prompt_blocks(self) -> None:
        row = executor_row(0)
        row["messages"][0]["content"] = system_prompt("planner")
        report = self.run_audit([planner_row(0), row])
        self.assertIn("separation", self.kinds(report))

    def test_a_plan_carrying_an_action_blocks(self) -> None:
        row = planner_row(0, plan_extra={"action": "right"})
        report = self.run_audit([row])
        self.assertIn("separation", self.kinds(report))

    def test_naming_an_opaque_game_id_blocks(self) -> None:
        row = planner_row(0, legal=("jg_1", "jg_2"), text="press jg_1 now")
        report = self.run_audit([row])
        self.assertIn("authority", self.kinds(report))

    def test_naming_a_generic_word_only_warns(self) -> None:
        # "confirm" is an English word as much as a control; treating it as a
        # violation would reject most of the batch over a stylistic habit.
        row = planner_row(0, legal=("confirm", "enter"), text="use a confirm-style key")
        report = self.run_audit([row])
        self.assertNotIn("authority", self.kinds(report))
        self.assertIn("authority", [check for check, _ in report.warnings])

    def test_a_blank_game_in_the_prompt_blocks(self) -> None:
        row = executor_row(0, world_game="")
        report = self.run_audit([planner_row(0), row])
        self.assertIn("payload", self.kinds(report))

    def test_a_planner_revision_out_of_sequence_blocks(self) -> None:
        row = planner_row(0, previous_revision=5)
        report = self.run_audit([row])
        self.assertIn("reproducible", self.kinds(report))

    def test_a_missing_step_with_no_recorded_cause_blocks(self) -> None:
        report = self.run_audit([
            planner_row(0), executor_row(0),
            executor_row(2),
        ])
        self.assertIn("continuity", self.kinds(report))

    def test_a_rejected_step_that_was_recorded_is_not_a_gap(self) -> None:
        # A rejected call still consumed a step number; the row is kept with an
        # empty label, which is exactly what whole-trajectory collection means.
        report = self.run_audit([
            planner_row(0), executor_row(0),
            executor_row(1, label={}, error="reasoning only"),
            executor_row(2),
        ])
        self.assertNotIn("continuity", self.kinds(report))

    def test_duplicate_steps_block(self) -> None:
        report = self.run_audit([
            planner_row(0), executor_row(0), executor_row(0),
        ])
        self.assertIn("continuity", self.kinds(report))

    def test_a_reasoning_reply_longer_than_its_limit_blocks(self) -> None:
        row = executor_row(0)
        row["label"]["why"] = "x" * 400
        report = self.run_audit([planner_row(0), row])
        self.assertIn("limits", self.kinds(report))

    def test_a_missing_frame_blocks(self) -> None:
        row = executor_row(0)
        row["frame_sha"] = "absent"
        report = self.run_audit([planner_row(0), row])
        self.assertIn("frames", self.kinds(report))

    def test_an_entirely_unproven_batch_warns(self) -> None:
        report = self.run_audit([planner_row(0, proven=False),
                                 executor_row(0, proven=False)])
        self.assertIn("supervision", [check for check, _ in report.warnings])

    def test_the_step_offset_between_world_and_row_is_recorded(self) -> None:
        # The runner counts from 0 and the policy from 1; a constant offset is
        # expected, an inconsistent one is not.
        report = self.run_audit([planner_row(0), executor_row(0)])
        self.assertEqual(dict(report.offsets), {1: 2})


if __name__ == "__main__":
    unittest.main()
