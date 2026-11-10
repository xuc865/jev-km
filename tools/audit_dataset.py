"""Audit a collected SFT batch and report what would poison training.

The point is not to count rows. It is to re-derive every label from the recorded
reply using the same parser the runtime uses, and to check that the two agree. A
row whose stored label cannot be reproduced from its stored reply is worse than a
missing row: it teaches a mapping the runtime never had, and it does so silently.

Checks, grouped by what kind of damage they prevent:

structure      required fields, types, schema version, message shape
separation     the two roles must not leak into each other -- an executor row
               carrying plan text, or a planner row carrying an action
reproducible   label == parse(raw_reply), action inside that step's legal set,
               plan merge replays to the stored directive
continuity     steps rise without gaps inside a run, plan revisions never go
               backwards, `at` never goes backwards
authority      the planner must not name a single control, which is the one rule
               that makes the two-role split real rather than decorative
hygiene        exact duplicates, identical prompts with conflicting labels,
               text over the contract limits, frames that do not resolve

Exit code is 1 when any blocker is found, so this can gate a batch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_agent.contract import (  # noqa: E402
    MAX_REVIEW_INTERVAL,
    MIN_REVIEW_INTERVAL,
    TEXT_LIMITS,
    PlanDirective,
    load_json_object,
    parse_executor_reply,
    parse_planner_reply,
)

EXECUTOR_KEYS = {"action", "why", "confidence", "expect", "belief", "stuck"}
PLANNER_KEYS = {"plan", "verdict", "note"}
PLAN_KEYS = {"schema", "revision", "issued_at_step", "briefing", "stage_goal",
             "tactics", "situation", "avoid", "exclude", "review_in"}
SINGLE_ACTION_NAMES = {"action", "act", "next_action", "key", "control", "press"}


class Report:
    """Collects findings so one run reports everything, not just the first fault."""

    def __init__(self) -> None:
        self.blockers: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self.stats: dict[str, object] = {}
        self.offsets: Counter = Counter()

    def block(self, check: str, detail: str) -> None:
        self.blockers.append((check, detail))

    def warn(self, check: str, detail: str) -> None:
        self.warnings.append((check, detail))

    def observe_offset(self, offset: int) -> None:
        self.offsets[offset] += 1

    def count(self, key: str, value) -> None:
        self.stats[key] = value


def load_runs(root: Path) -> dict[tuple[str, str], list[dict]]:
    """Rows grouped by run, in file order.

    File order is the chronological order the records were written. Re-reading
    through a sorted merged file would lose it whenever two rows share a
    timestamp, which is exactly the case a replay check needs to be exact about.
    """
    runs: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(root.glob("*/*/roles.jsonl")):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        if rows:
            runs[(rows[0].get("game", ""), rows[0].get("run_id", ""))] = rows
    return runs


def payload_of(row: dict) -> dict | None:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    try:
        parsed = json.loads(messages[1].get("content") or "{}")
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def legal_ids(payload: dict | None) -> set[str]:
    if not payload:
        return set()
    actions = ((payload.get("world") or {}).get("legal_actions")) or []
    return {str(item.get("id")) for item in actions if isinstance(item, dict)}


def check_structure(row: dict, tag: str, report: Report) -> None:
    required = ["schema", "role", "game", "run_id", "step", "messages", "label",
                "raw_reply", "frame_sha", "proven", "error"]
    missing = [key for key in required if key not in row]
    if missing:
        report.block("structure", f"{tag}: missing {missing}")
        return
    if row["schema"] != "fs2":
        report.block("structure", f"{tag}: schema {row['schema']!r}")
    if row["role"] not in {"executor", "planner"}:
        report.block("structure", f"{tag}: role {row['role']!r}")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        report.block("structure", f"{tag}: {len(messages) if isinstance(messages, list) else '?'} messages")
        return
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        report.block("structure", f"{tag}: message roles {[m.get('role') for m in messages]}")
    if not str(messages[0].get("content") or "").strip():
        report.block("structure", f"{tag}: empty system prompt")
    if not isinstance(row["step"], int) or row["step"] < 0:
        report.block("structure", f"{tag}: step {row['step']!r}")
    if not isinstance(row["label"], dict):
        report.block("structure", f"{tag}: label is not an object")


def check_separation(row: dict, tag: str, report: Report) -> None:
    system = str(row["messages"][0].get("content") or "")
    label = row["label"]
    if row["role"] == "executor":
        if "EXECUTOR" not in system or "PLANNER" in system:
            report.block("separation", f"{tag}: executor row has the wrong system prompt")
        if label:
            extra = set(label) - EXECUTOR_KEYS
            if extra:
                report.block("separation", f"{tag}: executor label has {sorted(extra)}")
            if "action" not in label:
                report.block("separation", f"{tag}: executor label has no action")
    else:
        if "PLANNER" not in system or "EXECUTOR" in system:
            report.block("separation", f"{tag}: planner row has the wrong system prompt")
        if label:
            extra = set(label) - PLANNER_KEYS
            if extra:
                report.block("separation", f"{tag}: planner label has {sorted(extra)}")
            plan = label.get("plan")
            if not isinstance(plan, dict):
                report.block("separation", f"{tag}: planner label has no plan object")
            else:
                stray = set(plan) - PLAN_KEYS
                if stray:
                    report.block("separation", f"{tag}: plan has {sorted(stray)}")
                # The whole point of the split: the planner advises, it does not
                # pick. A plan that carries an action turns the slow tier into a
                # second executor and the two roles stop being distinguishable in
                # the distilled data.
                named = SINGLE_ACTION_NAMES & set(plan)
                if named:
                    report.block("separation", f"{tag}: plan carries {sorted(named)}")


def check_limits(row: dict, tag: str, report: Report) -> None:
    label = row["label"]
    if not label:
        return
    if row["role"] == "executor":
        for key in ("why", "expect", "belief"):
            value = str(label.get(key) or "")
            if len(value) > TEXT_LIMITS[key]:
                report.block("limits", f"{tag}: {key} {len(value)} > {TEXT_LIMITS[key]}")
        confidence = label.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            report.block("limits", f"{tag}: confidence {confidence!r}")
        if not isinstance(label.get("stuck"), bool):
            report.block("limits", f"{tag}: stuck {label.get('stuck')!r}")
        return
    plan = label.get("plan") or {}
    for key in ("briefing", "stage_goal", "tactics", "situation"):
        value = str(plan.get(key) or "")
        if len(value) > TEXT_LIMITS[key]:
            report.block("limits", f"{tag}: {key} {len(value)} > {TEXT_LIMITS[key]}")
    review_in = plan.get("review_in")
    if not isinstance(review_in, int) or not MIN_REVIEW_INTERVAL <= review_in <= MAX_REVIEW_INTERVAL:
        report.block("limits", f"{tag}: review_in {review_in!r}")
    if label.get("verdict") not in {"continue", "pivot", "finish"}:
        report.block("limits", f"{tag}: verdict {label.get('verdict')!r}")
    if len(str(label.get("note") or "")) > 200:
        report.block("limits", f"{tag}: note too long")
    for key in ("avoid", "exclude"):
        value = plan.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            report.block("limits", f"{tag}: {key} {type(value).__name__}")
        elif len(set(value)) != len(value):
            report.block("limits", f"{tag}: {key} has duplicates")


def check_payload(row: dict, tag: str, report: Report) -> dict | None:
    payload = payload_of(row)
    if payload is None:
        report.block("payload", f"{tag}: user content is not a JSON object")
        return None
    keys = list(payload)
    if not keys or keys[0] != "output_schema":
        report.block("payload", f"{tag}: first key is {keys[0] if keys else None!r}")
    if row["role"] == "executor":
        expected = ["output_schema", "world", "plan", "recent_actions", "repeat_run"]
    else:
        expected = ["output_schema", "world", "current_plan", "executor_report",
                    "recent_actions", "review_trigger"]
    if keys != expected:
        report.block("payload", f"{tag}: keys {keys}")
    world = payload.get("world") or {}
    if not legal_ids(payload):
        report.block("payload", f"{tag}: no legal actions offered")
    # Two numbering conventions meet here: the runner counts executor steps from 0
    # (`for step in range(max_steps)`) while the policy increments its own counter
    # before building the world, so `world.step` is 1-based. Comparing them
    # directly reports a fault on every row. What must hold is that the offset is
    # the same constant everywhere and that the world step is the later of the two.
    offset = world.get("step", 0) - int(row["step"])
    report.observe_offset(offset)
    if offset < 1:
        report.block("payload", f"{tag}: world.step {world.get('step')} does not lead row step {row['step']}")
    if world.get("game") != row["game"]:
        # The prompt block is what the model reads; the row field is only metadata.
        # A blank game in the prompt is the more serious fault, so it is reported
        # separately from a mismatch.
        if not world.get("game"):
            report.block("payload", f"{tag}: world.game is blank in the prompt")
        else:
            report.block("payload", f"{tag}: world.game {world.get('game')} vs {row['game']}")
    if world.get("goal") != row["goal"]:
        report.block("payload", f"{tag}: world.goal differs from row goal")
    return payload


def check_executor(row: dict, tag: str, payload: dict, report: Report) -> None:
    label = row["label"]
    if not label:
        return
    legal = legal_ids(payload)
    action = str(label.get("action") or "")
    if action not in legal:
        report.block("reproducible", f"{tag}: action {action!r} not in legal set")
    if row.get("error"):
        report.block("reproducible", f"{tag}: labelled row has error {row['error']!r}")
    rebuilt = parse_executor_reply(str(row.get("raw_reply") or ""), legal)
    if rebuilt is None:
        report.block("reproducible", f"{tag}: label cannot be rebuilt from raw_reply")
        return
    if rebuilt.to_wire() != {
        "action": label.get("action"),
        "why": label.get("why", ""),
        "confidence": round(float(label.get("confidence") or 0.0), 3),
        "expect": label.get("expect", ""),
        "belief": label.get("belief", ""),
        "stuck": label.get("stuck", False),
    }:
        report.block("reproducible", f"{tag}: stored label differs from reparse")


def check_planner(row: dict, tag: str, payload: dict, previous: PlanDirective,
                  report: Report) -> PlanDirective:
    """Replay the planner reply through the runtime parser and compare.

    ``step`` is passed as the world step, not the row step: the runtime stamps
    ``issued_at_step`` from the policy's own counter, which is 1-based. Feeding
    the 0-based row step here would rebuild every plan off by one and report a
    fault on every planner row.
    """
    label = row["label"]
    if not label:
        return previous
    plan = label.get("plan") or {}
    if row.get("error"):
        report.block("reproducible", f"{tag}: labelled row has error {row['error']!r}")
    world_step = int((payload.get("world") or {}).get("step") or row["step"])
    rebuilt = parse_planner_reply(str(row.get("raw_reply") or ""), previous,
                                 world_step, legal_ids(payload))
    if rebuilt is None:
        report.block("reproducible", f"{tag}: planner label cannot be rebuilt")
        return previous
    stored = rebuilt.plan.to_wire()
    for key, value in stored.items():
        if plan.get(key) != value:
            report.block(
                "reproducible",
                f"{tag}: {key} differs (stored {str(plan.get(key))[:40]!r} vs "
                f"rebuilt {str(value)[:40]!r})",
            )
            break
    if plan.get("revision") != previous.revision + 1:
        report.block("reproducible",
                     f"{tag}: revision {plan.get('revision')} after {previous.revision}")
    if int(plan.get("issued_at_step") or 0) != world_step:
        report.block("reproducible",
                     f"{tag}: issued_at_step {plan.get('issued_at_step')} vs world step {world_step}")
    return rebuilt.plan


OPAQUE_ID = re.compile(r"^(jg_\d+|key_[a-z0-9]+|digit_\d+|k\d+)$")


def check_authority(row: dict, tag: str, payload: dict, report: Report,
                    counter: Counter) -> None:
    """The planner must not name a single control.

    Two grades, because the two grades mean different things for training:

    * naming an opaque game-specific id (`jg_13`, `key_j`) is a hard violation.
      Such an id only exists as a control, so writing it is the planner taking
      over the executor's decision, and a student trained on it learns to follow
      an order instead of choosing.
    * naming a generic control word (`confirm`, `enter`, `move_left`) is a
      warning. Those words carry ordinary English meaning, and a plan that says
      "probe with confirm"-style keys is describing a family, not issuing an
      order. Flagging them as violations would reject most of the batch over a
      stylistic habit.
    """
    label = row["label"]
    if not label or row["role"] != "planner":
        return
    plan = label.get("plan") or {}
    text = " ".join(str(plan.get(key) or "") for key in
                    ("briefing", "stage_goal", "tactics", "situation"))
    text += " " + str(label.get("note") or "")
    ids = {item for item in legal_ids(payload) if len(item) >= 3}
    hard: list[str] = []
    soft: list[str] = []
    for item in sorted(ids):
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(item)}(?![A-Za-z0-9_])", text):
            continue
        if OPAQUE_ID.match(item):
            hard.append(item)
        else:
            soft.append(item)
    if hard:
        counter[row["game"]] += 1
        report.block("authority", f"{tag}: plan names game-specific {hard[:4]}")
    elif soft:
        counter[">soft"] += 1
        report.warn("authority", f"{tag}: plan mentions control words {soft[:3]}")


def check_continuity(rows: list[dict], tag: str, report: Report) -> None:
    """Order and monotonicity, not completeness.

    A missing step is not automatically a defect: a rejected call still consumed a
    step number, and by design the rejected row is kept with an empty label. What
    must hold is that recorded steps never repeat, never go backwards, and that the
    number of unlabelled gaps matches the number of rejected calls actually present
    in the file. A gap with no recorded cause would mean a row was dropped, which is
    exactly what whole-trajectory collection forbids.
    """
    executor = [row for row in rows if row["role"] == "executor"]
    steps = [row["step"] for row in executor]
    if steps != sorted(steps):
        report.block("continuity", f"{tag}: executor steps out of order")
    duplicated = [step for step, n in Counter(steps).items() if n > 1]
    if duplicated:
        report.block("continuity", f"{tag}: duplicate steps {duplicated[:6]}")
    accounted = {row["step"] for row in executor if not row["label"]}
    if steps:
        holes = set(range(min(steps), max(steps) + 1)) - set(steps)
        if holes:
            report.block("continuity", f"{tag}: steps missing entirely {sorted(holes)[:6]}")
        labelled_gaps = set(range(min(steps), max(steps) + 1)) - {
            row["step"] for row in executor if row["label"]
        }
        unexplained = labelled_gaps - accounted
        if unexplained:
            report.block("continuity",
                         f"{tag}: unlabelled steps with no recorded cause "
                         f"{sorted(unexplained)[:6]}")
    times = [float(row.get("at") or 0) for row in rows]
    if times != sorted(times):
        report.block("continuity", f"{tag}: timestamps go backwards")
    revisions = [int((row["label"].get("plan") or {}).get("revision") or 0)
                 for row in rows if row["role"] == "planner" and row["label"]]
    if revisions != sorted(revisions):
        report.block("continuity", f"{tag}: revisions go backwards {revisions[:8]}")
    if revisions and revisions[0] != 1:
        report.warn("continuity", f"{tag}: first revision is {revisions[0]}, not 1")


def check_plan_age(rows: list[dict], tag: str, report: Report) -> None:
    """A step must never read a plan issued in its own future.

    Both sides of the comparison are taken on the policy's 1-based counter: the
    plan's ``issued_at_step`` and the ``world.step`` of the executor row that read
    it. Using the 0-based row step here would make every step look like it
    preceded its own plan.
    """
    current = 0
    for row in rows:
        plan = (row["label"].get("plan") or {}) if row["role"] == "planner" else None
        if plan:
            current = int(plan.get("issued_at_step") or 0)
            continue
        if row["role"] != "executor" or not row["label"]:
            continue
        stamped = int(row.get("plan_revision") or 0)
        if stamped < 0:
            report.block("age", f"{tag}: negative plan_revision at step {row['step']}")
        payload = payload_of(row)
        world_step = int(((payload or {}).get("world") or {}).get("step") or 0)
        if current and world_step and world_step < current:
            report.block("age", f"{tag}: world step {world_step} precedes plan issued at {current}")


def check_duplicates(all_rows: list[dict], report: Report) -> None:
    seen: dict[str, int] = defaultdict(int)
    for row in all_rows:
        key = json.dumps(
            [row.get("game"), row.get("run_id"), row.get("role"), row.get("step")],
            sort_keys=True,
        )
        seen[key] += 1
    dupes = {key: n for key, n in seen.items() if n > 1}
    if dupes:
        report.block("duplicates", f"{len(dupes)} duplicated (game, run, role, step) keys")
    prompts: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        if not row.get("label"):
            continue
        digest = json.dumps(row.get("messages"), sort_keys=True, ensure_ascii=False)
        prompts[digest].add(json.dumps(row["label"], sort_keys=True, ensure_ascii=False))
    conflicts = [items for items in prompts.values() if len(items) > 1]
    report.count("identical_prompts_with_conflicting_labels", len(conflicts))
    report.count("distinct_prompts", len(prompts))
    if conflicts:
        report.warn("duplicates", f"{len(conflicts)} prompts carry different labels")


def check_supervision(all_rows: list[dict], report: Report) -> None:
    """How much of the batch can be scored against the game at all.

    ``proven`` is false when the game exposed no readable metric that step, so the
    label cannot be checked against anything the game reported. That does not make
    the row wrong -- the action is still the one the model chose, and the trajectory
    is still whole -- but a batch that is mostly unproven cannot be used to measure
    whether a student improved, only to imitate behaviour.

    Games that are unproven for their entire run are collected separately, because
    they are a property of the game rather than of the collection: such a game needs
    a probe spec before its data can be scored, and no amount of re-running fixes it.
    """
    by_run: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in all_rows:
        by_run[(str(row.get("game")), str(row.get("run_id")))].append(bool(row.get("proven")))
    full = sorted({game for (game, _), flags in by_run.items() if not any(flags)})
    report.count("runs_with_no_metric_at_all", len(full))
    report.count("games_with_no_metric_at_all", full)
    total = len(by_run)
    if total and len(full) > total * 0.25:
        report.warn(
            "supervision",
            f"{len(full)} of {total} runs expose no readable metric; their labels can "
            "only be imitated, not scored",
        )
    partial = [
        key for key, flags in by_run.items() if any(flags) and not all(flags)
    ]
    report.count("runs_with_partial_metric_coverage", len(partial))


def check_frames(all_rows: list[dict], frames_root: Path, report: Report) -> None:
    """Resolve every frame reference against one of the two layouts on disk.

    Frames live in two shapes depending on what produced them: the collector writes
    ``<root>/<game>/<run>/frames/<sha>.png`` and ``build_dataset`` mirrors them to
    ``<frames>/<game>/<sha>.png``. Checking only one layout reports every row as
    broken against the other, which is how a real missing frame gets lost in noise.
    """
    if not frames_root.is_dir():
        report.warn("frames", f"{frames_root} does not exist, skipping")
        return
    flat: dict[str, set[str]] = defaultdict(set)
    nested: dict[str, set[str]] = defaultdict(set)
    for path in frames_root.glob("**/*"):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg"}:
            continue
        flat[path.parent.name].add(path.name)
        # In the packed layout the parent of the file's directory is the game.
        for ancestor in path.parts[:-1]:
            nested[ancestor].add(path.name)
    missing = 0
    attached = 0
    for row in all_rows:
        digest = str(row.get("frame_sha") or "")
        if not digest:
            continue
        attached += 1
        game = str(row.get("game"))
        candidates = {f"{digest}.png", f"{digest}.jpg"}
        if not (candidates & flat.get(game, set())) and not (
            candidates & nested.get(game, set())
        ):
            missing += 1
    report.count("rows_with_frame", attached)
    report.count("frames_missing", missing)
    if missing:
        report.block("frames", f"{missing} rows reference a frame that is not on disk")
    report.count("frames_on_disk", sum(len(names) for names in flat.values()))


def audit(root: Path, frames_root: Path) -> Report:
    report = Report()
    runs = load_runs(root)
    all_rows = [row for rows in runs.values() for row in rows]
    report.count("runs", len(runs))
    report.count("rows", len(all_rows))
    report.count("labelled_rows", sum(1 for row in all_rows if row.get("label")))
    report.count("games", len({row.get("game") for row in all_rows}))
    report.count("by_role", dict(Counter(row.get("role") for row in all_rows)))
    report.count("by_error", dict(Counter(row.get("error") or "ok" for row in all_rows)))
    report.count("proven", dict(Counter(row.get("proven") for row in all_rows)))

    authority = Counter()
    for (game, run_id), rows in runs.items():
        tag = f"{game}/{run_id}"
        previous = PlanDirective()
        for row in rows:
            label = f"{tag}/s{row.get('step')}/{row.get('role')}"
            check_structure(row, label, report)
            check_separation(row, label, report)
            check_limits(row, label, report)
            payload = check_payload(row, label, report)
            if payload is None:
                continue
            if row["role"] == "executor":
                check_executor(row, label, payload, report)
            else:
                previous = check_planner(row, label, payload, previous, report)
                check_authority(row, label, payload, report, authority)
        check_continuity(rows, tag, report)
        check_plan_age(rows, tag, report)

    check_duplicates(all_rows, report)
    check_supervision(all_rows, report)
    check_frames(all_rows, frames_root, report)
    report.count("planner_rows_naming_an_action_by_game", dict(authority))
    report.count("planner_rows_naming_an_action", sum(authority.values()))
    return report


def print_report(report: Report) -> None:
    blocker_kinds = Counter(check for check, _ in report.blockers)
    warning_kinds = Counter(check for check, _ in report.warnings)
    print("== 规模 ==")
    for key, value in report.stats.items():
        print(f"  {key}: {value}")
    print(f"  world_step_minus_row_step: {dict(report.offsets)}")
    print()
    print("== 阻断项 ==")
    if not report.blockers:
        print("  无")
    else:
        for kind, count in blocker_kinds.items():
            print(f"  [{kind}] {count} 处")
            shown = 0
            for check, detail in report.blockers:
                if check == kind and shown < 4:
                    print(f"      {detail}")
                    shown += 1
    print()
    print("== 警告 ==")
    if not report.warnings:
        print("  无")
    else:
        for kind, count in warning_kinds.items():
            print(f"  [{kind}] {count} 处")
            shown = 0
            for check, detail in report.warnings:
                if check == kind and shown < 3:
                    print(f"      {detail}")
                    shown += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "runs" / "sft"))
    parser.add_argument("--frames", default=str(ROOT / "dist" / "frames"))
    args = parser.parse_args()
    report = audit(Path(args.root), Path(args.frames))
    print_report(report)
    return 1 if report.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
