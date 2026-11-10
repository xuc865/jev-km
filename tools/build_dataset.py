"""Summarise a collected SFT batch and emit the files training needs.

The collector writes per-game folders for crash isolation. That is right for
collection and wrong for training: a trainer wants one file, a stable row order,
and a manifest that says which games contributed. This script builds those
without touching the source records, so a batch can be re-summarised at any time
and the numbers always agree with the raw rows.

Three outputs, all under ``--out``:

* ``sft.jsonl``     one row per model call, whole trajectories, nothing dropped;
* ``sft_clean.jsonl`` the same rows restricted to validated labels, which is what
                    a first training run should use;
* ``manifest.json`` counts by role, game, trigger and error reason, plus the
                    distribution of trajectory lengths.

Rows keep a pointer to their frame file rather than the image bytes, so the file
stays readable and the images remain deduplicated on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_rows(root: Path) -> list[dict]:
    """Every recorded call, ordered by game then by write time.

    The order is fixed on purpose: a shuffled dataset makes two runs of the same
    batch non-comparable, and a trainer that reports "row 812" cannot be checked.
    """
    rows: list[dict] = []
    for path in sorted(root.glob("*/*/roles.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    rows.sort(key=lambda row: (str(row.get("game") or ""), float(row.get("at") or 0)))
    return rows


def attach_frames(rows: list[dict], root: Path) -> int:
    """Rewrite ``frame_sha`` into a run-relative path when the file exists.

    The hash alone is not enough for a trainer: it says which frame was attached
    but not where it is. Resolving it here keeps the collector free of dataset
    layout concerns and makes a missing frame visible as a count instead of a
    silent blank image.
    """
    missing = 0
    for row in rows:
        digest = str(row.get("frame_sha") or "")
        if not digest:
            row["frame_path"] = ""
            continue
        folder = root / str(row["game"]) / str(row["run_id"]) / "frames"
        found = ""
        for suffix in (".png", ".jpg"):
            candidate = folder / (digest + suffix)
            if candidate.is_file():
                found = str(candidate.relative_to(root)).replace("\\", "/")
                break
        if found:
            row["frame_path"] = found
        else:
            row["frame_path"] = ""
            missing += 1
    return missing


def summarise(rows: list[dict]) -> dict:
    by_role = Counter(row.get("role") for row in rows)
    errors = Counter(row.get("error") or "ok" for row in rows)
    triggers = Counter(row.get("trigger") or "none" for row in rows)
    games = Counter(row.get("game") for row in rows)
    labelled = [row for row in rows if row.get("label")]
    proven = [row for row in labelled if row.get("proven")]
    frames = {row.get("frame_sha") for row in rows if row.get("frame_sha")}
    steps = Counter(row.get("game") for row in rows if row.get("role") == "executor")
    return {
        "rows": len(rows),
        "labelled_rows": len(labelled),
        "label_rate": round(len(labelled) / len(rows), 4) if rows else 0.0,
        "proven_rows": len(proven),
        "unproven_rows": len(labelled) - len(proven),
        "by_role": dict(by_role),
        "by_error": dict(errors),
        "by_trigger": dict(triggers),
        "games": len(games),
        "unique_frames": len(frames),
        "executor_steps_by_game": dict(sorted(steps.items())),
        "rows_by_game": dict(sorted(games.items())),
        "schemas": sorted({str(row.get("schema") or "") for row in rows}),
        "models": sorted({str(row.get("model") or "") for row in rows}),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "runs" / "sft"))
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--proven-only",
        action="store_true",
        help="drop labelled rows whose step had no numeric metric behind it",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    if not rows:
        print(f"no records under {root}", file=sys.stderr)
        return 1

    missing_frames = attach_frames(rows, root)
    write_jsonl(out / "sft.jsonl", rows)

    clean = [row for row in rows if row.get("label")]
    if args.proven_only:
        clean = [row for row in clean if row.get("proven")]
    write_jsonl(out / "sft_clean.jsonl", clean)

    summary = summarise(rows)
    summary["clean_rows"] = len(clean)
    summary["missing_frames"] = missing_frames
    summary["root"] = str(root)
    (out / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"rows: {summary['rows']} -> {out / 'sft.jsonl'}")
    print(f"clean: {len(clean)} -> {out / 'sft_clean.jsonl'}")
    print(f"label rate: {summary['label_rate']:.3f}  games: {summary['games']}  "
          f"frames: {summary['unique_frames']}  missing frames: {missing_frames}")
    print(f"by role: {summary['by_role']}")
    print(f"by error: {summary['by_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
