"""Batch SFT collector: run every known-good game once and keep the whole trace.

Design notes that matter for the dataset, not just for convenience:

* Both roles share one ``ModelClient``, so every exchange lands in one place and
  the recorder never has to reconstruct a prompt.
* Frames are stored once per unique hash under ``frames/``. A 200-step episode
  repeats the same picture for many steps; inlining base64 per row would make the
  image bytes (not the labels) dominate the dataset.
* Each game gets its own folder and its own ``roles.jsonl``, so a partial batch is
  still usable: a crash at game 30 does not corrupt games 1-29.
* Episodes continue past a single failed step. The only fatal conditions are a
  browser that will not start and a planner that stays down, which are reported
  per game rather than aborting the batch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from general_agent.adapters.generic_html import GenericHtmlAdapter  # noqa: E402
from general_agent.adapters.gzdoom import GzdoomAdapter  # noqa: E402
from general_agent.client import ModelClient  # noqa: E402
from general_agent.outer import PlannerPolicy  # noqa: E402
from general_agent.policy import ExecutorPolicy  # noqa: E402
from general_agent.runner import run_episode  # noqa: E402
from general_agent.sft import RoleRecorder, save_frames  # noqa: E402


def load_env() -> None:
    for candidate in (ROOT / ".env", Path(r"D:\kagurawang\jevgamer\.env")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def runnable_games(index: Path, root: Path) -> list[dict]:
    """The curated list of games that were actually driven to a score."""
    rows = list(csv.DictReader(index.open(encoding="utf-8-sig")))
    games = []
    for row in rows:
        entry = str(row.get("包内入口") or "").strip()
        if not entry.lower().endswith("index.html"):
            continue
        target = root / entry.replace("/", os.sep)
        if not target.is_file():
            continue
        games.append(
            {
                "id": str(row.get("游戏标识") or "").strip(),
                "entry": entry,
                "category": str(row.get("分类目录") or "").strip(),
                "description": str(row.get("运行载体/引擎") or "").strip(),
            }
        )
    return games


def build_adapter(game: dict, goal: str, artifacts: Path) -> GenericHtmlAdapter:
    """Adapter for one curated game.

    The curated pack and the catalogue use the same ids and the same relative
    entry paths, so the catalogue supplies the metadata and the environment
    supplies where the bytes live. That split is why ``JEVGAMER_MAP_ROOT`` points
    at the pack root rather than copying 240 MB into this repository.
    """
    from general_agent.adapters.generic_html import MAP_ROOT

    entry_path = str(game["entry"]).replace("h5/", "", 1)
    adapter = GenericHtmlAdapter(
        game_id=game["id"],
        goal=goal,
        entry={"id": game["id"], "entry_path": entry_path},
    )
    adapter.artifacts = artifacts
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=r"D:\kagurawang\jevgamer\dist\runnable_games")
    parser.add_argument("--out", default=str(ROOT / "runs" / "sft"))
    parser.add_argument("--games", default="", help="comma separated ids; default all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--review-every", type=int, default=6)
    parser.add_argument("--seed", default="")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="games to run at once; each gets its own browser and its own folder",
    )
    parser.add_argument("--resume", action="store_true", help="skip games already done")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved game list and budgets, then exit without launching a browser",
    )
    args = parser.parse_args()

    load_env()
    if args.seed:
        # Set once, before any worker starts. The previous per-game assignment
        # raced across threads and could give a game the seed meant for another.
        os.environ["JEVGAMER_SEED"] = args.seed
    pack = Path(args.pack)
    games = runnable_games(pack / "INDEX.csv", pack)
    if args.games:
        wanted = {item.strip() for item in args.games.split(",") if item.strip()}
        games = [game for game in games if game["id"] in wanted]
    if args.limit:
        games = games[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    done: set[str] = set()
    if args.resume and index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("ok"):
                done.add(str(row.get("game")))

    print(f"games: {len(games)} -> {out}", flush=True)
    if args.dry_run:
        # Printed before any browser or model call, so a batch can be checked
        # against the pack without spending tokens. This is the check that was
        # missing when the map root was unset and every game failed identically.
        print(f"pack: {pack}", flush=True)
        print(f"map root: {os.environ.get('JEVGAMER_MAP_ROOT', '(unset)')}", flush=True)
        print(
            f"budget: executor={os.environ.get('JEVGAMER_EXECUTOR_MAX_TOKENS', 'default')} "
            f"planner={os.environ.get('JEVGAMER_PLANNER_MAX_TOKENS', 'default')}",
            flush=True,
        )
        for position, game in enumerate(games, 1):
            mark = "done" if game["id"] in done else "todo"
            print(f"[{position}/{len(games)}] {game['id']:<24} {mark:<4} {game['entry']}",
                  flush=True)
        return 0
    pending = [game for game in games if game["id"] not in done]
    skipped = len(games) - len(pending)
    if skipped:
        print(f"resume: {skipped} already done, {len(pending)} to run", flush=True)

    write_lock = threading.Lock()

    def finish(game: dict, row: dict) -> None:
        row["seconds"] = row.get("seconds", 0)
        with write_lock:
            with index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"    [{game['id']}] ok={row.get('ok')} steps={row.get('steps')} "
            f"records={row.get('records')} calls={row.get('calls')} "
            f"frames={row.get('frames')} progress={row.get('progress')} "
            f"err={str(row.get('error'))[:110]}",
            flush=True,
        )

    def work(index: int, game: dict) -> None:
        print(f"[{index}/{len(pending)}] {game['id']} ...", flush=True)
        started = time.time()
        row = collect_one(game, args, out)
        row.setdefault("seconds", 0)
        row["seconds"] = round(time.time() - started, 1)
        finish(game, row)

    workers = max(1, min(int(args.workers or 1), len(pending) or 1))
    if workers == 1:
        for index, game in enumerate(pending, 1):
            work(index, game)
        return 0

    # Concurrent games, one browser each. Kept per-game rather than per-step
    # because the bottleneck is model latency (a planner review takes ~13s), not
    # the browser, and because a shared page would make two games contaminate each
    # other's DOM state.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(work, index, game): game
            for index, game in enumerate(pending, 1)
        }
        for future in concurrent.futures.as_completed(futures):
            game = futures[future]
            try:
                future.result()
            except Exception as exc:
                # A worker that dies must still leave a row, or `--resume` would
                # retry it forever without anyone knowing why.
                finish(game, {"game": game["id"], "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"})
    return 0


def collect_one(game: dict, args, out: Path) -> dict:
    run_id = f"{game['id']}-{int(time.time())}"
    folder = out / game["id"] / run_id
    folder.mkdir(parents=True, exist_ok=True)
    goal = f"play {game['id']} and make the game metric advance"
    adapter = None
    try:
        adapter = build_adapter(game, goal, folder / "browser")
        client = ModelClient(
            token=os.environ.get("JEVGAMER_TOKEN", ""),
            # No client-wide ceiling: each role is budgeted by its own policy, and a
            # single shared cap is what truncated every planner review at 512
            # tokens. The env vars below exist so a batch can be retuned without a
            # code change.
            max_tokens=None,
            thinking=False,
            attempts=4,
        )
        policy = ExecutorPolicy(
            client=client,
            max_candidates=int(getattr(adapter, "max_candidates", 32)),
            review_every=args.review_every,
        )
        planner = PlannerPolicy(client=client)
        if hasattr(adapter, "apply_subset"):
            policy.subset_sink = lambda exclude, note="": adapter.apply_subset(
                exclude, note=note, source="planner"
            )
        recorder = RoleRecorder(game["id"], run_id, goal=goal, root=out)
        trace = (folder / "trace.jsonl").open("a", encoding="utf-8")
        seen_frames: set[str] = set()

        def progress() -> dict:
            refresh = getattr(adapter, "refresh_progress", None)
            if callable(refresh):
                try:
                    return refresh() or {}
                except Exception:
                    pass
            return dict(getattr(adapter, "progress", {}) or {})

        def on_step(step, obs, decision) -> None:
            recorder.step = step
            recorder.plan_revision = policy.plan.revision
            recorder.trigger = planner.last_trigger
            recorder.set_progress(progress())
            # Store the frame before draining, so a record can never reference a
            # hash that has not been written yet.
            frame_sha = ""
            if obs.image:
                frame_sha = save_frames(recorder.frame_dir, obs.image, seen_frames)
            recorder.drain(client)
            # The observation is written with the frame *reference* instead of the
            # base64 body. Inlining it put 340 KB on every line and made a 13-step
            # trace 4 MB; the picture itself is already on disk under frames/, so
            # the hash is the whole link and the trace stays readable as text.
            shot = asdict(obs)
            shot["image"] = ""
            shot["image_sha"] = frame_sha
            trace.write(json.dumps({
                "step": step,
                "observation": shot,
                "decision": asdict(decision),
                "plan": policy.plan.to_wire(),
                "executor_step": policy.last_step.to_wire() if policy.last_step else None,
                "progress": progress(),
                "last_execution": getattr(adapter, "last_execution", {}),
            }, ensure_ascii=False) + "\n")
            trace.flush()

        result = run_episode(
            adapter, policy, goal,
            max_steps=args.steps, max_seconds=args.seconds,
            on_step=on_step, escalate=planner,
        )
        trace.close()
        meta = {
            "game": game["id"],
            "category": game["category"],
            "goal": goal,
            "seed": os.environ.get("JEVGAMER_SEED", "42"),
            "executor_model": policy.model,
            "planner_model": planner.model,
            "result": asdict(result),
            "final_plan": planner.plan.to_wire(),
            "planner_revisions": planner.revisions,
            "planner_error": planner.error,
            "progress": progress(),
            "sft": recorder.summary(),
            "frames": len(seen_frames),
        }
        (folder / "summary.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "game": game["id"],
            "ok": bool(recorder.counts["executor"] or recorder.counts["planner"]),
            "steps": result.steps,
            "records": recorder.counts["executor"] + recorder.counts["planner"],
            "calls": len(client.calls),
            "frames": len(seen_frames),
            "progress": progress().get("value"),
            "success": result.success,
            "error": result.error,
            "folder": str(folder),
        }
    except Exception as exc:
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"game": game["id"], "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "folder": str(folder)}
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
