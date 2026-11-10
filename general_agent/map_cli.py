"""Run one map/ game through the general JEV harness."""

from __future__ import annotations

import argparse
import json
import os

from .adapters.gzdoom import GzdoomAdapter, catalog_entry
from .adapters.html5 import HtmlGameAdapter
from .live import CARD_GAMES
from .outer import OuterEscalation
from .policy import JevPolicy
from .runner import run_episode


def adapter_for(game_id: str, goal: str):
    entry = catalog_entry(game_id)
    entry_path = str(entry.get("entry_path") or "").lower()
    if entry_path.endswith("run.bat"):
        return GzdoomAdapter(game_id=game_id, goal=goal)
    return HtmlGameAdapter(
        game_id=game_id,
        goal=goal,
        mode="clicks" if game_id in CARD_GAMES else "keys",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, help="catalog id, e.g. basic or t-rex-runner")
    parser.add_argument("--goal", default="")
    parser.add_argument("--no-outer", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--qpm", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-seconds", type=float, default=240.0)
    parser.add_argument("--hint", default="")
    args = parser.parse_args()

    entry = catalog_entry(args.game)
    goal = args.goal or str(entry.get("description") or entry.get("title") or args.game)
    adapter = adapter_for(args.game, goal)
    policy = JevPolicy(
        token=os.environ.get("JEVGAMER_TOKEN") or "",
        confidence=args.confidence,
        qpm=args.qpm,
    )
    outer = None if args.no_outer else OuterEscalation(
        token=os.environ.get("JEVGAMER_TOKEN") or "",
        model=args.model,
    )

    def on_step(step, obs, decision) -> None:
        print(
            f"step {step:3d} level {obs.level} "
            f"actions={[action.id for action in obs.actions]} "
            f"by={decision.by} choice={decision.action_id or '-'} "
            f"image={decision.image_count} {decision.ms}ms "
            f"{decision.note}",
            flush=True,
        )

    result = run_episode(
        adapter,
        policy,
        goal,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        on_step=on_step,
        hint=args.hint,
        escalate=outer,
    )
    if outer is not None and outer.hint:
        print(f"outer hint: {outer.hint}")
    print(
        json.dumps(
            {
                "game": args.game,
                "goal": goal,
                "success": result.success,
                "steps": result.steps,
                "seconds": round(result.seconds, 2),
                "error": result.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
