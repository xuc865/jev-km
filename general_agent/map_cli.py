"""Run one map/ game through the general agent harness."""

from __future__ import annotations

import argparse
import json
import os

from .adapters.gzdoom import GzdoomAdapter, catalog_entry
from .adapters.html5 import HtmlGameAdapter
from .client import ModelClient
from .live import CARD_GAMES
from .outer import PlannerPolicy
from .policy import ExecutorPolicy
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
    parser.add_argument("--no-planner", action="store_true")
    parser.add_argument("--model", default=None, help="model for both tiers")
    parser.add_argument("--executor-model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--review-every", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-seconds", type=float, default=240.0)
    args = parser.parse_args()

    entry = catalog_entry(args.game)
    goal = args.goal or str(entry.get("description") or entry.get("title") or args.game)
    adapter = adapter_for(args.game, goal)
    client = ModelClient(token=os.environ.get("JEVGAMER_TOKEN") or "")
    policy = ExecutorPolicy(
        client=client,
        model=args.executor_model or args.model or "",
        confidence=args.confidence,
        review_every=args.review_every,
        max_candidates=int(getattr(adapter, "max_candidates", 32)),
    )
    planner = None if args.no_planner else PlannerPolicy(
        client=client,
        model=args.planner_model or args.model or "",
    )

    def on_step(step, obs, decision) -> None:
        print(
            f"step {step:3d} level {obs.level} "
            f"actions={[action.id for action in obs.actions]} "
            f"by={decision.by} choice={decision.action_id or '-'} "
            f"rev={policy.plan.revision} {decision.ms}ms "
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
        escalate=planner,
    )
    print(
        json.dumps(
            {
                "game": args.game,
                "goal": goal,
                "success": result.success,
                "steps": result.steps,
                "seconds": round(result.seconds, 2),
                "error": result.error,
                "plan": policy.plan.to_wire(),
                "model_calls": len(client.calls),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
