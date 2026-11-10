"""Run one general-agent episode against the live Astray console.

Example:
    python -m general_agent.cli --goal "通关第2关" --max-steps 200

Both tiers use the same client. ``--model`` sets both; ``--executor-model`` and
``--planner-model`` override them individually, which is how a run can compare a
cheap executor against a stronger planner without touching code.
"""

from __future__ import annotations

import argparse
import json
import os

from .adapters.astray import DEFAULT_IMAGE_URL, AstrayAdapter
from .client import ModelClient
from .outer import PlannerPolicy
from .policy import ExecutorPolicy
from .runner import run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default="通关第2关")
    parser.add_argument("--game", default="Astray")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--no-planner", action="store_true")
    parser.add_argument("--model", default=None, help="model for both tiers")
    parser.add_argument("--executor-model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--review-every", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    args = parser.parse_args()

    adapter = AstrayAdapter(
        game_id=args.game,
        goal=args.goal,
        image_url=args.image_url,
        include_image=not args.no_image,
    )
    client = ModelClient(token=os.environ.get("JEVGAMER_TOKEN") or "")
    policy = ExecutorPolicy(
        client=client,
        model=args.executor_model or args.model or "",
        confidence=args.confidence,
        review_every=args.review_every,
    )
    planner = None if args.no_planner else PlannerPolicy(
        client=client,
        model=args.planner_model or args.model or "",
    )

    def on_step(step, obs, decision) -> None:
        plan = policy.plan
        print(
            f"step {step:3d} level {obs.level} "
            f"actions={[action.id for action in obs.actions]} "
            f"by={decision.by} choice={decision.action_id or '-'} "
            f"rev={plan.revision} {decision.ms}ms "
            f"{decision.note}",
            flush=True,
        )

    result = run_episode(
        adapter,
        policy,
        args.goal,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        on_step=on_step,
        escalate=planner,
    )
    final = {
        "result": result.__dict__,
        "plan": policy.plan.to_wire(),
        "model_calls": [call.to_wire() for call in client.calls],
    }
    print(json.dumps(final, ensure_ascii=False, default=str, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
