"""Run one general JEV episode against the live Astray console.

Example:
    python -m general_agent.cli --goal "通关第2关" --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import os

from .adapters.astray import DEFAULT_IMAGE_URL, AstrayAdapter
from .outer import OuterEscalation
from .policy import JevPolicy
from .runner import run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default="通关第2关")
    parser.add_argument("--game", default="Astray")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--no-outer", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--qpm", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--hint", default="")
    args = parser.parse_args()

    adapter = AstrayAdapter(
        game_id=args.game,
        goal=args.goal,
        image_url=args.image_url,
        include_image=not args.no_image,
    )
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
        args.goal,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        on_step=on_step,
        hint=args.hint,
        escalate=outer,
    )
    if outer is not None and outer.hint:
        print(f"outer hint: {outer.hint}")
    print(json.dumps(result.__dict__, ensure_ascii=False, default=str, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
