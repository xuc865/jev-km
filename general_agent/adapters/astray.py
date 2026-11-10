"""Astray adapter: the only Astray-specific code in the general harness."""

from __future__ import annotations

import base64
import time
import urllib.request

import harness
import knowledge
from feed import FrameClient

from ..types import Action, Observation

DEFAULT_IMAGE_URL = "http://127.0.0.1:8766/__agent/image.jpg"


class AstrayAdapter:
    name = "Astray"

    def __init__(
        self,
        game_id: str = "Astray",
        goal: str = "",
        image_url: str = DEFAULT_IMAGE_URL,
        include_image: bool = True,
    ) -> None:
        self.game = knowledge.load(game_id)
        self.goal = goal
        self.target = int(self.game.target_level(goal) or 1)
        self.feed = (
            FrameClient(self.game.feed_url(), max_failures=10**9)
            if self.game.feed_url()
            else None
        )
        self.image_url = image_url
        self.include_image = include_image
        self.frame: dict = {}

    def reset(self) -> None:
        self.frame = {}

    def observe(self) -> Observation:
        frame = self.feed.fetch() if self.feed is not None else None
        if not isinstance(frame, dict):
            frame = {}
        if self.feed is not None:
            deadline = time.time() + 1.5
            while time.time() < deadline:
                current = frame.get("state") if isinstance(frame.get("state"), dict) else {}
                if current.get("settled"):
                    break
                time.sleep(0.05)
                nxt = self.feed.fetch()
                if isinstance(nxt, dict):
                    frame = nxt
        self.frame = frame
        state = frame.get("state") if isinstance(frame.get("state"), dict) else {}
        try:
            from policy import RulePolicy

            if not hasattr(self, "explorer"):
                self.explorer = RulePolicy()
                self.explorer.target_level = 99
            suggestion = self.explorer.choose(frame, None, "")
            state["hint"] = (
                f"BFS explorer suggests {suggestion.get('choice')} "
                f"({suggestion.get('why')})"
            )
            state["visited"] = len(self.explorer.visited)
            if getattr(self.explorer, "plan", None):
                state["frontier"] = str(self.explorer.plan[-1].get("next"))
        except Exception:
            pass
        actions: list[Action] = []
        for item in frame.get("actions") or []:
            if not isinstance(item, dict) or not item.get("available"):
                continue
            action_id = str(item.get("id") or "")
            if not action_id:
                continue
            spec = self.game.action(action_id) or {}
            payload = spec.get("input") or {}
            kind = str(payload.get("type") or "hold")
            keys = tuple(str(key) for key in payload.get("keys") or ())
            seconds = (
                self.game.feed_hold()
                if kind == "hold"
                else float(payload.get("seconds") or 0.1)
            )
            actions.append(
                Action(
                    id=action_id,
                    label=str(item.get("label") or spec.get("label") or action_id),
                    kind=kind,
                    keys=keys,
                    seconds=seconds,
                    params={
                        "clearance": item.get("clearance"),
                        "input": payload,
                    },
                )
            )
        level = int(state.get("level") or 1)
        # Astray keeps generating levels; `won` is only a level-clear fade.
        # The task's target level decides success, not a fake Level-2 cap.
        done = self.target > 1 and level >= self.target
        return Observation(
            actions=actions,
            image=self._image() if self.include_image else "",
            state=state,
            level=level,
            done=done,
            success=done,
            raw=frame,
        )

    def act(self, action: Action) -> None:
        harness.act_feed(
            action.id,
            self.game,
            action.seconds,
            frame=self.frame,
        )

    def _image(self) -> str:
        try:
            with urllib.request.urlopen(self.image_url, timeout=1.5) as response:
                data = response.read()
        except Exception:
            return ""
        if not data:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

    def close(self) -> None:
        return None
