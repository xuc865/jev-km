"""GameAdapter for the live panel's same-origin, visible DOM bridge.

The panel owns browser interaction; this adapter only consumes visible text,
enumerated controls, and command acknowledgements. Game rules stay in packs.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from ..types import Action, Observation
from webenv.dom import terminal as dom_terminal, phase as dom_phase

ROOT = Path(__file__).resolve().parents[2]


class PanelDomAdapter:
    name = "HTML5 DOM"

    def __init__(self, game_id: str, goal: str = "", config: dict | None = None, entry: dict | None = None):
        self.game_id = game_id
        self.entry = entry or {}
        self.config = config if config is not None else self.load_config(game_id)
        self.goal = goal or self.config.get("goal", "")
        self._condition = threading.Condition()
        self._frame: dict = {}
        self._received = 0.0
        self._changed = 0.0
        self._closed = False
        self.pending_click: dict | None = None
        self.last_execution: dict = {}
        self.executions: list[dict] = []

    @staticmethod
    def load_config(game_id: str) -> dict:
        path = ROOT / "knowledge" / game_id / "panel.json"
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}

    def reset(self) -> None:
        # The live panel loads a new iframe URL for each run.
        with self._condition:
            self._closed = False

    def update_frame(self, payload: dict) -> None:
        with self._condition:
            if payload.get("signature") != self._frame.get("signature"):
                self._changed = time.monotonic()
            self._frame = {
                "signature": str(payload.get("signature") or ""),
                "text": str(payload.get("text") or "")[:22000],
                "headings": list(payload.get("headings") or [])[:40],
                "candidates": list(payload.get("candidates") or [])[:31],
                "url": str(payload.get("url") or ""),
            }
            self._received = time.monotonic()
            self._condition.notify_all()

    def acknowledge(self, payload: dict) -> bool:
        with self._condition:
            if not self.pending_click or payload.get("command_id") != self.pending_click["command_id"]:
                return False
            self.last_execution = {
                "command_id": payload["command_id"],
                "action": self.pending_click["id"],
                "ok": payload.get("ok") is True,
                "error": str(payload.get("error") or ""),
                "after_text": str(payload.get("after_text") or "")[:22000],
            }
            self.executions.append(dict(self.last_execution))
            self.pending_click = None
            self._condition.notify_all()
            return True

    def _terminal(self, headings: list) -> tuple[bool, bool]:
        return dom_terminal(headings, self.config)

    def observe(self) -> Observation:
        deadline = time.monotonic() + 20
        with self._condition:
            while not self._closed:
                frame = dict(self._frame)
                done, success = self._terminal(frame.get("headings") or [])
                fresh = self._received and time.monotonic() - self._received < 4
                quiet = time.monotonic() - self._changed >= float(self.config.get("observation_quiet_seconds", 0))
                if fresh and (done or (quiet and frame.get("candidates"))):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("panel DOM feed unavailable: keep the live panel open")
                self._condition.wait(min(0.25, remaining))
            else:
                return Observation(actions=[], done=True, success=False)
        actions = [
            Action(
                str(item["id"]), str(item.get("label") or item["id"]),
                kind="scroll" if item.get("kind") == "scroll" else "dom_click", seconds=0,
                params={"selector": item["selector"], "signature": frame["signature"],
                        "direction": item.get("direction", "down")},
            )
            for item in frame.get("candidates") or []
            if item.get("id") and item.get("selector")
        ]
        if not done:
            actions.append(Action("wait", "等待页面动画或敌方行动结束，再重新观察", kind="wait", seconds=0.6))
        phase = dom_phase(frame, self.config)
        return Observation(
            actions=actions, done=done, success=success,
            state={
                "game": self.game_id,
                "phase": phase,
                "instructions": self.config.get("instructions", ""),
                "screen": frame.get("text", ""),
                "headings": frame.get("headings", []),
                "last_execution": {k: v for k, v in self.last_execution.items() if k != "after_text"},
            },
            raw=frame,
        )

    def act(self, action: Action) -> None:
        if action.kind == "wait":
            time.sleep(action.seconds)
            self.last_execution = {"action": action.id, "ok": True}
            self.executions.append(dict(self.last_execution))
            return
        with self._condition:
            if self._closed:
                return
            if action.params.get("signature") != self._frame.get("signature"):
                self.last_execution = {"action": action.id, "ok": False, "error": "stale observation; reobserve"}
                self.executions.append(dict(self.last_execution))
                return
            self.pending_click = {
                "command_id": uuid.uuid4().hex,
                "id": action.id,
                "kind": action.kind,
                "direction": action.params.get("direction"),
                "selector": action.params["selector"],
                "signature": action.params["signature"],
            }
            self._condition.notify_all()
            deadline = time.monotonic() + 8
            while self.pending_click and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.pending_click = None
                    raise TimeoutError("panel did not acknowledge the DOM action")
                self._condition.wait(min(0.25, remaining))
        # Let transitions settle and let the panel publish the resulting state.
        time.sleep(float(self.config.get("action_settle_seconds", 0.7)))

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self.pending_click = None
            self._condition.notify_all()
