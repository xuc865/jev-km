"""Structured per-frame action feed exposed by an instrumented game.

The game publishes the actions it can accept right now plus a small local
state snapshot (cell, level, whether the avatar is settled on a cell). When
that feed is available the inner loop can choose an action without a vision
call; when it is missing the existing screenshot path stays the fallback.

Protocol (see html-games/.../astray/serve.py):

    GET /__agent/frame
    {
      "ok": true,
      "age": 0.031,
      "actions": [
        {"id": "move_up", "keys": ["UP"], "available": true,
         "clearance": 3, "why": "open corridor 3"},
        {"id": "move_left", "keys": ["LEFT"], "available": false,
         "clearance": 0, "why": "brick wall"},
        ...
      ],
      "state": {"cell": [1, 1], "settled": true, "level": 1,
                "dimension": 11, "won": false}
    }
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class FrameClient:
    """Poll a game's structured frame feed. Disable after repeated failures."""

    def __init__(self, url: str, timeout: float = 1.0, max_failures: int = 2) -> None:
        self.url = str(url or "").strip()
        self.timeout = timeout
        self.max_failures = max_failures
        self.failures = 0

    @property
    def disabled(self) -> bool:
        return not self.url or self.failures >= self.max_failures

    def fetch(self) -> dict | None:
        if self.disabled:
            return None
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError):
            self.failures += 1
            return None
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self.failures += 1
            return None
        if not isinstance(data, dict) or data.get("ok") is False:
            self.failures += 1
            return None
        if not isinstance(data.get("actions"), list) or not data["actions"]:
            self.failures += 1
            return None
        age = data.get("age")
        if isinstance(age, (int, float)) and age > 3.0:
            # The page stopped publishing (closed, backgrounded, or crashed).
            self.failures += 1
            return None
        self.failures = 0
        return data
