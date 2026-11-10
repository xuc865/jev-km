# -*- coding: utf-8 -*-
"""A second scoring channel, so collection is not capped by one serial queue.

Measured sustained capacity of WeMM-JEV is ~45 calls/min and flat from three to
sixteen callers: it is a serial queue, so worker count cannot raise output. The
chat channel answers text-only steps in ~1s and is already rate-limited by the
local proxy. Some games can be scored from the visible text state alone (score,
kills, distance are published in `state`), so those steps can skip the image and
use the cheaper channel while vision-required steps stay on WeMM-JEV.

This is the plumbing only: `text_only` decides whether a game can be scored
without pixels, and the proxy bucket keeps the chat channel inside its quota.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def chat_available() -> bool:
    """True when the local rate-limit proxy is listening."""
    import socket
    from urllib.parse import urlparse

    proxy = os.environ.get("JEVGAMER_CHAT_PROXY",
                          "http://127.0.0.1:8799/v1/chat/completions").strip()
    try:
        parts = urlparse(proxy)
        with socket.create_connection((parts.hostname, parts.port or 80), timeout=0.4):
            return True
    except Exception:
        return False


def text_state_summary(state: dict, limit: int = 2400) -> str:
    """Compact text view of the public state, without the screenshot.

    Only fields a policy can act on are kept: phase/level, the progress metric,
    the last execution result and any published field values. Instructions and
    long scene descriptions are dropped because they are the same every step and
    would only inflate the prompt.
    """
    if not isinstance(state, dict):
        return str(state)[:limit]
    keep = {}
    for key in ("game", "phase", "level", "screen", "headings", "public_fields",
                "progress", "last_execution", "cell", "hud"):
        value = state.get(key)
        if value in (None, "", [], {}):
            continue
        keep[key] = value
    text = json.dumps(keep, ensure_ascii=False)
    return text[:limit]
