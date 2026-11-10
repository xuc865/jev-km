"""Read the per-game progress metric that the catalogue already publishes.

``map/_agent/progress.js`` holds a hand-written spec for every catalogue game:
which JS value is the real progress signal (level, wave, kills, distance, lap,
floor...) and whether higher or lower is better. It was only ever loaded by
``direct/html_host.py``, so the sheet in WeCom got "max progress" for each game
while every SFT episode recorded none.

This module injects the same script into whatever frame an adapter drives and
reads the value back on demand. Two details matter:

* The script is injected after the page has loaded, so anything the game scored
  before injection is lost. Reads are therefore explicit per step instead of
  relying on the script's own 250ms timer, which would sample the wrong moment
  under a step clock that advances faster than wall time.
* ``__agentProgressQuiet`` suppresses the on-screen badge the script paints.
  Without it the badge becomes a DOM node inside the game page, which both
  pollutes the snapshot the model reads and shifts the page layout.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_ROOT = ROOT / "map"

INJECT = """
(game) => {
  window.__agentProgressQuiet = true;
  if (window.__agentProgressNow) { return true; }
  var parent = document.head || document.documentElement;
  if (!parent) { return false; }
  var script = document.createElement('script');
  script.src = '/map/_agent/progress.js';
  script.setAttribute('data-game', game);
  parent.appendChild(script);
  return true;
}
"""

READ = """
() => {
  try {
    if (!window.__agentProgressNow) { return null; }
    var record = window.__agentProgressNow();
    if (!record || typeof record.best !== 'number') { return null; }
    return {
      metric: record.metric || '',
      sense: record.sense || 'high',
      value: (typeof record.value === 'number') ? record.value : null,
      best: record.best,
      origin: record.origin || ''
    };
  } catch (error) {
    return null;
  }
}
"""


def _frame(host):
    frame = getattr(host, "frame", None)
    return frame if frame is not None else None


def read(host) -> dict | None:
    """Current progress record, or None when the game exposes no metric."""
    frame = _frame(host)
    if frame is None:
        return None
    try:
        value = frame.evaluate(READ)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def inject(host, game_id: str, wait_s: float = 4.0) -> bool:
    """Load the progress script into the driven frame.

    Returns True when the script is live in the page, which is a different fact
    from having a value: a game sitting on its menu legitimately exposes nothing
    yet. Reporting those separately keeps "we failed to inject" from being
    confused with "the game has not started", which need different fixes.
    """
    frame = _frame(host)
    if frame is None or not game_id:
        return False
    try:
        frame.evaluate(INJECT, str(game_id))
    except Exception:
        return False
    deadline = time.time() + max(0.0, float(wait_s))
    while time.time() < deadline:
        try:
            live = frame.evaluate("() => typeof window.__agentProgressNow === 'function'")
        except Exception:
            live = False
        if live:
            return True
        time.sleep(0.1)
    return False


def live(host) -> bool:
    """True when the progress script is still resident in the driven frame.

    A game that restarts itself (mkjs reloads the page when Space is pressed)
    replaces the whole document and takes the injected script with it. Telling
    that apart from "this frame never had a probe" is what lets the caller
    decide whether to re-inject or to accept that the game has no metric.
    """
    frame = _frame(host)
    if frame is None:
        return False
    try:
        return bool(frame.evaluate("() => typeof window.__agentProgressNow === 'function'"))
    except Exception:
        return False


def absent() -> dict:
    """Explicit 'no metric available' marker.

    Distinguishing "this game has no progress signal" from "we forgot to record
    it" is the whole point: an episode tagged `metric: ''` can be filtered out
    knowingly, whereas a missing field silently looks like progress never
    happened.
    """
    return {"metric": "", "sense": "high", "value": None, "best": None,
            "source": "none", "note": "catalogue has no progress spec for this game"}


_SPEC_CACHE: dict[str, dict] = {}


def spec(game_id: str) -> dict:
    """Statically read this game's declared metric from progress.js.

    Without this the first samples of every episode carry an empty metric name,
    because the game has not started yet and therefore exposes no value. The name
    is knowable in advance even when the value is not, and downstream filtering
    needs the name to decide what kind of counter it is looking at.
    """
    if not game_id:
        return {}
    cached = _SPEC_CACHE.get(game_id)
    if cached is not None:
        return cached
    try:
        text = (MAP_ROOT / "_agent" / "progress.js").read_text(encoding="utf-8", errors="ignore")
    except Exception:
        _SPEC_CACHE[game_id] = {}
        return {}
    body = text.split("var SPECS = {", 1)[-1].split("\n  };", 1)[0]
    for line in body.splitlines():
        match = re.match(r'\s*"([^"]+)"\s*:\s*\{(.*)\},?\s*$', line)
        if not match or match.group(1) != game_id:
            continue
        rest = match.group(2)
        found = re.search(r'metric:\s*"([^"]+)"', rest)
        entry = {"metric": found.group(1) if found else ""}
        sense = re.search(r'sense:\s*"([^"]+)"', rest)
        if sense:
            entry["sense"] = sense.group(1)
        _SPEC_CACHE[game_id] = entry
        return entry
    _SPEC_CACHE[game_id] = {}
    return {}


def pending(game_id: str, best: float | None = None) -> dict:
    """The game has a declared metric but has not exposed a value yet.

    This is the normal state on a menu screen and must not be reported as a
    missing metric: the distinction is what keeps a startup frame from looking
    like a game with no progress support at all.
    """
    known = spec(game_id)
    if not known.get("metric"):
        return absent()
    return {"metric": known["metric"], "sense": known.get("sense") or "high",
            "value": None, "best": best, "source": "game-pending",
            "note": "game has not exposed the metric value yet"}
