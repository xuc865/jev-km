"""Probe the chat endpoint before a batch run.

Three questions, in the order they can waste the most time if unanswered:

1. Does the token work at all? A 401 here means every game would fail after a
   browser launch, which is minutes of setup per game.
2. Does the executor model accept an image? The executor decides from a frame; if
   it cannot see one, every label in the dataset is a guess about a screen the
   model never saw.
3. Does the model answer with the contract JSON shape? A model that replies in
   prose passes transport and fails validation on every step.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_agent.client import ModelClient, _extract_text  # noqa: E402
from general_agent.contract import encode  # noqa: E402
from general_agent.prompts import executor_system  # noqa: E402


def load_env() -> None:
    for candidate in (
        Path(__file__).resolve().parents[1] / ".env",
        Path(r"D:\kagurawang\jevgamer\.env"),
    ):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5"
    "AAAADUlEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="
)


def main() -> int:
    load_env()
    token = os.environ.get("JEVGAMER_TOKEN", "")
    print(f"token: {'set' if token else 'MISSING'} ({len(token)} chars)")
    client = ModelClient(token=token, attempts=1, timeout=60.0)

    payload = {
        "world": {"game": "probe", "goal": "reach the exit", "level": 1, "step": 1,
                  "phase": "play", "image_attached": True,
                  "legal_actions": [{"id": "right", "label": "go right", "kind": "hold",
                                     "role": "move", "seconds": 0.2}],
                  "state": {"note": "probe"}, "progress": {}},
        "plan": {"stage_goal": "probe", "tactics": "probe"},
        "output_schema": '{"action":"right"}',
    }
    system = executor_system("probe player", "reach the exit")
    print(f"endpoint: {client.url}")

    # Text-only first: separates "auth/model broken" from "vision unsupported".
    started = time.monotonic()
    call = client.complete("executor", "deepseek-v4.1-flash", system, payload)
    print(f"text-only: error={call.error!r} status={call.status} "
          f"{int((time.monotonic() - started) * 1000)}ms model={call.model}")
    print(f"  reply: {call.text[:300]!r}")

    # Then the same question with the frame attached as a data URL part.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": encode(payload)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + TINY_PNG}},
        ]},
    ]
    raw = client._send(client._payload("deepseek-v4.1-flash", messages, 0.0))
    print(f"vision: error={raw.get('_error')!r} status={raw.get('_status')}")
    text, from_reasoning = _extract_text(raw)
    flag = " (reasoning draft, no content)" if from_reasoning else ""
    print(f"  reply: {text[:300]!r}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
