"""Test doubles for the two-role design; no network and no sleeps.

The previous tests substituted a ``score_fn`` for the fast tier and a ``chat_fn``
for the slow tier, because the two tiers spoke to different endpoints. Now that
both roles call ``ModelClient.complete``, the natural seam is the client itself:
a fake client returns canned replies in order and records what each role was
asked, which is strictly more information than the old wrappers kept.
"""

from __future__ import annotations

import json

from general_agent.client import ModelCall
from general_agent.contract import SCHEMA_VERSION


class ScriptedClient:
    """A ModelClient stand-in that answers from a per-role script.

    Replies are keyed by role so a test can fix the executor's behaviour and the
    planner's independently. Once a role's script is exhausted the last entry
    repeats, which keeps tests about cadence from having to enumerate every call.
    """

    def __init__(self, executor=None, planner=None, models=None) -> None:
        self.scripts = {
            "executor": list(executor or []),
            "planner": list(planner or []),
        }
        self.models = models or {}
        self.calls: list[ModelCall] = []
        self._pending: list[ModelCall] = []
        self.last_error = ""
        self.asked: list[dict] = []

    def complete(self, role, model, system, user_payload, *, temperature=0.0, image="",
                 max_tokens=None) -> ModelCall:
        script = self.scripts.get(role) or [""]
        index = sum(1 for call in self.calls if call.role == role)
        text = script[min(index, len(script) - 1)]
        self.asked.append(
            {"role": role, "system": system, "payload": user_payload, "model": model,
             "image": image, "max_tokens": max_tokens}
        )
        error = ""
        if isinstance(text, Exception):
            error = str(text)
            text = ""
        call = ModelCall(
            role=role,
            model=self.models.get(role, model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            text=text,
            ms=1,
            usage={"input_tokens": 10, "output_tokens": 5},
            error=error,
            status=0 if not error else 500,
            schema=SCHEMA_VERSION,
            frame_sha="f" * 16 if image else "",
        )
        self.calls.append(call)
        self._pending.append(call)
        if error:
            self.last_error = error
        return call

    def drain_calls(self) -> list[ModelCall]:
        pending = self._pending
        self._pending = []
        return pending

    def payloads(self, role: str) -> list[dict]:
        return [item["payload"] for item in self.asked if item["role"] == role]

    def systems(self, role: str) -> list[str]:
        return [item["system"] for item in self.asked if item["role"] == role]


def executor_reply(action, why="because", confidence=0.9, expect="something", belief="ok", stuck=False) -> str:
    return json.dumps(
        {
            "action": action,
            "why": why,
            "confidence": confidence,
            "expect": expect,
            "belief": belief,
            "stuck": stuck,
        },
        ensure_ascii=False,
    )


def planner_reply(
    stage_goal="reach the exit",
    tactics="walk right",
    briefing="",
    situation="",
    avoid=None,
    exclude=None,
    review_in=8,
    verdict="continue",
    note="",
) -> str:
    """Build a planner reply; pass ``None`` for a field to omit it entirely.

    Omitting is the interesting case for the contract, so it has to be
    expressible: a helper that always emitted every key would make a partial
    revision impossible to test.
    """
    body = {
        "review_in": review_in,
        "verdict": verdict,
    }
    if stage_goal is not None:
        body["stage_goal"] = stage_goal
    if tactics is not None:
        body["tactics"] = tactics
    if briefing:
        body["briefing"] = briefing
    if situation:
        body["situation"] = situation
    if avoid is not None:
        body["avoid"] = avoid
    if exclude is not None:
        body["exclude"] = exclude
    if note:
        body["note"] = note
    return json.dumps(body, ensure_ascii=False)
