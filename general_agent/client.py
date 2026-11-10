"""One model client, two roles.

The previous design had two transport paths with different semantics: a
``choice``-scoring endpoint for the fast tier that returned probabilities over a
criteria map, and a chat endpoint for the slow tier that returned JSON prose. The
scoring endpoint could not name an action outside the map it was given, which is
why the fast tier had to be handed a fully enumerated candidate list every step,
and why the planner ended up authoring per-action usage conditions to make that
list meaningful.

With one model family serving both tiers there is no reason to keep two paths.
Both call ``chat/completions`` and differ only in the system prompt and the
expected JSON shape. What the task still needs is:

- retries that distinguish throttling from a dead endpoint, because a 429 is
  back-pressure while a hung socket is not;
- one place that records every exchange in contract-encoded form, so the SFT
  writer never re-derives a prompt from loose parts;
- a hard token ceiling, because the slow tier's reasoning can otherwise consume
  the whole budget and return an empty ``content``.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

# A 429 on a shared key is not a failure of this call, it is a queue. Measured on
# the shared llmproxy key: single calls answer in ~0.5s, but eighteen concurrent
# collectors produced bursts of sixty requests per second and every one of them
# came back 429. Treating that as an error discarded the step's label; waiting
# recovers the same step.
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_SLEEP = 1.6


@dataclass
class ModelCall:
    """One request/response pair, kept in wire form for the SFT writer.

    ``messages`` is stored exactly as sent rather than as the parts it was built
    from. A distilled student has to train on the token sequence the teacher
    actually saw, and rebuilding it later from the world/plan objects is the
    exact place a silent prompt/label drift used to appear.

    ``schema`` records which contract version produced the prompt, so a dataset
    assembled across a refactor can be filtered instead of silently mixing two
    incompatible prompt shapes.

    ``label`` is filled in by whichever policy accepted the reply. Keeping it here
    rather than in the recorder is what lets the record carry the exact
    contract object that was validated: the policy is the only place that knows
    whether a reply passed validation, and re-parsing it later in the writer would
    duplicate that logic and could disagree with it.
    """

    role: str
    model: str
    messages: list[dict]
    text: str
    ms: int
    usage: dict = field(default_factory=dict)
    error: str = ""
    status: int = 0
    attempts: int = 1
    schema: str = ""
    label: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "schema": self.schema,
            "role": self.role,
            "model": self.model,
            "messages": self.messages,
            "text": self.text,
            "ms": self.ms,
            "usage": self.usage,
            "error": self.error,
            "status": self.status,
            "attempts": self.attempts,
            "label": self.label,
        }


class ModelClient:
    """Chat-completions client shared by both tiers.

    Attributes are plain and settable so a test can point ``url`` at a local stub
    without patching module globals, which is how the previous design made
    transport tests depend on import order.
    """

    def __init__(
        self,
        url: str = "",
        token: str = "",
        timeout: float = 90.0,
        attempts: int = 3,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> None:
        self.url = url or _default_url()
        self.token = token
        self.timeout = max(3.0, float(timeout))
        self.attempts = max(1, int(attempts))
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.calls: list[ModelCall] = []
        self._pending: list[ModelCall] = []
        self.last_error = ""

    def _payload(self, model: str, messages: list[dict], temperature: float) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)
        # Explicit reasoning switch: left on, the slow tier spent ~1250 completion
        # tokens and 10.6s per review on a reply that is a small JSON object, and
        # sometimes consumed the whole budget so `content` came back empty.
        if self.thinking is not None:
            payload["chat_template_kwargs"] = {"thinking": bool(self.thinking)}
        return payload

    def _send(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        last: dict = {}
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8", "replace"))
                if not isinstance(parsed, dict):
                    raise ValueError("model service returned non-object JSON")
                parsed["_attempts"] = attempt + 1
                return parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                last = {"_status": exc.code, "_error": detail}
                # Auth failures are definitive: retrying cannot fix a bad token,
                # and an HTML error page from a gateway is not a model reply.
                if exc.code in {401, 403}:
                    return last
                if exc.code == 429:
                    # Back-pressure. Sleep and ask the identical question again
                    # rather than converting a throttled step into a lost label.
                    if attempt + 1 < self.attempts:
                        time.sleep(RATE_LIMIT_SLEEP * (attempt + 1))
                        continue
                    return last
                if exc.code not in RETRYABLE_STATUS:
                    return last
            except (urllib.error.URLError, socket.timeout, TimeoutError,
                    ConnectionResetError, OSError, ValueError, UnicodeError) as exc:
                last = {"_error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            if attempt + 1 < self.attempts:
                time.sleep(1.2 * (attempt + 1))
        last["_attempts"] = self.attempts
        return last

    def complete(
        self,
        role: str,
        model: str,
        system: str,
        user_payload: dict,
        *,
        temperature: float = 0.0,
    ) -> ModelCall:
        """One completion, recorded whether it succeeds or fails.

        The user turn is contract-encoded so its bytes are identical for
        identical situations. Recording the failure path matters: a run whose slow
        tier never answered is not training data, and the SFT writer decides that
        from ``error`` rather than from an empty ``text``.
        """
        from .contract import SCHEMA_VERSION, encode

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": encode(user_payload)},
        ]
        started = time.monotonic()
        raw = self._send(self._payload(model, messages, temperature))
        elapsed = int((time.monotonic() - started) * 1000)

        error = str(raw.get("_error") or "")
        status = int(raw.get("_status") or 0)
        attempts = int(raw.get("_attempts") or 1)
        text = ""
        if not error:
            text = _extract_text(raw)
            if not text:
                error = "model returned empty content"
        self.last_error = error
        call = ModelCall(
            role=role,
            model=str(raw.get("model") or model),
            messages=messages,
            text=text,
            ms=elapsed,
            usage=raw.get("usage") or {},
            error=error,
            status=status,
            attempts=attempts,
            schema=SCHEMA_VERSION,
        )
        self.calls.append(call)
        self._pending.append(call)
        return call

    def drain_calls(self) -> list[ModelCall]:
        """Return calls recorded since the previous drain and forget them.

        The SFT writer pulls rather than being pushed to, so the client stays
        unaware of the training pipeline. Keeping the queue bounded also stops an
        unattended run from retaining every prompt in memory.
        """
        pending = self._pending
        self._pending = []
        return pending


def _extract_text(raw: dict) -> str:
    """Pull the assistant text out of a completion body.

    Handles three shapes that all occur here: a plain string, a content-part
    list, and a reply whose visible text is empty because a reasoning model spent
    the whole budget thinking. In the last case the JSON is frequently already
    drafted inside ``reasoning_content``, so it is returned as a last resort --
    a parsed answer beats a discarded step.
    """
    try:
        message = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = message.get("content")
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content if isinstance(part, dict)).strip()
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = message.get("reasoning_content")
    return str(reasoning).strip() if isinstance(reasoning, str) else ""


def _default_url() -> str:
    """Resolve the chat endpoint, preferring the local rate proxy when up.

    Several collectors share one key. Without a single owner of that budget, each
    process independently discovered the limit by being throttled. The proxy
    exists to own it, so it wins when it is listening; the direct URL remains the
    fallback for a single-process run.
    """
    direct = os.environ.get(
        "JEVGAMER_CHAT",
        "http://v2.open.venus.oa.com/llmproxy/chat/completions",
    ).strip()
    if os.environ.get("JEVGAMER_CHAT_NO_PROXY"):
        return direct
    proxy = os.environ.get(
        "JEVGAMER_CHAT_PROXY", "http://127.0.0.1:8799/v1/chat/completions"
    ).strip()
    try:
        parts = urllib.parse.urlparse(proxy)
        with socket.create_connection((parts.hostname, parts.port or 80), timeout=0.5):
            return proxy
    except Exception:
        return direct
