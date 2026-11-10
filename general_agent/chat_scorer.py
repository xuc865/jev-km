# -*- coding: utf-8 -*-
"""用可用的 chat 通道模拟 SystemOne 的 choice 打分。

现场发现 WeMM-JEV（http://29.116.237.135:8080/v1/systemone）已经在 TCP 层可连
但 HTTP 请求永久不返回：20s、90s、120s 三种超时全部耗尽，说明推理进程挂死
而不是网络不通。慢模型通道（llmproxy）却是 0.4s 正常返回。

采集需要的是"给定一组动作条件，选出哪一条成立"，这本质上就是一个分类问题，
chat 模型也能做。此模块把同样的 criteria 转成 chat prompt，要求返回严格 JSON，
解析成与 jev_scores 相同结构的 {probabilities, tracks, error, model}，从而在
WeMM-JEV 挂掉时仍能继续产出可用样本。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

DIRECT = os.environ.get("JEVGAMER_CHAT", "http://v2.open.venus.oa.com/llmproxy/chat/completions").strip()
# Every collector is its own process, so each one used to spend the shared key's
# quota without knowing about the others: eighteen concurrent episodes put sixty
# requests in one second and all of them came back 429, which turned a step with
# no label into a wasted step and starved whole episodes down to a handful of
# samples. When the local rate proxy is listening it owns the budget, so prefer
# it and let the direct URL stay as the fallback for single-process runs.
PROXY = os.environ.get("JEVGAMER_CHAT_PROXY", "http://127.0.0.1:8799/v1/chat/completions").strip()


def _pick_url() -> str:
    if os.environ.get("JEVGAMER_CHAT_NO_PROXY"):
        return DIRECT
    try:
        import socket
        from urllib.parse import urlparse
        parts = urlparse(PROXY)
        with socket.create_connection((parts.hostname, parts.port or 80), timeout=0.5):
            return PROXY
    except Exception:
        return DIRECT


CHAT_URL = _pick_url()
# The project keeps its credentials in `.env`, which harness loads at import
# time. Reading the environment directly would leave TOKEN empty whenever this
# module is imported before harness and every call would fail with retCode 10401.
try:
    from harness import TOKEN as _HARNESS_TOKEN
except Exception:
    _HARNESS_TOKEN = ""
TOKEN = (_HARNESS_TOKEN or os.environ.get("JEVGAMER_TOKEN") or os.environ.get("LLM_TOKEN") or "").strip()
MODEL = os.environ.get("JEVGAMER_FALLBACK_FAST", "deepseek-v4.1-flash").strip()

INSTRUCTIONS = (
    "You are scoring which single action condition is satisfied right now.\n"
    "Read the world description and the candidate criteria, then output ONLY a JSON object "
    "of the form {\"probabilities\": {\"<action_id>\": <float 0..1>}} covering every candidate id.\n"
    "The values must sum to about 1. Do not add prose, code fences or explanation."
)


def _post(payload: dict, timeout: float = 60.0, attempts: int = 5) -> dict:
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = "Bearer " + TOKEN
    body = json.dumps(payload).encode()
    last = ""
    for attempt in range(max(1, attempts)):
        try:
            request = urllib.request.Request(CHAT_URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            last = "%s %s" % (exc.code, exc.read().decode("utf-8", "replace")[:200])
            if exc.code in {400, 401, 403}:
                break
            # A throttled step is not a bad step: waiting retries the same
            # question, whereas returning the error throws the label away and
            # the collected frame becomes a row with no target.
            if exc.code == 429:
                import time
                time.sleep(min(6.0, 1.2 * (attempt + 1)))
                continue
        except Exception as exc:
            last = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        if attempt + 1 < attempts:
            import time
            time.sleep(1.5 * (attempt + 1))
    return {"_error": last}


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def score(criteria: dict, memory: str = "", images: list | None = None) -> dict:
    """Return a jev_scores-shaped result from the chat model."""
    if not criteria:
        return {"probabilities": {}, "tracks": {}, "error": "no criteria", "model": MODEL}
    listing = "\n".join("- %s: %s" % (name, str(cond)[:400]) for name, cond in criteria.items())
    prompt = "%s\n\nWorld:\n%s\n\nCandidates:\n%s" % (INSTRUCTIONS, (memory or "(none)")[:4000], listing)
    # 400 tokens was not enough for a reasoning model: deepseek-v4.1-flash writes
    # its thoughts into `reasoning_content`, so the budget was consumed before any
    # answer was emitted and `content` came back empty. Measured on a 15-candidate
    # step, reasoning alone ran 1391-2158 tokens, so the ceiling has to clear that
    # plus the ~280-token JSON body. Without this, only ~25% of calls parsed and
    # the channel looked broken rather than starved.
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(os.environ.get("JEVGAMER_CHAT_MAX_TOKENS") or 2400),
    }
    try:
        body = _post(payload)
    except Exception as exc:  # noqa: BLE001 - scoring must never raise into a step
        return {"probabilities": {}, "tracks": {}, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200]), "model": MODEL}
    if body.get("_error"):
        return {"probabilities": {}, "tracks": {}, "error": str(body["_error"])[:400], "model": MODEL}
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {"probabilities": {}, "tracks": {}, "error": "malformed chat response", "model": MODEL}
    if isinstance(text, list):
        text = "".join(str(part.get("text") or "") for part in text if isinstance(part, dict))
    if not text:
        # A reasoning model can still emit only thoughts when the budget or the
        # server trims the answer; the JSON is often already drafted inside that
        # trace, so parse it rather than dropping a usable label.
        try:
            text = str(body["choices"][0]["message"].get("reasoning_content") or "")
        except (KeyError, IndexError, TypeError):
            text = ""
    parsed = _extract_json(text)
    if not parsed:
        return {"probabilities": {}, "tracks": {}, "error": "unparsable scoring json", "model": MODEL}
    raw = parsed.get("probabilities") or parsed
    clean = {}
    for name, value in raw.items():
        if name not in criteria:
            continue
        try:
            clean[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    if not clean:
        return {"probabilities": {}, "tracks": {}, "error": "no usable probability", "model": MODEL}
    total = sum(clean.values())
    if total > 0:
        clean = {name: round(value / total, 6) for name, value in clean.items()}
    return {
        "probabilities": clean,
        "tracks": {"action": clean},
        "error": None,
        "model": MODEL,
        "usage": body.get("usage") or {},
        "image_count": len(images or []),
    }
