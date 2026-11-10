# -*- coding: utf-8 -*-
"""把大量采集进程的打分请求收进一个受控出口。

采集是「一个 episode 一个子进程」，每个子进程各自算自己的节奏，谁也不知道
别人发了多少。十八个并发一起打 llmproxy，实测 60 个请求 1 秒内全部 429，
每局只能挤出 2 到 15 条样本就断流——连接本身没坏，是配额被同伴吃光了。

配额是滑动窗口，只有把进程数、请求数放在同一处才能守住。所以这里起一个
本地小服务：所有子进程把打分请求发给它，它用令牌桶把总量压在窗口内，再
转发上游。上游偶尔回 429 就自动收紧速率并重试，而不是把这个错误原样丢回
给采集端——对采集端来说一次 429 就是一步没有标签，直接变成废样本。

用法：
    python general_agent/rate_proxy.py --limit 90 --port 8799
    curl http://127.0.0.1:8799/stats
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get(
    "JEVGAMER_CHAT_UPSTREAM",
    "http://v2.open.venus.oa.com/llmproxy/chat/completions",
).strip()

try:
    from harness import TOKEN as _HARNESS_TOKEN
except Exception:
    _HARNESS_TOKEN = ""
TOKEN = (_HARNESS_TOKEN or os.environ.get("JEVGAMER_TOKEN") or "").strip()


class Bucket:
    """Sliding-window limiter shared by every caller in this process."""

    def __init__(self, limit: int, window: float = 60.0, floor: int = 20) -> None:
        self.limit = max(1, limit)
        self.ceiling = max(1, limit)
        self.floor = max(1, floor)
        self.window = window
        self.stamps: collections.deque = collections.deque()
        self.lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a slot is free; return how long we waited."""
        waited = 0.0
        while True:
            with self.lock:
                now = time.time()
                while self.stamps and now - self.stamps[0] > self.window:
                    self.stamps.popleft()
                if len(self.stamps) < self.limit:
                    self.stamps.append(now)
                    return waited
                wait = self.window - (now - self.stamps[0]) + 0.02
            wait = max(0.01, min(wait, 5.0))
            time.sleep(wait)
            waited += wait

    def tighten(self) -> int:
        """Upstream said 429: give up some headroom rather than keep probing."""
        with self.lock:
            self.limit = max(self.floor, int(self.limit * 0.85))
            return self.limit

    def relax(self) -> int:
        """A long run of clean answers means we can creep back up."""
        with self.lock:
            self.limit = min(self.ceiling, self.limit + 1)
            return self.limit


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ok = 0
        self.rate_limited = 0
        self.failed = 0
        self.waited = 0.0
        self.latency = 0.0
        self.started = time.time()

    def record(self, outcome: str, waited: float, latency: float) -> None:
        with self.lock:
            self.waited += waited
            self.latency += latency
            if outcome == "ok":
                self.ok += 1
            elif outcome == "429":
                self.rate_limited += 1
            else:
                self.failed += 1

    def snapshot(self, bucket: Bucket) -> dict:
        with self.lock:
            total = self.ok + self.rate_limited + self.failed
            elapsed = max(0.001, time.time() - self.started)
            return {
                "ok": self.ok,
                "rate_limited": self.rate_limited,
                "failed": self.failed,
                "total": total,
                "limit_now": bucket.limit,
                "avg_wait": round(self.waited / total, 3) if total else 0.0,
                "avg_latency": round(self.latency / total, 3) if total else 0.0,
                "throughput_per_min": round(self.ok / elapsed * 60, 1),
                "uptime": round(elapsed, 1),
            }


BUCKET = Bucket(90)
STATS = Stats()
CLEAN = {"n": 0}
CLEAN_LOCK = threading.Lock()


def _post_once(body: bytes, timeout: float) -> tuple[int, bytes, str]:
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = "Bearer " + TOKEN
    request = urllib.request.Request(UPSTREAM, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), ""
    except Exception as exc:
        return 0, b"", "%s: %s" % (type(exc).__name__, str(exc)[:200])


def forward(body: bytes, timeout: float = 60.0, retries: int = 6) -> tuple[int, bytes]:
    """Send one request upstream, absorbing 429 until it fits the budget.

    A 429 returned to the collector costs a training step: the action gets no
    probability label and the row is unusable. Waiting here instead costs only
    wall time, which is the cheaper of the two.
    """
    last_body = b""
    last_status = 0
    for attempt in range(retries):
        waited = BUCKET.acquire()
        started = time.time()
        status, payload, error = _post_once(body, timeout)
        latency = time.time() - started
        if status == 200:
            STATS.record("ok", waited, latency)
            with CLEAN_LOCK:
                CLEAN["n"] += 1
                if CLEAN["n"] % 25 == 0:
                    BUCKET.relax()
            return status, payload
        if status == 429:
            STATS.record("429", waited, latency)
            with CLEAN_LOCK:
                CLEAN["n"] = 0
            limit = BUCKET.tighten()
            # Back off long enough for the window to move; the exact wait is
            # driven by how deep we are over budget, capped so one bad call
            # still cannot stall an episode for minutes.
            pause = min(8.0, 1.5 * (attempt + 1)) if attempt > 0 else 1.5
            time.sleep(pause)
            last_status, last_body = status, payload
            print("[proxy] 429 -> limit=%d, slept %.1fs (attempt %d)" % (limit, pause, attempt + 1),
                  flush=True)
            continue
        if status in (400, 401, 403):
            STATS.record("fail", waited, latency)
            return status, payload or error.encode()
        STATS.record("fail", waited, latency)
        last_status, last_body = status, payload or (error or "").encode()
        if attempt + 1 < retries:
            time.sleep(1.0 * (attempt + 1))
    return last_status or 503, last_body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def _send(self, status: int, payload: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.startswith("/stats"):
            body = json.dumps(STATS.snapshot(BUCKET), ensure_ascii=False).encode()
            self._send(200, body)
            return
        if self.path.startswith("/health"):
            self._send(200, b'{"status":"ready"}')
            return
        self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = 0
        raw = self.rfile.read(size) if size else b""
        try:
            json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            self._send(400, b'{"error":"bad json"}')
            return
        status, payload = forward(raw)
        self._send(status if status else 502, payload or b'{"error":"upstream empty"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--limit", type=int, default=90,
                        help="requests per minute allowed through to the upstream key")
    parser.add_argument("--floor", type=int, default=20,
                        help="lowest limit the auto-tightening may walk down to")
    args = parser.parse_args()

    global BUCKET
    BUCKET = Bucket(args.limit, floor=args.floor)

    print("[proxy] listening on http://%s:%d -> %s" % (args.host, args.port, UPSTREAM), flush=True)
    print("[proxy] budget: %d req/min (floor %d)" % (args.limit, args.floor), flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[proxy] stopping", flush=True)


if __name__ == "__main__":
    main()
