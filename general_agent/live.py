"""Generic JevGamer live panel.

One run thread, one state API, one panel. The left game view is the only part
that changes per game: Astray uses its iframe, GZDoom/H5 use the adapter's
latest captured frame. The right-hand panel is identical for every game.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import uuid
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .adapters.astray import AstrayAdapter
from .adapters.generic_html import GenericHtmlAdapter
from .adapters.gzdoom import GzdoomAdapter
from .adapters.gzdoom import catalog_entry as map_entry
from .adapters.html5 import HtmlGameAdapter
from .adapters.panel_dom import PanelDomAdapter
from .adapters.web_runtime import WebRuntimeAdapter
from .outer import PlannerPolicy
from .client import ModelClient
from .policy import ExecutorPolicy
from .runner import run_episode
from .monitor import EpisodeMonitor
from .sft import RoleRecorder
from webenv.timing import timing_contract

ROOT = Path(__file__).resolve().parent.parent
LIVE_HTML = ROOT / "web" / "live.html"
ASTRAY_URL = "http://127.0.0.1:8766/"
DEFAULT_PORT = 8767

CARD_GAMES = {
    "ember-spire-555b",
    "demo-4d68",
    "demo-70bd",
    "demo-e4c6",
    "kk-poker-534e",
    "demo-46fa",
    "demo-5d72",
    "demo-a68c",
}


def make_adapter(game: str, goal: str):
    if game.lower() == "astray":
        return AstrayAdapter(goal=goal)
    entry = map_entry(game)
    if str(entry.get("entry_path") or "").lower().endswith("run.bat"):
        return GzdoomAdapter(game_id=game, goal=goal)
    if (ROOT / "knowledge" / game / "runtime.json").is_file():
        return WebRuntimeAdapter(game_id=game, goal=goal, entry=entry)
    if game in CARD_GAMES and os.environ.get("JEVGAMER_HEADLESS", "").strip() not in {"1", "true", "yes"}:
        # Panel DOM games need the live panel page to publish frames. That is
        # fine interactively, but a batch run has no panel, so it would block
        # on "panel DOM feed unavailable" until the timeout.
        return PanelDomAdapter(game_id=game, goal=goal, entry=entry)
    # No pack: stay headless instead of driving an OS-visible Chrome window.
    return GenericHtmlAdapter(game_id=game, goal=goal, entry=entry)


def game_catalog() -> list[dict]:
    games = [
        {
            "id": "astray",
            "title": "Astray 3D rolling-ball maze",
            "kind": "astray",
            "description": "3D rolling-ball maze",
        }
    ]
    catalog_path = ROOT / "map" / "games_catalog.json"
    if catalog_path.is_file():
        for entry in json.loads(catalog_path.read_text(encoding="utf-8-sig")):
            games.append(
                {
                    "id": entry["id"],
                    "title": entry.get("title") or entry["id"],
                    "kind": "gzdoom"
                    if str(entry.get("entry_path") or "").lower().endswith("run.bat")
                    else "html5",
                    "description": entry.get("description") or "",
                }
            )
    return games


class LiveRun:
    def __init__(self, game: str, goal: str, token: str = "") -> None:
        self.game = game
        self.goal = goal
        self.adapter = make_adapter(game, goal)
        self.goal = goal or getattr(self.adapter, "goal", "")
        self.run_id = uuid.uuid4().hex[:12]
        self.folder = ROOT / "runs" / "panel" / self.run_id
        self.folder.mkdir(parents=True, exist_ok=True)
        if isinstance(self.adapter, (WebRuntimeAdapter, GenericHtmlAdapter)):
            self.adapter.artifacts = self.folder / "browser"
        identity = str((getattr(self.adapter, "config", {}).get("agent") or {}).get("self")
                       or "You are the autonomous player of the current game. Use only available game controls.")
        config = getattr(self.adapter, "config", {}) or {}
        review_every = int(config.get("outer_review_every", 8))
        # One client for both tiers. They differ in role prompt and expected JSON
        # shape, not in transport, so a second client would only duplicate the
        # retry and recording logic.
        self.client = ModelClient(
            token=token or os.environ.get("JEVGAMER_TOKEN", ""),
            max_tokens=config.get("max_output_tokens"),
            thinking=config.get("slow_generation", {}).get("thinking")
            if isinstance(config.get("slow_generation"), dict) else None,
        )
        self.policy = ExecutorPolicy(
            client=self.client,
            identity=identity,
            max_candidates=int(getattr(self.adapter, "max_candidates", 32)),
            review_every=review_every,
        )
        self.outer = PlannerPolicy(client=self.client, identity=identity)

        if isinstance(self.adapter, GenericHtmlAdapter) and hasattr(self.adapter, "apply_subset"):
            # Narrowing the keyboard space is a standing decision the planner can
            # make, but the adapter still owns the final say on what is usable.
            self.policy.subset_sink = lambda exclude, note="": self.adapter.apply_subset(
                exclude, note=note, source="planner")
        # Records are pulled from the shared client, so nothing has to wrap the
        # policies and no monkey-patching of __call__ is needed.
        self.sft = RoleRecorder(self.game, self.run_id, goal=self.goal)
        self.sft_meta = {"adapter": type(self.adapter).__name__}
        self.kind = (
            "astray"
            if isinstance(self.adapter, AstrayAdapter)
            else "gzdoom"
            if isinstance(self.adapter, GzdoomAdapter)
            else "runtime"
            if isinstance(self.adapter, WebRuntimeAdapter)
            else "generic"
            if isinstance(self.adapter, GenericHtmlAdapter)
            else "html5"
        )
        self.game_url = ""
        if self.kind == "html5":
            entry_page = Path(str(self.adapter.entry.get("entry_path"))).name
            self.game_url = f"http://127.0.0.1:{DEFAULT_PORT}/game/{self.game}/{entry_page}?run={self.run_id}"
        self.running = False
        self.status = "待命"
        self.steps = 0
        self.level = 1
        self.error = ""
        self.obs = None
        self.decision = None
        self.recent = []
        self.result = None
        self.started_at = 0.0
        self._stop = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self.running:
            return False
        self.running = True
        self.status = "启动中"
        self.steps = 0
        self.error = ""
        self.result = None
        self._stop = False
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop = True
        self.status = "正在停下"
        if isinstance(self.adapter, PanelDomAdapter):
            self.adapter.close()

    def set_candidates(self, candidates) -> None:
        if hasattr(self.adapter, "candidates"):
            self.adapter.candidates = list(candidates or [])

    def ack_click(self, payload: dict) -> None:
        if isinstance(self.adapter, PanelDomAdapter):
            self.adapter.acknowledge(payload)
        elif hasattr(self.adapter, "click_acked"):
            self.adapter.click_acked = True

    def set_frame(self, payload: dict) -> None:
        if isinstance(self.adapter, PanelDomAdapter):
            self.adapter.update_frame(payload)
        else:
            self.set_candidates(payload.get("candidates") or [])

    def _adapter_progress(self) -> dict:
        """Uniform progress snapshot across the adapter families.

        GenericHtmlAdapter stores it on `progress`; WebRuntimeAdapter uses
        `metric` because its `progress` attribute is the pack-declared outcome
        tracker. Adapters whose value comes from an external source (GZDoom reads
        its engine log) expose `refresh_progress`, which is called first so a
        sample never publishes a record from an earlier step.
        """
        refresh = getattr(self.adapter, "refresh_progress", None)
        if callable(refresh):
            try:
                refreshed = refresh()
            except Exception:
                refreshed = None
            if isinstance(refreshed, dict):
                return refreshed
        for name in ("progress", "metric"):
            value = getattr(self.adapter, name, None)
            if isinstance(value, dict) and value.get("metric") is not None:
                return value
        return {"metric": "", "sense": "high", "value": None, "best": None,
                "source": "none", "note": "adapter exposes no progress metric"}

    def _run(self) -> None:
        def on_step(step, obs, decision) -> None:
            with self._lock:
                self.steps = step
                self.level = obs.level
                self.obs = obs
                self.decision = decision
                self.recent = (self.recent + [decision])[-8:]
                self.status = (
                    "执行者决策"
                    if decision.by == "executor"
                    else "规划者审核"
                    if decision.by == "planner"
                    else "兜底动作"
                    if decision.by == "fallback"
                    else decision.by
                )

            # Drain the shared client so every model exchange this step becomes a
            # training record. Pulling here means a crash loses at most the step
            # in flight.
            self.sft.step = step
            self.sft.plan_revision = getattr(self.policy, "plan", None).revision if getattr(self.policy, "plan", None) else 0
            self.sft.trigger = getattr(self.outer, "last_trigger", "") or ""
            self.sft.set_progress(self._adapter_progress())
            self.sft.drain(self.client)

            record = {"step": step, "at": time.time(), "elapsed": round(time.time() - self.started_at, 3),
                      "observation": asdict(obs), "decision": asdict(decision),
                      "plan": getattr(self.policy, "plan", None).to_wire() if getattr(self.policy, "plan", None) else None,
                      "executor_step": (
                          self.policy.last_step.to_wire()
                          if getattr(self.policy, "last_step", None) else None
                      ),
                      "progress": self._adapter_progress(),
                      "last_execution": getattr(self.adapter, "last_execution", {})}
            with (self.folder / "trace.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        try:
            config = getattr(self.adapter, "config", {})
            self.result = run_episode(
                self.adapter,
                self.policy,
                self.goal,
                max_steps=int(config.get("max_steps", 400)),
                max_seconds=float(config.get("max_seconds", 600)),
                on_step=on_step,
                escalate=self.outer,
                should_stop=lambda: self._stop,
            )
            self.status = "任务完成" if self.result.success else "已停止"
            self.error = self.result.error
        except Exception as exc:
            self.error = str(exc)[:300]
            self.status = "出错"
        finally:
            summary = {"game": self.game, "goal": self.goal, "run_id": self.run_id,
                       "result": asdict(self.result) if self.result else None,
                       "error": self.error, "executor_model": self.policy.model,
                       "planner_model": self.outer.model,
                       "planner_revisions": self.outer.revisions,
                       "planner_error": self.outer.error,
                       "final_plan": self.outer.plan.to_wire(),
                       "model_calls": len(self.client.calls),
                       "executions": getattr(self.adapter, "executions", []),
                       "sft": self.sft.summary(),
                       "final_visible_frame": getattr(self.adapter, "_frame", getattr(self.adapter, "_state", {}))}
            try:
                temporary=self.folder/"summary.json.tmp"
                temporary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
                temporary.replace(self.folder/"summary.json")
            except Exception as exc:
                self.error="summary write failed: "+str(exc)[:240]
                self.status="出错"
            finally:
                # CLI may exit as soon as running becomes false. Publish that
                # flag only after the complete audit artifact is on disk.
                self.running=False

    def _pending_click_payload(self) -> dict | None:
        pending = getattr(self.adapter, "pending_click", None)
        if pending is None:
            return None
        if isinstance(pending, dict):
            return dict(pending)
        # Preserve the original screenshot-backed HTML adapter's bridge.
        return {"id": pending.id, "command_id": f"{self.run_id}:{self.steps}",
                "selector": pending.params.get("selector"),
                "x": pending.params.get("x"), "y": pending.params.get("y")}

    def snapshot(self) -> dict:
        with self._lock:
            obs = self.obs
            decision = self.decision
            actions = []
            if obs is not None:
                for action in obs.actions:
                    actions.append(
                        {
                            "id": action.id,
                            "label": action.label or action.id,
                            "clearance": action.params.get("clearance"),
                        }
                    )
            recent = [
                {
                    "by": item.by,
                    "choice": item.action_id,
                    "model": item.model,
                    "ms": item.ms,
                    "note": item.note,
                }
                for item in self.recent
            ]
            plan = self.policy.plan
            step = self.policy.last_step
            return {
                "game": self.game,
                "run_id": self.run_id,
                "run_folder": str(self.folder),
                "timing_contract": timing_contract(self.adapter.config) if isinstance(self.adapter,WebRuntimeAdapter) else None,
                "goal": self.goal,
                "executor_model": self.policy.model,
                "planner_model": self.outer.model,
                "planner_calls": self.outer.revisions,
                "planner_error": self.outer.error,
                "planner_verdict": self.outer.verdict,
                "model_calls": len(self.client.calls),
                "kind": self.kind,
                "running": self.running,
                "status": self.status,
                "steps": self.steps,
                "level": self.level,
                "elapsed": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
                "actions": actions,
                "choice": decision.action_id if decision else "",
                "by": decision.by if decision else "",
                "model": decision.model if decision else "",
                "ms": decision.ms if decision else 0,
                "note": decision.note if decision else "",
                "briefing": plan.briefing,
                "stage_goal": plan.stage_goal,
                "tactics": plan.tactics,
                "situation": plan.situation,
                "avoid": list(plan.avoid),
                "exclude": list(plan.exclude),
                "review_in": plan.review_in,
                "plan_revision": plan.revision,
                "plan_age": plan.age(getattr(self.policy, "steps", 0)) if plan.revision else 0,
                "stage_age": plan.age(getattr(self.policy, "steps", 0)) if plan.revision else 0,
                "executor": {
                    "why": step.why if step else "",
                    "confidence": round(step.confidence, 3) if step else 0.0,
                    "expect": step.expect if step else "",
                    "belief": step.belief if step else "",
                    "stuck": step.stuck if step else False,
                    "repeat_run": step.repeat_run if step else 0,
                    "outcome": step.outcome if step else "",
                },
                "recent": recent,
                "error": self.error,
                "image": getattr(self.adapter, "preview", "") or (obs.image if obs else ""),
                "astray_url": ASTRAY_URL if self.kind == "astray" else "",
                "game_url": self.game_url,
                "candidate_count": len(getattr(self.adapter, "candidates", []) or []),
                "pending_click": self._pending_click_payload(),
            }


class Handler(BaseHTTPRequestHandler):
    run: LiveRun | None = None
    episode_monitor = EpisodeMonitor(ROOT / "runs" / "panel")

    def log_message(self, *args) -> None:
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/__jev/dom.js":
            self._send((ROOT / "webenv" / "dom.js").read_bytes(), "application/javascript; charset=utf-8")
            return
        if path == "/__jev/panel_bridge.js":
            self._send((ROOT / "web" / "panel_bridge.js").read_bytes(), "application/javascript; charset=utf-8")
            return
        if path == "/live":
            self._send(LIVE_HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/runtime/runs":
            self._json({"runs": self.episode_monitor.snapshots()})
            return
        if path == "/api/live/games":
            self._json({"games": game_catalog()})
            return
        if path == "/api/live/state":
            self._json(self.run.snapshot() if self.run is not None else {"running": False})
            return
        if path.startswith("/game/"):
            parts = path.split("/", 3)
            if len(parts) < 4:
                self._json({"error": "bad game path"}, 404)
                return
            try:
                entry = map_entry(parts[2])
                base = (ROOT / "map" / str(entry.get("entry_path"))).parent.resolve()
                target = (base / parts[3]).resolve()
            except Exception:
                self._json({"error": "game not found"}, 404)
                return
            if target != base and base not in target.parents:
                self._json({"error": "forbidden"}, 403)
                return
            if not target.is_file():
                self._json({"error": "not found"}, 404)
                return
            types = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".woff": "font/woff",
                ".woff2": "font/woff2",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
            }
            self._send(target.read_bytes(), types.get(target.suffix.lower(), "application/octet-stream"))
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            body = {}
        if path == "/api/live/start":
            game = str(body.get("game") or "astray")
            goal = str(body.get("goal") or "")
            if self.run is not None:
                self.run.stop()
            Handler.run = LiveRun(game, goal)
            Handler.run.start()
            self._json(Handler.run.snapshot())
            return
        if path == "/api/live/stop":
            if self.run is not None:
                self.run.stop()
            self._json({"ok": True})
            return
        if path == "/api/live/frame":
            if self.run is None or body.get("run_id") != self.run.run_id or body.get("game") != self.run.game:
                self._json({"error": "stale run"}, 409)
                return
            self.run.set_frame(body)
            self._json({"ok": True})
            return
        if path == "/api/live/candidates":
            if self.run is not None:
                self.run.set_candidates(body.get("candidates") or [])
            self._json({"ok": True})
            return
        if path == "/api/live/clicked":
            if self.run is not None:
                if body.get("run_id") == self.run.run_id:
                    self.run.ack_click(body)
            self._json({"ok": True})
            return
        self._json({"error": "not found"}, 404)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"JevGamer live panel: http://127.0.0.1:{args.port}/live", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
