"""Contract tests for the panel bridge; no browser or model calls."""
import threading
import time
import unittest

from general_agent.adapters.panel_dom import PanelDomAdapter
from general_agent.types import Action


class PanelDomTests(unittest.TestCase):
    def adapter(self):
        return PanelDomAdapter("test", config={"terminal": {
            "success_headings": [r"登\s*顶\s*烬\s*塔"],
            "failure_headings": [r"灰\s*烬\s*湮\s*灭"],
        }})

    def frame(self, signature="one", headings=None):
        return {"signature": signature, "text": "player-visible text",
                "headings": headings or [], "candidates": [
                    {"id": "same_button", "label": "Play", "selector": "#play"}]}

    def test_only_final_heading_is_success(self):
        adapter = self.adapter()
        adapter.update_frame(self.frame(headings=["战斗胜利"]))
        self.assertFalse(adapter.observe().done)
        adapter.update_frame(self.frame(headings=["登 顶 烬 塔"]))
        self.assertTrue(adapter.observe().success)
        adapter.update_frame(self.frame(headings=["灰 烬 湮 灭"]))
        obs = adapter.observe()
        self.assertTrue(obs.done)
        self.assertFalse(obs.success)

    def test_observation_waits_for_stable_signature_despite_feed_heartbeats(self):
        adapter = self.adapter()
        adapter.config["observation_quiet_seconds"] = 0.08
        adapter.update_frame(self.frame("animating"))
        observations = []
        thread = threading.Thread(target=lambda: observations.append(adapter.observe()))
        thread.start()
        time.sleep(0.04)
        adapter.update_frame(self.frame("settled"))
        time.sleep(0.04)
        self.assertTrue(thread.is_alive())
        adapter.update_frame(self.frame("settled"))
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observations[0].raw["signature"], "settled")

    def test_empty_frame_clears_old_controls(self):
        adapter = self.adapter()
        adapter.update_frame(self.frame())
        adapter.update_frame({"signature": "loading", "candidates": []})
        self.assertEqual(adapter._frame["candidates"], [])

    def test_stale_action_is_rejected(self):
        adapter = self.adapter()
        adapter.update_frame(self.frame())
        action = adapter.observe().actions[0]
        adapter.update_frame(self.frame("two"))
        adapter.act(action)
        self.assertIsNone(adapter.pending_click)
        self.assertFalse(adapter.last_execution["ok"])

    def test_same_control_gets_distinct_acknowledged_commands(self):
        adapter = self.adapter()
        adapter.update_frame(self.frame())
        commands = []
        for _ in range(2):
            action = adapter.observe().actions[0]
            thread = threading.Thread(target=adapter.act, args=(action,))
            thread.start()
            with adapter._condition:
                self.assertTrue(adapter._condition.wait_for(lambda: adapter.pending_click is not None, timeout=2))
                command = dict(adapter.pending_click)
            self.assertFalse(adapter.acknowledge({"command_id": "old", "ok": True}))
            self.assertTrue(adapter.acknowledge({"command_id": command["command_id"], "ok": True}))
            commands.append(command["command_id"])
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertNotEqual(*commands)
        self.assertEqual(len(adapter.executions), 2)

    def test_observation_excludes_previous_screen_and_projects_pack_phase(self):
        adapter = self.adapter()
        adapter.config["phases"] = [{"name": "current", "pattern": "player-visible"}]
        adapter.last_execution = {"action": "old", "ok": True, "after_text": "obsolete screen"}
        adapter.update_frame(self.frame())
        obs = adapter.observe()
        self.assertEqual(obs.state["phase"], "current")
        self.assertNotIn("after_text", obs.state["last_execution"])
        self.assertIn("wait", [action.id for action in obs.actions])


    def test_live_panel_preserves_legacy_click_payload(self):
        from types import SimpleNamespace
        from general_agent.live import LiveRun
        run = LiveRun.__new__(LiveRun)
        run.run_id = "run"
        run.steps = 1
        run.adapter = SimpleNamespace(pending_click=Action("button", kind="dom_click", params={"selector": "#button"}))
        first = run._pending_click_payload()
        run.steps = 2
        second = run._pending_click_payload()
        self.assertEqual(first["selector"], "#button")
        self.assertNotEqual(first["command_id"], second["command_id"])


    def test_scroll_controls_remain_environment_actions(self):
        adapter = self.adapter()
        frame = self.frame()
        frame["candidates"] = [{"id": "scroll_down", "label": "Scroll down", "selector": "#region", "kind": "scroll", "direction": "down"}]
        adapter.update_frame(frame)
        action = adapter.observe().actions[0]
        self.assertEqual(action.kind, "scroll")
        self.assertEqual(action.params["direction"], "down")
        self.assertEqual(action.params["selector"], "#region")

if __name__ == "__main__":
    unittest.main()
