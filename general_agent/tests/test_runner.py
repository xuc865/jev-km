"""Tests for the general JEV harness (stdlib unittest; no pytest dependency)."""

from __future__ import annotations

import unittest

from general_agent import JevPolicy, run_episode
from general_agent.types import Action, Observation


class FakeAdapter:
    name = "fake"

    def __init__(self, win_after: int = 2, extra_actions: int = 0) -> None:
        self.win_after = win_after
        self.extra_actions = extra_actions
        self.steps = 0
        self.closed = False

    def reset(self) -> None:
        self.steps = 0

    def observe(self) -> Observation:
        done = self.steps >= self.win_after
        actions = [] if done else [Action("right", "go right"), Action("wait", "wait")]
        actions.extend(Action(f"extra{i}", f"extra {i}") for i in range(self.extra_actions))
        return Observation(
            actions=actions,
            state={"steps": self.steps},
            done=done,
            success=done,
        )

    def act(self, action: Action) -> None:
        if action.id == "right":
            self.steps += 1

    def close(self) -> None:
        self.closed = True


class FakeScorer:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.calls: list[dict] = []

    def __call__(self, criteria, token, tracks, memory="", images=None):
        self.calls.append(
            {"criteria": criteria, "tracks": tracks, "memory": memory, "images": images}
        )
        return {
            "probabilities": dict(self.probabilities),
            "model": "fake-jev",
            "image_count": 1 if images else 0,
            "usage": {"input_tokens": 10},
        }


class RunnerTests(unittest.TestCase):
    def test_jev_top_choice_wins(self) -> None:
        adapter = FakeAdapter(win_after=2)
        scorer = FakeScorer({"right": 0.9, "wait": 0.1})
        policy = JevPolicy(score_fn=scorer)
        result = run_episode(adapter, policy, "reach the exit", max_steps=10)
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 2)
        self.assertEqual([d.by for d in result.decisions], ["jev", "jev"])
        self.assertTrue(all(d.action_id == "right" for d in result.decisions))
        self.assertTrue(adapter.closed)
        self.assertEqual(scorer.calls[0]["tracks"], {"action": ("right", "wait")})

    def test_low_confidence_escalates_to_outer(self) -> None:
        adapter = FakeAdapter(win_after=1)
        scorer = FakeScorer({"right": 0.55, "wait": 0.45})
        policy = JevPolicy(score_fn=scorer, confidence=0.6)
        seen: list[str] = []

        def escalate(obs, goal, decision):
            seen.append(decision.note)
            return "right"

        result = run_episode(adapter, policy, "goal", max_steps=5, escalate=escalate)
        self.assertTrue(result.success)
        self.assertIn(result.decisions[0].by, {"escalate", "commit"})
        self.assertEqual(result.decisions[0].action_id, "right")
        self.assertEqual(len(seen), 1)

    def test_low_confidence_without_outer_uses_jev_non_wait(self) -> None:
        adapter = FakeAdapter(win_after=1)
        scorer = FakeScorer({"right": 0.51, "wait": 0.49})
        policy = JevPolicy(score_fn=scorer, confidence=0.9)
        result = run_episode(adapter, policy, "goal", max_steps=5)
        self.assertTrue(result.success)
        self.assertEqual(result.decisions[0].by, "jev-low")
        self.assertEqual(result.decisions[0].action_id, "right")

    def test_initial_slow_exception_stops_before_any_input(self) -> None:
        adapter = FakeAdapter(win_after=1)
        scorer = FakeScorer({"right": 0.95, "wait": 0.05})
        def unavailable(*args, **kwargs):
            raise RuntimeError("slow connection unavailable")
        result = run_episode(adapter, JevPolicy(score_fn=scorer), "goal", max_steps=2, escalate=unavailable)
        self.assertFalse(result.success)
        self.assertIn("slow connection unavailable", result.error)
        self.assertEqual(adapter.steps, 0)
        self.assertEqual(scorer.calls, [])
        self.assertTrue(adapter.closed)

    def test_failed_slow_escalation_does_not_execute_one_extra_jev_input(self) -> None:
        adapter = FakeAdapter(win_after=10)
        class Slow:
            hint = ""
            error = ""
            calls = 0
            def __call__(self, *args, **kwargs):
                self.calls += 1
                if self.calls > 1:
                    self.error = "please auth first"
                    raise RuntimeError(self.error)
                self.hint = "Valid initial strategy"
                return None
        slow = Slow()
        scores = iter([{"right": 0.95, "wait": 0.05}, {"right": 0.55, "wait": 0.45}])
        def score(*args, **kwargs):
            return {"model": "test-jev", "probabilities": next(scores)}
        result = run_episode(adapter, JevPolicy(score_fn=score), "goal", max_steps=2, escalate=slow)
        self.assertFalse(result.success)
        self.assertIn("please auth first", result.error)
        self.assertEqual(adapter.steps, 1)
        self.assertEqual(slow.calls, 2)
        self.assertTrue(adapter.closed)

    def test_reset_failure_closes_partially_created_adapter(self) -> None:
        adapter = FakeAdapter()
        def failed_reset():
            raise RuntimeError("browser startup failed")
        adapter.reset = failed_reset
        with self.assertRaisesRegex(RuntimeError, "browser startup failed"):
            run_episode(adapter, JevPolicy(score_fn=FakeScorer({"right": 0.9})), "goal")
        self.assertTrue(adapter.closed)

    def test_no_legal_actions_stops(self) -> None:
        adapter = FakeAdapter(win_after=10)
        adapter.observe = lambda: Observation(actions=[], state={}, done=False)  # type: ignore[method-assign]
        policy = JevPolicy(score_fn=FakeScorer({"right": 0.9}))
        result = run_episode(adapter, policy, "goal", max_steps=3)
        self.assertFalse(result.success)
        self.assertIn("no legal actions", result.error)

    def test_too_many_candidates_stops_without_pruning(self) -> None:
        adapter = FakeAdapter(win_after=10, extra_actions=3)
        policy = JevPolicy(score_fn=FakeScorer({"right": 0.9}), max_candidates=3)
        result = run_episode(adapter, policy, "goal", max_steps=3)
        self.assertFalse(result.success)
        self.assertIn("adapter must prune", result.error)

    def test_image_forwarded_to_scorer(self) -> None:
        adapter = FakeAdapter(win_after=1)
        obs = adapter.observe()
        obs.image = "data:image/jpeg;base64,AAAA"
        adapter.observe = lambda: obs  # type: ignore[method-assign]
        scorer = FakeScorer({"right": 0.9, "wait": 0.1})
        policy = JevPolicy(score_fn=scorer)
        run_episode(adapter, policy, "goal", max_steps=3)
        self.assertEqual(scorer.calls[0]["images"], ["data:image/jpeg;base64,AAAA"])

    def test_bogus_probability_does_not_become_an_action(self) -> None:
        adapter = FakeAdapter(win_after=10)
        scorer = FakeScorer({"bogus": 0.99})
        policy = JevPolicy(score_fn=scorer)
        result = run_episode(adapter, policy, "goal", max_steps=3)
        self.assertFalse(result.success)
        self.assertIn("no probability for a legal action", result.error)


if __name__ == "__main__":
    unittest.main()
