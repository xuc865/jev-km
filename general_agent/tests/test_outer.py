"""Tests for the outer escalation parser (no network)."""

from __future__ import annotations

import json
import unittest

from general_agent.outer import OuterEscalation
from general_agent.types import Action, Decision, Observation


class FakeChat:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def __call__(self, messages, token, *, model=None):
        self.calls.append({"messages": messages, "token": token, "model": model})
        return {"choices": [{"message": {"content": self.text}}]}


class OuterTests(unittest.TestCase):
    def test_returns_legal_action_and_hint(self) -> None:
        chat = FakeChat(json.dumps({"action": "right", "hint": "go right"}))
        outer = OuterEscalation(token="t", model="m", chat_fn=chat)
        obs = Observation(
            actions=[Action("right"), Action("wait")],
            image="data:image/jpeg;base64,AAAA",
            state={"cell": [1, 1]},
        )
        decision = Decision("", by="low", probabilities={"right": 0.4, "wait": 0.6})
        action = outer(obs, "reach exit", decision)
        self.assertEqual(action, "right")
        self.assertEqual(outer.hint, "go right")
        self.assertEqual(len(chat.calls), 1)
        content = chat.calls[0]["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(json.loads(content[0]["text"])["image_attached"])
        self.assertEqual(outer.calls[0]["image_count"], 1)

    def test_empty_action_returns_none(self) -> None:
        chat = FakeChat('{"action":"","hint":"look around"}')
        outer = OuterEscalation(chat_fn=chat)
        action = outer(Observation(actions=[Action("a")], image=""), "goal", Decision(""))
        self.assertIsNone(action)
        self.assertEqual(outer.hint, "look around")
        content = chat.calls[0]["messages"][1]["content"]
        self.assertEqual(len(content), 1)
        self.assertFalse(json.loads(content[0]["text"])["image_attached"])
        self.assertEqual(outer.calls[0]["image_count"], 0)

    def test_garbage_returns_none(self) -> None:
        outer = OuterEscalation(chat_fn=FakeChat("not json"))
        action = outer(Observation(actions=[Action("a")], image=""), "goal", Decision(""))
        self.assertIsNone(action)


    def test_pack_generation_budget_is_forwarded_without_overwriting_the_goal(self):
        recorded=[]
        def chat(messages,token,**kwargs):
            recorded.append((messages,kwargs))
            return {'model':'actual-slow','choices':[{'message':{'content':'{"action":"a"}'}}]}
        outer=OuterEscalation(chat_fn=chat,generation={'thinking':True,'max_output_tokens':4096})
        self.assertEqual(outer(Observation(actions=[Action('a')]),'locked goal'),'a')
        self.assertEqual(recorded[0][1]['thinking'],True)
        self.assertEqual(recorded[0][1]['max_output_tokens'],4096)
        self.assertIn('locked goal',recorded[0][0][0]['content'])
        self.assertEqual(outer.calls[0]['generation'],{'thinking':True,'max_output_tokens':4096})


if __name__ == "__main__":
    unittest.main()
