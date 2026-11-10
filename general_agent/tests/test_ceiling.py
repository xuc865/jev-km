"""Ceiling behaviour: what happens when a reply is cut off or never written.

Two distinct failures looked identical in the dataset before this module existed:

* the model spent its whole budget on ``reasoning_content`` and returned no
  ``content`` at all -- the reply is a draft, not an answer;
* the model answered but ran past ``max_tokens``, so the JSON object is
  unterminated and cannot be parsed.

Both produced ``raw_reply`` text that failed validation, which made the prompt
look wrong when the budget was. These tests pin the two cases apart and pin the
recovery: one retry at double the budget, accepted only if the retry yields real
content.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from general_agent.client import ModelClient, _extract_text, _hit_the_ceiling


def body(content=None, reasoning=None, finish=None, completion_tokens=None) -> bytes:
    message = {}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    payload = {"model": "m", "choices": [{"message": message}]}
    if finish:
        payload["choices"][0]["finish_reason"] = finish
    if completion_tokens is not None:
        payload["usage"] = {"completion_tokens": completion_tokens}
    return json.dumps(payload).encode()


class Response:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.data


ANSI = "http://test/model"


def client(**kwargs) -> ModelClient:
    return ModelClient(url=ANSI, token="", attempts=1, **kwargs)


class ExtractionTests(unittest.TestCase):
    def test_a_plain_content_reply_is_an_answer(self) -> None:
        text, draft = _extract_text({"choices": [{"message": {"content": '{"a":1}'}}]})
        self.assertEqual(text, '{"a":1}')
        self.assertFalse(draft)

    def test_content_parts_are_joined(self) -> None:
        raw = {"choices": [{"message": {"content": [
            {"type": "text", "text": "ab"},
            {"type": "text", "text": "cd"},
        ]}}]}
        self.assertEqual(_extract_text(raw), ("abcd", False))

    def test_reasoning_only_reply_is_flagged_as_a_draft(self) -> None:
        raw = {"choices": [{"message": {"content": "", "reasoning_content": "thinking"}}]}
        self.assertEqual(_extract_text(raw), ("thinking", True))

    def test_an_empty_message_yields_nothing(self) -> None:
        self.assertEqual(_extract_text({"choices": [{"message": {}}]}), ("", False))

    def test_a_malformed_body_yields_nothing(self) -> None:
        self.assertEqual(_extract_text({}), ("", False))
        self.assertEqual(_extract_text({"choices": []}), ("", False))


class CeilingTests(unittest.TestCase):
    def test_using_exactly_the_budget_counts_as_cut_off(self) -> None:
        self.assertTrue(_hit_the_ceiling({"completion_tokens": 512}, 512))

    def test_using_less_than_the_budget_is_a_finished_reply(self) -> None:
        self.assertFalse(_hit_the_ceiling({"completion_tokens": 511}, 512))

    def test_no_budget_means_no_judgement(self) -> None:
        self.assertFalse(_hit_the_ceiling({"completion_tokens": 512}, None))

    def test_missing_usage_is_not_treated_as_truncation(self) -> None:
        self.assertFalse(_hit_the_ceiling({}, 512))


class RetryTests(unittest.TestCase):
    def test_a_truncated_reply_is_retried_with_double_the_budget(self) -> None:
        sent = []

        def send(request, timeout):
            sent.append(json.loads(request.data.decode()))
            if len(sent) == 1:
                return Response(body(content='{"action":"ri', completion_tokens=64))
            return Response(body(content='{"action":"right"}', completion_tokens=8))

        with patch("general_agent.client.urllib.request.urlopen", side_effect=send):
            call = client(max_tokens=64).complete(
                role="executor", model="m", system="s", user_payload={"a": 1}
            )
        self.assertEqual([item["max_tokens"] for item in sent], [64, 128])
        self.assertEqual(call.text, '{"action":"right"}')
        self.assertEqual(call.attempts, 2)
        self.assertFalse(call.from_reasoning)

    def test_a_reasoning_draft_is_also_retried(self) -> None:
        sent = []

        def send(request, timeout):
            sent.append(json.loads(request.data.decode()))
            if len(sent) == 1:
                return Response(body(content="", reasoning="let me think", completion_tokens=64))
            return Response(body(content='{"action":"right"}', completion_tokens=9))

        with patch("general_agent.client.urllib.request.urlopen", side_effect=send):
            call = client(max_tokens=64).complete(
                role="planner", model="m", system="s", user_payload={}
            )
        self.assertEqual(len(sent), 2)
        self.assertEqual(call.text, '{"action":"right"}')
        self.assertFalse(call.from_reasoning)

    def test_a_second_draft_is_kept_and_flagged_rather_than_retried_forever(self) -> None:
        def send(request, timeout):
            payload = json.loads(request.data.decode())
            used = payload["max_tokens"]
            return Response(body(content="", reasoning="still thinking", completion_tokens=used))

        with patch("general_agent.client.urllib.request.urlopen", side_effect=send) as open_:
            call = client(max_tokens=64).complete(
                role="planner", model="m", system="s", user_payload={}
            )
        self.assertEqual(open_.call_count, 2)
        self.assertEqual(call.text, "still thinking")
        self.assertTrue(call.from_reasoning)

    def test_a_finished_reply_is_not_retried(self) -> None:
        def send(request, timeout):
            return Response(body(content='{"action":"wait"}', completion_tokens=12))

        with patch("general_agent.client.urllib.request.urlopen", side_effect=send) as open_:
            call = client(max_tokens=512).complete(
                role="executor", model="m", system="s", user_payload={}
            )
        self.assertEqual(open_.call_count, 1)
        self.assertEqual(call.attempts, 1)

    def test_an_empty_retry_leaves_the_first_draft_intact(self) -> None:
        def send(request, timeout):
            return Response(body(content="", completion_tokens=64))

        with patch("general_agent.client.urllib.request.urlopen", side_effect=send):
            call = client(max_tokens=64).complete(
                role="executor", model="m", system="s", user_payload={}
            )
        self.assertEqual(call.error, "model returned empty content")
        self.assertEqual(call.text, "")


if __name__ == "__main__":
    unittest.main()
