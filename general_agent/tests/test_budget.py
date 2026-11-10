"""Output budget and answer-shape guarantees, per role.

These exist because of a concrete failure: with one shared 512-token ceiling,
every planner review came back as an unterminated JSON object and 100% of the
planner labels were rejected. A truncated reply is not a short reply, it is an
unparseable one, so the budget is a correctness property of the dataset rather
than a tuning knob.

The second half checks the ordering of the rendered payload: the schema must be
the first key, since a model that reads the world block first tends to answer in
prose before it reaches the shape it was asked for.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from general_agent.client import ModelClient
from general_agent.contract import (
    EXECUTOR_SCHEMA_HINT,
    PLANNER_SCHEMA_HINT,
    LegalAction,
    PlanDirective,
    WorldView,
    render_executor_payload,
    render_planner_payload,
)
from general_agent.outer import PlannerPolicy
from general_agent.policy import ExecutorPolicy
from general_agent.prompts import (
    EXECUTOR_SYSTEM,
    PLANNER_SYSTEM,
    executor_system,
    planner_system,
)
from general_agent.types import Action, Observation

from .fake_models import ScriptedClient, executor_reply, planner_reply


def observation() -> Observation:
    return Observation(actions=[Action("right"), Action("wait")], level=1, state={})


def clean_env(**overrides) -> dict:
    """Environment with both budget variables removed.

    ``os.environ`` in a test session may already carry whatever the collector was
    last run with, so a default-value assertion has to clear them explicitly.
    """
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("JEVGAMER_")}
    env.update(overrides)
    return env


class ClientBudgetTests(unittest.TestCase):
    def test_no_ceiling_is_sent_when_none_is_configured(self) -> None:
        client = ModelClient(url="http://test/model", token="")
        payload = client._payload("m", [], 0.0)
        self.assertNotIn("max_tokens", payload)

    def test_one_call_can_raise_the_client_ceiling(self) -> None:
        client = ModelClient(url="http://test/model", token="", max_tokens=512)
        payload = client._payload("m", [], 0.0, 2048)
        self.assertEqual(payload["max_tokens"], 2048)

    def test_without_an_override_the_client_default_still_applies(self) -> None:
        client = ModelClient(url="http://test/model", token="", max_tokens=512)
        payload = client._payload("m", [], 0.0, None)
        self.assertEqual(payload["max_tokens"], 512)


class RoleBudgetTests(unittest.TestCase):
    def test_the_two_roles_get_independent_budgets(self) -> None:
        client = ScriptedClient(
            executor=[executor_reply("right")],
            planner=[planner_reply(stage_goal="g")],
        )
        executor = ExecutorPolicy(client=client, model="fake", max_tokens=200)
        planner = PlannerPolicy(client=client, model="fake", max_tokens=900)
        executor.choose(observation(), "goal", escalate=planner)
        budgets = {item["role"]: item["max_tokens"] for item in client.asked}
        self.assertEqual(budgets["executor"], 200)
        self.assertEqual(budgets["planner"], 900)

    def test_planner_default_leaves_room_for_a_full_briefing(self) -> None:
        # A briefing may fill 1200 characters on its own. A ceiling below roughly
        # a thousand tokens cuts that string, and a cut string is an invalid reply.
        with patch.dict(os.environ, clean_env(), clear=True):
            planner = PlannerPolicy(client=ScriptedClient(), model="fake")
        self.assertGreaterEqual(planner.max_tokens, 1500)

    def test_executor_default_is_tighter_than_the_planner_default(self) -> None:
        with patch.dict(os.environ, clean_env(), clear=True):
            executor = ExecutorPolicy(client=ScriptedClient(), model="fake")
            planner = PlannerPolicy(client=ScriptedClient(), model="fake")
        self.assertLess(executor.max_tokens, planner.max_tokens)
        self.assertGreaterEqual(executor.max_tokens, 512)

    def test_each_role_reads_its_own_environment_override(self) -> None:
        with patch.dict(
            os.environ,
            clean_env(JEVGAMER_EXECUTOR_MAX_TOKENS="300",
                      JEVGAMER_PLANNER_MAX_TOKENS="2600"),
            clear=True,
        ):
            executor = ExecutorPolicy(client=ScriptedClient(), model="fake")
            planner = PlannerPolicy(client=ScriptedClient(), model="fake")
        self.assertEqual(executor.max_tokens, 300)
        self.assertEqual(planner.max_tokens, 2600)

    def test_a_bad_environment_value_falls_back_instead_of_crashing(self) -> None:
        with patch.dict(os.environ, clean_env(JEVGAMER_PLANNER_MAX_TOKENS="lots"),
                        clear=True):
            planner = PlannerPolicy(client=ScriptedClient(), model="fake")
        self.assertGreater(planner.max_tokens, 0)


class AnswerShapeTests(unittest.TestCase):
    def world(self) -> WorldView:
        return WorldView(
            game="demo",
            goal="reach the exit",
            level=1,
            step=4,
            phase="play",
            actions=(LegalAction("right", "go right", "hold", "move", 0.2),),
            state={"hp": 3},
            image_attached=True,
            progress={"metric": "score", "value": 10},
        )

    def test_executor_schema_leads_the_payload(self) -> None:
        payload = render_executor_payload(self.world(), PlanDirective(), ["right"], 1)
        self.assertEqual(next(iter(payload)), "output_schema")
        self.assertEqual(payload["output_schema"], EXECUTOR_SCHEMA_HINT)

    def test_planner_schema_leads_the_payload(self) -> None:
        payload = render_planner_payload(self.world(), PlanDirective(), None, [], "first_step")
        self.assertEqual(next(iter(payload)), "output_schema")
        self.assertEqual(payload["output_schema"], PLANNER_SCHEMA_HINT)

    def test_both_system_prompts_open_with_a_json_only_rule(self) -> None:
        # The rule has to be in the opening lines: a trailing instruction competes
        # with hundreds of tokens of world state for the model's attention.
        for text in (EXECUTOR_SYSTEM, PLANNER_SYSTEM):
            head = text.split("\n\n")[1]
            self.assertIn("OUTPUT CONTRACT", head)
            self.assertIn("exactly one JSON object", head)

    def test_both_prompts_warn_that_non_json_is_not_read(self) -> None:
        # Wording may differ per role; what must not differ is the consequence:
        # a reply that is not one JSON object is never read as an answer.
        for text in (EXECUTOR_SYSTEM, PLANNER_SYSTEM):
            self.assertIn("unread", text)

    def test_identity_and_goal_are_appended_not_interleaved(self) -> None:
        system = executor_system("player one", "score points")
        self.assertTrue(system.startswith(EXECUTOR_SYSTEM))
        self.assertIn("player one", system)
        self.assertIn("score points", system)

    def test_planner_identity_also_keeps_the_body_intact(self) -> None:
        system = planner_system("", "")
        self.assertTrue(system.startswith(PLANNER_SYSTEM))
        self.assertIn("autonomous game player", system)


if __name__ == "__main__":
    unittest.main()
