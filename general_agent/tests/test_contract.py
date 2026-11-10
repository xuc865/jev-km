"""Tests for the wire contract: parsing, validation, merging, encoding."""

from __future__ import annotations

import json
import unittest

from general_agent.contract import (
    MAX_REVIEW_INTERVAL,
    MIN_REVIEW_INTERVAL,
    SCHEMA_VERSION,
    PlanDirective,
    WorldView,
    LegalAction,
    encode,
    load_json_object,
    parse_executor_reply,
    parse_planner_reply,
    render_executor_payload,
    render_planner_payload,
)


LEGAL = {"right", "left", "wait"}


class ExecutorParsingTests(unittest.TestCase):
    def test_valid_reply_is_accepted_field_by_field(self) -> None:
        reply = parse_executor_reply(
            '{"action":"right","why":"exit is east","confidence":0.8,'
            '"expect":"move east","belief":"open corridor","stuck":false}',
            LEGAL,
        )
        self.assertIsNotNone(reply)
        self.assertEqual(reply.action, "right")
        self.assertEqual(reply.why, "exit is east")
        self.assertAlmostEqual(reply.confidence, 0.8)
        self.assertEqual(reply.expect, "move east")
        self.assertEqual(reply.belief, "open corridor")
        self.assertFalse(reply.stuck)

    def test_action_outside_the_legal_set_is_rejected_not_repaired(self) -> None:
        # A fabricated id means the model misread its own list. Substituting the
        # nearest match would hide the bug behind a working-looking step.
        self.assertIsNone(parse_executor_reply('{"action":"teleport"}', LEGAL))

    def test_missing_action_is_rejected(self) -> None:
        self.assertIsNone(parse_executor_reply('{"why":"no action"}', LEGAL))

    def test_garbage_is_rejected(self) -> None:
        for text in ("", "not json", "{}", "[1,2,3]", '{"action": 5}'):
            self.assertIsNone(parse_executor_reply(text, LEGAL))

    def test_confidence_is_clamped_and_non_numeric_falls_to_zero(self) -> None:
        high = parse_executor_reply('{"action":"right","confidence":7}', LEGAL)
        low = parse_executor_reply('{"action":"right","confidence":-3}', LEGAL)
        bad = parse_executor_reply('{"action":"right","confidence":"very"}', LEGAL)
        self.assertEqual(high.confidence, 1.0)
        self.assertEqual(low.confidence, 0.0)
        self.assertEqual(bad.confidence, 0.0)

    def test_text_fields_are_single_line_and_bounded(self) -> None:
        reply = parse_executor_reply(
            json.dumps({"action": "right", "why": "a\nb\r\nc   d", "belief": "x" * 900}),
            LEGAL,
        )
        self.assertEqual(reply.why, "a b c d")
        self.assertLessEqual(len(reply.belief), 400)

    def test_stuck_accepts_string_booleans(self) -> None:
        self.assertTrue(parse_executor_reply('{"action":"right","stuck":"true"}', LEGAL).stuck)
        self.assertFalse(parse_executor_reply('{"action":"right","stuck":"no"}', LEGAL).stuck)

    def test_fenced_json_is_extracted(self) -> None:
        text = 'Here you go:\n```json\n{"action":"left"}\n```\nDone.'
        reply = parse_executor_reply(text, LEGAL)
        self.assertEqual(reply.action, "left")

    def test_prose_around_json_is_tolerated(self) -> None:
        text = 'I will go {"action":"wait","why":"animating"} now.'
        self.assertEqual(parse_executor_reply(text, LEGAL).action, "wait")


class PlannerParsingTests(unittest.TestCase):
    def test_partial_reply_keeps_every_other_field(self) -> None:
        previous = PlanDirective(
            briefing="rules", stage_goal="survive", tactics="block", situation="calm",
            avoid=("wait",), review_in=9, revision=3, issued_at_step=10,
        )
        reply = parse_planner_reply('{"situation":"under attack"}', previous, 14, LEGAL)
        self.assertIsNotNone(reply)
        self.assertEqual(reply.plan.briefing, "rules")
        self.assertEqual(reply.plan.stage_goal, "survive")
        self.assertEqual(reply.plan.tactics, "block")
        self.assertEqual(reply.plan.situation, "under attack")
        self.assertEqual(reply.plan.avoid, ("wait",))
        self.assertEqual(reply.plan.review_in, 9)
        self.assertEqual(reply.plan.revision, 4)
        self.assertEqual(reply.plan.issued_at_step, 14)

    def test_empty_reply_is_rejected_so_a_working_plan_survives(self) -> None:
        previous = PlanDirective(stage_goal="survive", revision=2)
        self.assertIsNone(parse_planner_reply("{}", previous, 5, LEGAL))
        self.assertIsNone(parse_planner_reply("no json here", previous, 5, LEGAL))

    def test_reply_that_only_echoes_an_empty_briefing_is_rejected(self) -> None:
        previous = PlanDirective(revision=1)
        self.assertIsNone(
            parse_planner_reply('{"briefing":"","stage_goal":"","tactics":""}', previous, 3, LEGAL)
        )

    def test_avoid_drops_unknown_ids_and_accepts_clearing(self) -> None:
        previous = PlanDirective(stage_goal="g", avoid=("wait",))
        reply = parse_planner_reply(
            '{"avoid":["wait","teleport","left"]}', previous, 4, LEGAL
        )
        self.assertEqual(reply.plan.avoid, ("left", "wait"))
        cleared = parse_planner_reply('{"avoid":"clear"}', previous, 4, LEGAL)
        self.assertEqual(cleared.plan.avoid, ())

    def test_omitted_avoid_preserves_the_previous_prohibition(self) -> None:
        previous = PlanDirective(stage_goal="g", avoid=("wait",))
        reply = parse_planner_reply('{"situation":"still calm"}', previous, 4, LEGAL)
        self.assertEqual(reply.plan.avoid, ("wait",))

    def test_review_interval_is_clamped_to_the_band(self) -> None:
        previous = PlanDirective(stage_goal="g")
        tiny = parse_planner_reply('{"review_in":0}', previous, 1, LEGAL)
        huge = parse_planner_reply('{"review_in":9999}', previous, 1, LEGAL)
        self.assertEqual(tiny.plan.review_in, MIN_REVIEW_INTERVAL)
        self.assertEqual(huge.plan.review_in, MAX_REVIEW_INTERVAL)

    def test_verdict_is_restricted_to_the_declared_set(self) -> None:
        previous = PlanDirective(stage_goal="g")
        self.assertEqual(
            parse_planner_reply('{"verdict":"pivot"}', previous, 1, LEGAL).verdict, "pivot"
        )
        self.assertEqual(
            parse_planner_reply('{"verdict":"explode"}', previous, 1, LEGAL).verdict, "continue"
        )

    def test_revision_always_increases(self) -> None:
        previous = PlanDirective(stage_goal="g", revision=7)
        reply = parse_planner_reply('{"stage_goal":"h"}', previous, 9, LEGAL)
        self.assertEqual(reply.plan.revision, 8)


class DirectiveTests(unittest.TestCase):
    def test_age_and_forbids(self) -> None:
        plan = PlanDirective(issued_at_step=4, avoid=("wait",))
        self.assertEqual(plan.age(10), 6)
        self.assertEqual(plan.age(2), 0)
        self.assertTrue(plan.forbids("wait"))
        self.assertFalse(plan.forbids("right"))

    def test_wire_shape_is_declared_and_order_stable(self) -> None:
        plan = PlanDirective(
            briefing="b", stage_goal="g", tactics="t", situation="s",
            avoid=("wait",), review_in=6, revision=2, issued_at_step=3,
        )
        wire = plan.to_wire()
        self.assertEqual(
            list(wire),
            ["schema", "revision", "issued_at_step", "briefing", "stage_goal",
             "tactics", "situation", "avoid", "exclude", "review_in"],
        )
        self.assertEqual(wire["schema"], SCHEMA_VERSION)
        self.assertEqual(wire["avoid"], ["wait"])


class RenderingTests(unittest.TestCase):
    def world(self) -> WorldView:
        return WorldView(
            game="demo",
            goal="reach the exit",
            level=2,
            step=11,
            phase="play",
            actions=(LegalAction("right", "go right", "hold", "move", 0.2),),
            state={"hp": 3},
            image_attached=True,
            progress={"metric": "score", "value": 40},
        )

    def test_executor_payload_declares_exactly_four_sections(self) -> None:
        payload = render_executor_payload(self.world(), PlanDirective(), ["right"], 2)
        self.assertEqual(
            list(payload), ["world", "plan", "recent_actions", "repeat_run", "output_schema"]
        )
        self.assertEqual(payload["repeat_run"], 2)
        self.assertEqual(payload["world"]["step"], 11)

    def test_planner_payload_adds_the_report_and_trigger(self) -> None:
        payload = render_planner_payload(self.world(), PlanDirective(), None, [], "first_step")
        self.assertEqual(
            list(payload),
            ["world", "current_plan", "executor_report", "recent_actions",
             "review_trigger", "output_schema"],
        )
        self.assertEqual(payload["review_trigger"], "first_step")
        self.assertIsNone(payload["executor_report"])

    def test_recent_actions_are_capped(self) -> None:
        payload = render_executor_payload(self.world(), PlanDirective(), [f"a{i}" for i in range(40)], 0)
        self.assertEqual(len(payload["recent_actions"]), 12)

    def test_encode_is_canonical(self) -> None:
        first = encode({"b": 1, "a": [2, 3]})
        second = encode({"b": 1, "a": [2, 3]})
        self.assertEqual(first, second)
        self.assertNotIn(" ", first)
        self.assertIn("路径", encode({"goal": "路径"}))


class JsonExtractionTests(unittest.TestCase):
    def test_returns_none_for_non_objects(self) -> None:
        self.assertIsNone(load_json_object(""))
        self.assertIsNone(load_json_object("[1]"))
        self.assertIsNone(load_json_object("{broken"))

    def test_prefers_a_fenced_block(self) -> None:
        text = 'meta {"x":1}\n```json\n{"y":2}\n```'
        self.assertEqual(load_json_object(text), {"y": 2})


if __name__ == "__main__":
    unittest.main()
