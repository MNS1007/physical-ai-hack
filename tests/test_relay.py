"""The relay's state machine with a fake Dream-1 link and scripted verdicts: no robot, no LiveKit, no API calls.

    uv run python -m unittest discover tests
"""
import asyncio
import base64
import unittest

import numpy as np

from physical_ai_hack.judge import Verdict, encode, question
from physical_ai_hack.relay import Config, Relay
from physical_ai_hack.stages import RELAY, score


class FakeLink:
    def __init__(self):
        self.calls = []

    async def set_task(self, task):
        self.calls.append(("set_task", task))

    async def resume(self):
        self.calls.append(("resume",))

    async def pause(self):
        self.calls.append(("pause",))

    async def to_teleop(self):
        self.calls.append(("to_teleop",))


def v(met=False, count=0, clear=False, later=False):
    return Verdict(goal_met=met, count=count, gripper_clear=clear, later_stage_contact=later, reason="test")


def run(coro):
    return asyncio.run(coro)


class RelayTest(unittest.TestCase):
    def setUp(self):
        self.link = FakeLink()
        self.said = []
        self.relay = Relay(self.link, Config(), announce=self.said.append)
        run(self.relay.start(0.0))

    def test_start_sets_first_task_and_resumes(self):
        self.assertEqual(self.link.calls, [("set_task", RELAY[0].prompts[0]), ("resume",)])

    def test_one_met_verdict_does_not_advance(self):
        run(self.relay.on_verdict(v(True, 1), 10.0, 1.0))
        self.assertEqual(self.relay.step.name, "unplug")

    def test_a_dip_restarts_the_hold(self):
        run(self.relay.on_verdict(v(True, 1), 10.0, 1.0))
        run(self.relay.on_verdict(v(False, 0), 11.0, 1.0))
        run(self.relay.on_verdict(v(True, 1), 12.0, 1.0))
        self.assertEqual(self.relay.step.name, "unplug")

    def test_held_unplug_advances_to_stack(self):
        run(self.relay.on_verdict(v(True, 1), 10.0, 1.0))
        run(self.relay.on_verdict(v(True, 1), 12.5, 1.0))
        self.assertEqual(self.relay.step.name, "stack")
        self.assertEqual(self.relay.passed, {1: 12.5})
        self.assertEqual(self.link.calls[-2:], [("set_task", RELAY[1].prompts[0]), ("resume",)])

    def test_stack_needs_three_layers_and_a_clear_gripper(self):
        self._pass_unplug()
        run(self.relay.on_verdict(v(True, 3, clear=False), 40.0, 1.0))
        run(self.relay.on_verdict(v(True, 3, clear=False), 44.0, 1.0))
        self.assertFalse(self.relay.done)
        run(self.relay.on_verdict(v(True, 2, clear=True), 45.0, 1.0))
        run(self.relay.on_verdict(v(True, 2, clear=True), 49.0, 1.0))
        self.assertFalse(self.relay.done)

    def test_stack_held_three_seconds_finishes_and_pauses(self):
        self._pass_unplug()
        run(self.relay.on_verdict(v(True, 3, clear=True), 40.0, 1.0))
        run(self.relay.on_verdict(v(True, 3, clear=True), 43.0, 1.0))
        self.assertTrue(self.relay.done)
        self.assertEqual(self.link.calls[-1], ("pause",))
        self.assertIn("CALL STOP NOW", self.said[-1])
        self.assertNotIn(("to_teleop",), self.link.calls)

    def test_stuck_retries_with_next_prompt_never_teleop(self):
        run(self.relay.tick(61.0, still_for_s=0.0))
        self.assertEqual(self.link.calls[-3:], [("pause",), ("set_task", RELAY[0].prompts[1]), ("resume",)])
        run(self.relay.tick(62.0, still_for_s=20.0))
        self.assertEqual(self.link.calls[-2], ("set_task", RELAY[0].prompts[2]))
        self.assertNotIn(("to_teleop",), self.link.calls)

    def test_progress_resets_the_timeout(self):
        self._pass_unplug()
        run(self.relay.on_verdict(v(False, 2), 50.0, 1.0))
        run(self.relay.tick(100.0, still_for_s=0.0))  # 50 s after progress, under the stack's 75 s
        self.assertEqual(self.relay.attempt, 0)

    def test_practice_mode_hands_to_teleop_on_stuck(self):
        link = FakeLink()
        relay = Relay(link, Config(practice_teleop=True), announce=lambda _: None)
        run(relay.start(0.0))
        run(relay.tick(61.0, still_for_s=0.0))
        self.assertEqual(link.calls[-1], ("to_teleop",))

    def test_dry_run_never_calls_the_link(self):
        link = FakeLink()
        relay = Relay(link, Config(dry_run=True), announce=lambda _: None)
        run(relay.start(0.0))
        run(relay.tick(61.0, still_for_s=0.0))
        run(relay.on_verdict(v(True, 1), 70.0, 1.0))
        run(relay.on_verdict(v(True, 1), 73.0, 1.0))
        self.assertEqual(link.calls, [])

    def test_match_clock_ends_the_run(self):
        run(self.relay.tick(300.0, still_for_s=0.0))
        self.assertTrue(self.relay.done)

    def test_dq_risk_is_counted(self):
        run(self.relay.on_verdict(v(False, 0, later=True), 5.0, 1.0))
        self.assertEqual(self.relay.dq_warnings, 1)

    def test_unplug_judge_is_told_blocks_are_later(self):
        self.assertIn("stacking blocks", self.relay.later_objects())

    def _pass_unplug(self):
        run(self.relay.on_verdict(v(True, 1), 10.0, 1.0))
        run(self.relay.on_verdict(v(True, 1), 12.5, 1.0))


class ScoreAndJudgeTest(unittest.TestCase):
    def test_score_matches_rubric(self):
        self.assertEqual(score(True, True, 120.0), 10 + 20 + 90)
        self.assertEqual(score(True, False, 120.0), 10)
        self.assertEqual(score(True, True, None), 30)

    def test_encode_shrinks_and_returns_jpeg(self):
        data = base64.b64decode(encode(np.zeros((1080, 1920, 3), dtype=np.uint8)))
        self.assertEqual(data[:2], b"\xff\xd8")

    def test_question_names_the_count(self):
        self.assertIn("layers", question(RELAY[1], ""))


if __name__ == "__main__":
    unittest.main()
