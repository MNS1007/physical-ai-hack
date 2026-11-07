"""The relay's state machine with a fake Dream-1 link: no robot, no LiveKit.

    uv run python -m unittest discover tests
"""
import asyncio
import unittest

from physical_ai_hack.relay import Config, Relay
from physical_ai_hack.stages import RELAY, score

UNPLUG, STACK = RELAY


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


def run(coro):
    return asyncio.run(coro)


class RelayTest(unittest.TestCase):
    def setUp(self):
        self.link = FakeLink()
        self.said = []
        self.relay = Relay(self.link, Config(), announce=self.said.append)
        run(self.relay.start(0.0))

    def tick(self, now, still):
        run(self.relay.tick(now, still))

    def finish_unplug(self):
        self.tick(5.0, 0.0)  # moving
        self.tick(12.0, 4.0)  # still 4 s, past min_s

    def test_start_sets_first_task_and_resumes(self):
        self.assertEqual(self.link.calls, [("set_task", UNPLUG.prompts[0]), ("resume",)])

    def test_still_without_moving_is_not_finished(self):
        self.tick(12.0, 12.0)  # still since before the prompt: Dream-1 never started
        self.assertEqual(self.relay.step.name, "unplug")

    def test_still_before_min_s_is_not_finished(self):
        self.tick(2.0, 0.0)
        self.tick(7.0, 5.0)
        self.assertEqual(self.relay.step.name, "unplug")

    def test_moved_then_settled_advances_to_stack(self):
        self.finish_unplug()
        self.assertEqual(self.relay.step.name, "stack")
        self.assertEqual(self.link.calls[-2:], [("set_task", STACK.prompts[0]), ("resume",)])

    def test_stack_settled_finishes_pauses_and_says_call_stop(self):
        self.finish_unplug()
        self.tick(20.0, 0.0)
        self.tick(35.0, 4.0)  # 23 s on the stack step, past min_s
        self.assertTrue(self.relay.done)
        self.assertEqual(self.link.calls[-1], ("pause",))
        self.assertIn("CALL STOP", self.said[-1])
        self.assertNotIn(("to_teleop",), self.link.calls)

    def test_timeout_retries_with_next_prompt_never_teleop(self):
        self.tick(60.0, 0.0)
        self.assertEqual(self.link.calls[-3:], [("pause",), ("set_task", UNPLUG.prompts[1]), ("resume",)])
        self.tick(120.0, 0.0)
        self.assertEqual(self.link.calls[-2], ("set_task", UNPLUG.prompts[2]))
        self.assertNotIn(("to_teleop",), self.link.calls)

    def test_practice_mode_hands_to_teleop_on_timeout(self):
        link = FakeLink()
        relay = Relay(link, Config(practice_teleop=True), announce=lambda _: None)
        run(relay.start(0.0))
        run(relay.tick(60.0, 0.0))
        self.assertEqual(link.calls[-1], ("to_teleop",))

    def test_dry_run_never_calls_the_link(self):
        link = FakeLink()
        relay = Relay(link, Config(dry_run=True), announce=lambda _: None)
        run(relay.start(0.0))
        run(relay.tick(5.0, 0.0))
        run(relay.tick(12.0, 4.0))
        run(relay.tick(80.0, 0.0))
        self.assertEqual(link.calls, [])

    def test_match_clock_ends_the_run(self):
        self.tick(300.0, 0.0)
        self.assertTrue(self.relay.done)


class ScoreTest(unittest.TestCase):
    def test_score_matches_rubric(self):
        self.assertEqual(score(True, True, 120.0), 10 + 20 + 90)
        self.assertEqual(score(True, False, 120.0), 10)
        self.assertEqual(score(True, True, None), 30)


if __name__ == "__main__":
    unittest.main()
