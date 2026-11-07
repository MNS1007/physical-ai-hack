"""The timeline with a fake station and a fake clock: no robot, no network.

    uv run python -m unittest discover tests
"""
import unittest

from physical_ai_hack.harness import run_timeline
from physical_ai_hack.stages import RELAY, Step, score


class FakeStation:
    def __init__(self, end_at_tick=None):
        self.calls, self.ticks, self.end_at_tick = [], 0, end_at_tick

    def start(self, prompt, checkpoint=None):
        self.calls.append(("start", prompt, checkpoint))

    def prompt(self, prompt):
        self.calls.append(("prompt", prompt))

    def stop(self):
        self.calls.append(("stop",))

    def status(self):
        self.ticks += 1
        return {"state": "error" if self.ticks == self.end_at_tick else "running", "chunks": self.ticks}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def go(station, steps, **kw):
    clock = Clock()
    elapsed = run_timeline(station, steps, lambda *a, **k: None, clock=clock, sleep=clock.sleep, **kw)
    return elapsed


class TimelineTest(unittest.TestCase):
    def test_relay_starts_once_switches_prompt_then_stops(self):
        station = FakeStation()
        elapsed = go(station, RELAY, checkpoint="dream-1-ft:x")
        self.assertEqual(station.calls, [("start", RELAY[0].prompt, "dream-1-ft:x"), ("prompt", RELAY[1].prompt),
                                         ("stop",)])
        self.assertAlmostEqual(elapsed, sum(s.seconds for s in RELAY))

    def test_run_ending_on_its_own_ends_early_and_still_stops(self):
        station = FakeStation(end_at_tick=3)
        elapsed = go(station, RELAY)
        self.assertEqual(station.calls[-1], ("stop",))
        self.assertNotIn(("prompt", RELAY[1].prompt), station.calls)
        self.assertEqual(elapsed, 3.0)

    def test_match_clock_caps_the_run(self):
        station = FakeStation()
        elapsed = go(station, [Step("a", 1, "x", 1000.0), Step("b", 2, "y", 10.0)], limit_s=300.0)
        self.assertEqual(elapsed, 300.0)
        self.assertEqual(station.calls[-1], ("stop",))

    def test_prompts_never_name_a_later_stage(self):
        later = {"block", "stack", "ball"}
        self.assertFalse(any(word in RELAY[0].prompt for word in later))


class ScoreTest(unittest.TestCase):
    def test_score_matches_rubric(self):
        self.assertEqual(score(True, True, 120.0), 10 + 20 + 90)
        self.assertEqual(score(True, False, 120.0), 10)
        self.assertEqual(score(True, True, None), 30)


if __name__ == "__main__":
    unittest.main()
