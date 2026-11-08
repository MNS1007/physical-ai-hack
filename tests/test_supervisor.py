"""The supervisor with a fake station (arms that move for a while, then rest), a scripted judge and a fake clock.

    uv run python -m unittest tests.test_supervisor
"""
import unittest

from physical_ai_hack.judge import Verdict
from physical_ai_hack.stages import RELAY, STACK, UNPLUG
from physical_ai_hack.supervisor import Supervisor


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeStation:
    """Arms move for `move_s` after each start/prompt, then rest."""

    def __init__(self, clock, move_s=8.0, end_run_at=None):
        self.clock, self.move_s, self.end_run_at = clock, move_s, end_run_at
        self.calls, self.state, self.moving_until = [], "idle", 0.0

    def start(self, prompt, checkpoint=None):
        self.calls.append(("start", prompt))
        self.state, self.moving_until = "running", self.clock() + self.move_s

    def prompt(self, prompt):
        self.calls.append(("prompt", prompt))
        self.moving_until = self.clock() + self.move_s

    def stop(self):
        self.calls.append(("stop",))
        self.state = "idle"

    def status(self):
        if self.end_run_at is not None and self.clock() >= self.end_run_at:
            self.state = "error"
        return {"state": self.state}

    def arms(self):
        t = self.clock()
        x = t if t < self.moving_until else self.moving_until
        return {"right": {"positions_rad": [x, 0, 0, 0, 0, 0], "gripper": 0.5}}

    def cameras(self):
        return ["scene"]

    def frame(self, name):
        return b"jpeg"


class ScriptedJudge:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    def check(self, step, images):
        self.asked.append(step.name)
        return self.answers.pop(0) if self.answers else Verdict(False, reason="no more answers")


def make(answers, **kw):
    clock = Clock()
    station = FakeStation(clock, **kw)
    judge = ScriptedJudge(answers)
    said = []
    sup = Supervisor(station, judge, clock=clock, sleep=clock.sleep, announce=lambda text, **e: said.append(text))
    return sup, station, judge, said, clock


YES, NO = Verdict(True, 0.9, "yes", 3, False), Verdict(False, 0.9, "no")


class AttemptTest(unittest.TestCase):
    def test_attempt_ends_when_arms_settle(self):
        sup, station, _, _, clock = make([])
        a = sup.attempt(UNPLUG, 1, deadline=1e9)
        self.assertEqual(a.ended, "settled")
        self.assertAlmostEqual(a.seconds, 8.0 + UNPLUG.settle_s, delta=0.5)
        self.assertEqual(station.calls[-1], ("stop",))

    def test_attempt_times_out_when_arms_keep_moving(self):
        sup, *_ = make([], move_s=1000)
        self.assertEqual(sup.attempt(UNPLUG, 1, deadline=1e9).ended, "timeout")

    def test_run_ending_on_its_own_is_reported(self):
        sup, *_ = make([], end_run_at=3.0)
        self.assertEqual(sup.attempt(UNPLUG, 1, deadline=1e9).ended, "run ended")


class MatchTest(unittest.TestCase):
    def test_both_pass_first_try(self):
        sup, station, judge, said, _ = make([YES, YES])
        result = sup.match(RELAY)
        self.assertEqual(result["passed"], {"unplug": True, "stack": True})
        self.assertEqual(judge.asked, ["unplug", "stack"])
        self.assertIn("Call stop", said[-1])
        self.assertEqual(station.calls[-1], ("stop",))

    def test_retry_then_pass(self):
        sup, station, judge, said, _ = make([NO, YES, YES])
        result = sup.match(RELAY)
        self.assertTrue(all(result["passed"].values()))
        self.assertEqual(judge.asked, ["unplug", "unplug", "stack"])
        self.assertEqual([c for c in station.calls if c[0] == "start"][:2],
                         [("start", UNPLUG.prompt), ("start", UNPLUG.prompt)])

    def test_unplug_never_verified_means_no_stack(self):
        sup, station, judge, said, _ = make([NO, NO, NO])
        result = sup.match(RELAY)
        self.assertEqual(result["passed"], {"unplug": False})
        self.assertNotIn(("start", STACK.prompt), station.calls)
        self.assertTrue(any("DQ" in s for s in said))

    def test_judge_error_is_never_a_pass(self):
        sup, *_ = make([Verdict(False, ok=False, reason="judge error")] * 3)
        self.assertEqual(sup.match(RELAY)["passed"], {"unplug": False})

    def test_trust_mode_no_judge_no_stop_between_steps(self):
        sup, station, judge, said, _ = make([])
        result = sup.match(RELAY, trust=True)
        self.assertEqual(judge.asked, [])
        self.assertEqual(station.calls, [("start", UNPLUG.prompt), ("prompt", STACK.prompt), ("stop",)])
        self.assertTrue(all(result["passed"].values()))

    def test_extra_block_only_when_judge_says_it_can(self):
        sup, station, judge, said, _ = make([YES, Verdict(True, 0.9, "3 high", 3, True), YES])
        sup.match(RELAY, extra_block=True)
        self.assertEqual(judge.asked, ["unplug", "stack", "stack"])
        sup2, _, judge2, _, _ = make([YES, YES])
        sup2.match(RELAY, extra_block=True)
        self.assertEqual(judge2.asked, ["unplug", "stack"])


class EvalTest(unittest.TestCase):
    def test_eval_records_each_trial(self):
        sup, *_ = make([YES, NO])
        rows = sup.evaluate(UNPLUG, 2, ready=lambda i: None)
        self.assertEqual([r["success"] for r in rows], [True, False])


if __name__ == "__main__":
    unittest.main()


class OperatorWordsTest(unittest.TestCase):
    def test_parse_word(self):
        from physical_ai_hack.judge import parse_word

        self.assertEqual(parse_word("Done."), "done")
        self.assertEqual(parse_word("okay retry that"), "retry")
        self.assertEqual(parse_word("fail, move on"), "fail")
        self.assertEqual(parse_word("hmm wait"), None)

    def test_give_up_stops_retrying_and_gates_stack(self):
        sup, station, judge, said, _ = make([Verdict(False, 1.0, "fail", give_up=True)])
        result = sup.match(RELAY)
        self.assertEqual(judge.asked, ["unplug"])
        self.assertEqual(result["passed"], {"unplug": False})


class VoiceWordsTest(unittest.TestCase):
    def test_next_and_start(self):
        from physical_ai_hack.judge import is_start, parse_word

        self.assertEqual(parse_word("go to the next task"), "done")
        self.assertEqual(parse_word("next"), "done")
        self.assertEqual(parse_word("fail, move on"), "fail")
        self.assertTrue(is_start("Start."))
        self.assertTrue(is_start("okay go"))
        self.assertFalse(is_start("retry"))
