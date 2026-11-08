"""The supervisor: runs Dream-1 one attempt at a time, decides when an attempt is over, asks the judge, retries.

Dream-1 never says it is done, so an attempt is over when the arms have moved and then stayed still for the step's
`settle_s` (after `min_s`), or at its `timeout_s`. Then the run is stopped (the arms hold where they are) and the
judge looks at the cameras.

- `attempt()`: one try at one step.
- `match()`: the rubric relay. Supervised (default): retry a step until the judge passes it, within `max_attempts`
  and `budget_s`; never start the stack unless the unplug passed (touching the cubes first is a DQ). Trusting
  (`trust=True`): one go per step, in one run, a step ends when the arms settle; the judge is not asked.
- `evaluate()`: per-task rollouts; the operator resets the scene between trials.

Everything outside (station, judge, clock, sleep, announcer, operator prompt) is passed in, so tests fake it.
"""
from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

from .judge import Judge, Verdict
from .stages import EXTRA_BLOCK_MIN_LEFT_S, MATCH_S, RELAY, Step

POLL_S = 0.25
MOVED_RAD = 0.02  # a joint (or the 0-1 gripper) changing more than this counts as motion


@dataclasses.dataclass
class Attempt:
    step: str
    number: int
    ended: str  # "settled", "timeout", "run ended", "deadline"
    seconds: float
    verdict: Verdict | None = None


def _pose(arms: dict) -> list[float]:
    return [v for name in sorted(arms) for v in (*arms[name]["positions_rad"], arms[name]["gripper"])]


class Supervisor:
    def __init__(self, station, judge: Judge | None, *, checkpoint: str | None = None,
                 log: Callable[..., None] = lambda *a, **k: None, announce: Callable[..., None] = print,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 cameras: list[str] | None = None) -> None:
        self.station, self.judge, self.checkpoint = station, judge, checkpoint
        self.log, self.announce, self.clock, self.sleep = log, announce, clock, sleep
        self.cameras = cameras
        self.running = False

    # --- one attempt -------------------------------------------------------------------------------------------

    def attempt(self, step: Step, number: int, deadline: float, *, keep_running: bool = False) -> Attempt:
        """Gives Dream-1 the step's prompt and waits until the attempt is over; stops the run unless keep_running."""
        if self.running:
            self.station.prompt(step.prompt)
        else:
            self.station.start(step.prompt, self.checkpoint)
            self.running = True
        t0 = self.clock()
        self.log("attempt", step=step.name, number=number, prompt=step.prompt)
        last, moved, still_since, ended = None, False, t0, "timeout"
        while True:
            self.sleep(POLL_S)
            now = self.clock()
            state = self.station.status().get("state")
            if state not in ("starting", "running"):
                ended = "run ended"
                self.running = False
                break
            pose = _pose(self.station.arms())
            if last is not None and max(abs(a - b) for a, b in zip(pose, last)) > MOVED_RAD:
                moved, still_since = True, now
            last = pose
            if now >= deadline:
                ended = "deadline"
                break
            if now - t0 >= step.timeout_s:
                break
            if moved and now - t0 >= step.min_s and now - still_since >= step.settle_s:
                ended = "settled"
                break
        if not keep_running:
            self.stop()
        result = Attempt(step.name, number, ended, self.clock() - t0)
        self.log("attempt over", step=step.name, number=number, ended=ended, seconds=round(result.seconds, 1))
        return result

    def stop(self) -> None:
        if self.running:
            self.station.stop()  # the arms hold their last command, under power
            self.running = False

    def verify(self, step: Step) -> Verdict:
        if self.judge is None:
            return Verdict(False, reason="no judge", ok=False)
        names = self.cameras or self.station.cameras()
        images = {name: self.station.frame(name) for name in names}
        verdict = self.judge.check(step, images)
        self.log("verdict", step=step.name, done=verdict.done, ok=verdict.ok, confidence=verdict.confidence,
                 layers=verdict.layers, can_add=verdict.can_add, reason=verdict.reason,
                 seconds=round(verdict.seconds, 1), images=images)
        return verdict

    # --- the match ---------------------------------------------------------------------------------------------

    def match(self, steps: tuple[Step, ...] = RELAY, *, trust: bool = False, extra_block: bool = False) -> dict:
        """Runs the relay from GO; returns what passed and when it stopped."""
        t0 = self.clock()
        end = t0 + MATCH_S
        passed: dict[str, bool] = {}
        try:
            for step in steps:
                if trust:
                    a = self.attempt(step, 1, end, keep_running=True)
                    passed[step.name] = a.ended in ("settled", "timeout")
                    self.announce(f"Task {step.stage} {step.name} done", step=step.name, ended=a.ended)
                    if a.ended in ("run ended", "deadline"):
                        break
                    continue
                passed[step.name] = self._supervised_step(step, t0, end, extra_block)
                if not passed[step.name]:
                    if step.stage == 1:
                        self.announce("Unplug not verified. Not starting the cubes: touching them first is a DQ.",
                                      step=step.name)
                    break
        finally:
            self.stop()
        elapsed = self.clock() - t0
        all_passed = len(passed) == len(steps) and all(passed.values())
        self.announce(f"All tasks done at {elapsed:.0f} seconds. Call stop." if all_passed
                      else f"Run over at {elapsed:.0f} seconds; passed: {[k for k, v in passed.items() if v]}",
                      passed=passed, seconds=round(elapsed, 1))
        self.log("match over", passed=passed, seconds=round(elapsed, 1), trust=trust)
        return {"passed": passed, "seconds": elapsed}

    def _supervised_step(self, step: Step, t0: float, end: float, extra_block: bool) -> bool:
        step_end = min(end, self.clock() + step.budget_s)
        self.announce(f"Task {step.stage}: {step.name}", step=step.name)
        for number in range(1, step.max_attempts + 1):
            if self.clock() >= step_end:
                break
            a = self.attempt(step, number, step_end)
            a.verdict = self.verify(step)
            if a.verdict.done:
                self.announce(f"Task {step.stage} {step.name} done", step=step.name, attempt=number,
                              reason=a.verdict.reason)
                if extra_block and step.stage == 2 and a.verdict.can_add \
                        and end - self.clock() >= EXTRA_BLOCK_MIN_LEFT_S:
                    self._extra_block(step, a.verdict, end)
                return True
            if a.verdict.give_up:
                self.announce(f"{step.name} failed: moving on", step=step.name, attempt=number)
                break
            self.announce(f"{step.name} not done yet: retrying" if number < step.max_attempts
                          else f"{step.name} not done after {number} tries", step=step.name, attempt=number,
                          reason=a.verdict.reason)
        return False

    def _extra_block(self, step: Step, before: Verdict, end: float) -> None:
        self.announce("Adding one more cube", step=step.name)
        self.attempt(step, 99, end)
        after = self.verify(step)
        self.announce(f"Stack is {after.layers} high" if after.done
                      else "Extra cube did not work; check the stack", step=step.name, layers=after.layers,
                      before=before.layers)

    # --- per-task rollouts -------------------------------------------------------------------------------------

    def evaluate(self, step: Step, trials: int, ready: Callable[[int], None], *, judge: bool = True) -> list[dict]:
        """Runs `trials` single attempts at one step; `ready(i)` waits for the operator to reset the scene."""
        results = []
        for i in range(1, trials + 1):
            ready(i)
            a = self.attempt(step, i, self.clock() + step.timeout_s + 1)
            verdict = self.verify(step) if judge else None
            row = {"trial": i, "ended": a.ended, "seconds": round(a.seconds, 1),
                   "success": None if verdict is None else verdict.done,
                   "reason": "" if verdict is None else verdict.reason}
            results.append(row)
            self.announce(f"Trial {i}: {'success' if row['success'] else 'fail' if verdict else a.ended}", **row)
        return results
