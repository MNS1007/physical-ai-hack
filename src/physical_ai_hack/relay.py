"""The relay's state machine, with no LiveKit in it: it reads verdicts and arm motion, and drives Dream-1 through a
`link` with `set_task`, `resume`, `pause` and `to_teleop` coroutines. `live.py` gives it the real link; tests give it
a fake.

Rules it keeps (rubric + Portal):
- After "go" nothing human drives: the harness only sets Dream-1's task, resumes it, and pauses it. `to_teleop` is
  used only with `practice_teleop=True`, for collecting HG-DAgger corrections in practice, never in a match.
- A step ends only on `Step.met` verdicts spanning `hold_s` (at least two of them), so one wrong verdict cannot
  advance it and a later stage's objects are never named early.
- Stuck (no progress for `timeout_s`, or arms still for `still_s`): pause, then resume on the next prompt, which
  starts a new Dream-1 episode. Retries never end before the match clock: without Stage 2 there is nothing to keep.
- Done: pause Dream-1 so the arms hold, and tell the team to call stop. It never parks or powers the arms down.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .stages import MATCH_S, RELAY, Step, score


@dataclass
class Config:
    still_s: float = 15.0  # arms still this long, with the step not met: stuck
    practice_teleop: bool = False  # on stuck, hand the arms to the teleoperator (practice only: a DQ in a match)
    dry_run: bool = False  # judge and log, but never call the link


@dataclass
class Event:
    t: float
    kind: str
    data: dict = field(default_factory=dict)


class Relay:
    def __init__(self, link, cfg: Config | None = None, steps: tuple[Step, ...] = RELAY,
                 log: Callable[[Event], None] | None = None, announce: Callable[[str], None] = print) -> None:
        self.link, self.cfg, self.steps = link, cfg or Config(), steps
        self.log, self.announce = log or (lambda event: None), announce
        self.started: float | None = None
        self.index = 0
        self.attempt = 0
        self.best = 0
        self.progress_at = 0.0
        self.met_since: float | None = None
        self.met_count = 0
        self.passed: dict[int, float] = {}  # stage -> seconds from go when it passed
        self.finished_at: float | None = None
        self.dq_warnings = 0

    @property
    def step(self) -> Step:
        return self.steps[self.index]

    @property
    def done(self) -> bool:
        return self.finished_at is not None

    def later_objects(self) -> str:
        """Words for the judge: objects of steps after the current one (USB hub objects never count as later)."""
        names = {"stack": "the stacking blocks", "balls": "the ping-pong balls, the black 8-ball and the bins"}
        later = [names[s.name] for s in self.steps[self.index + 1:] if s.name in names]
        return ", ".join(later + ([names["balls"]] if self.step.name != "balls" else []))

    def prompt(self) -> str:
        return self.step.prompts[self.attempt % len(self.step.prompts)]

    async def start(self, now: float) -> None:
        """"Go": the match clock starts and Dream-1 gets the first step."""
        self.started, self.progress_at = now, now
        self._emit(now, "go", {"step": self.step.name, "prompt": self.prompt()})
        await self._drive(self.prompt())

    async def on_verdict(self, verdict, asked_at: float, took_s: float) -> None:
        """A judge verdict on frames captured at `asked_at`."""
        if self.done or self.started is None:
            return
        step = self.step
        self._emit(asked_at, "verdict", {"step": step.name, "took_s": round(took_s, 2), **verdict.model_dump()})
        if verdict.later_stage_contact:
            self.dq_warnings += 1
            self._emit(asked_at, "dq_risk", {"step": step.name, "reason": verdict.reason})
        if verdict.count > self.best:
            self.best, self.progress_at = verdict.count, asked_at
        if not step.met(verdict):
            self.met_since, self.met_count = None, 0
            return
        if self.met_since is None:
            self.met_since = asked_at
        self.met_count += 1
        if self.met_count >= 2 and asked_at - self.met_since >= step.hold_s:
            await self._advance(asked_at)

    async def tick(self, now: float, still_for_s: float) -> None:
        """Called a few times a second with how long the arms have been still."""
        if self.done or self.started is None:
            return
        elapsed = now - self.started
        if elapsed >= MATCH_S:
            self._finish(now, "time")
            return
        if self.met_since is not None:
            return  # the goal looks met: let the hold run, the arms may just be letting go
        if now - self.progress_at >= self.step.timeout_s:
            await self._retry(now, f"no progress in {now - self.progress_at:.0f}s")
        elif still_for_s >= self.cfg.still_s:
            await self._retry(now, f"arms still for {still_for_s:.0f}s")

    async def _advance(self, now: float) -> None:
        step = self.step
        self.passed[step.stage] = now - self.started
        self._emit(now, "passed", {"step": step.name, "stage": step.stage, "t_s": round(self.passed[step.stage], 1)})
        self.announce(f"PASSED stage {step.stage} ({step.name}) at {self.passed[step.stage]:.0f}s")
        if self.index + 1 >= len(self.steps):
            if not self.cfg.dry_run:
                await self.link.pause()  # the arms hold where they are; no park, no power-down
            self._finish(now, "done")
            return
        self.index += 1
        self.attempt, self.best, self.progress_at = 0, 0, now
        self.met_since, self.met_count = None, 0
        await self._drive(self.prompt())

    async def _retry(self, now: float, why: str) -> None:
        self.attempt += 1
        self.progress_at, self.met_since, self.met_count = now, None, 0
        self._emit(now, "retry", {"step": self.step.name, "attempt": self.attempt, "why": why,
                                  "prompt": self.prompt(), "teleop": self.cfg.practice_teleop})
        if self.cfg.dry_run:
            return
        if self.cfg.practice_teleop:
            await self.link.to_teleop()
            self.announce(f"STUCK on {self.step.name} ({why}): teleoperator has the arms; correct, then resume")
            return
        await self.link.pause()
        await self._drive(self.prompt())

    async def _drive(self, prompt: str) -> None:
        """Set Dream-1's task and resume it: each resume is a new Dream-1 episode."""
        self._emit(time.monotonic(), "task", {"step": self.step.name, "prompt": prompt})
        if self.cfg.dry_run:
            return
        await self.link.set_task(prompt)
        await self.link.resume()

    def _finish(self, now: float, why: str) -> None:
        self.finished_at = now
        t = now - self.started
        both = 1 in self.passed and 2 in self.passed
        points = score(1 in self.passed, 2 in self.passed, t if both else None)
        summary = {"why": why, "t_s": round(t, 1), "passed": self.passed, "points_if_stop_now": points,
                   "dq_warnings": self.dq_warnings}
        self._emit(now, "finished", summary)
        if both:
            self.announce("\a\n" + "=" * 60 + f"\n   CALL STOP NOW   (t={t:.0f}s, about {points:.0f} points)\n"
                          + "=" * 60)
        else:
            self.announce(f"finished ({why}) at {t:.0f}s without both stages: {json.dumps(self.passed)}")

    def _emit(self, t: float, kind: str, data: dict) -> None:
        if self.started is not None:
            data = {"elapsed_s": round(t - self.started, 2), **data}
        self.log(Event(t, kind, data))


class JsonlLog:
    """Events as JSON lines in `folder/log.jsonl`; judge frames go beside it as JPEGs."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.file = (folder / "log.jsonl").open("a")

    def __call__(self, event: Event) -> None:
        self.file.write(json.dumps({"kind": event.kind, **event.data}) + "\n")
        self.file.flush()
        if event.kind in ("go", "task", "retry", "passed", "dq_risk", "finished"):
            print(f"[harness] {event.kind}: {json.dumps(event.data)}")
