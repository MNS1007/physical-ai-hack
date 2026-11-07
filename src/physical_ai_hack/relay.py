"""The relay's state machine, with no LiveKit in it: it reads arm motion and drives Dream-1 through a `link` with
`set_task`, `resume`, `pause` and `to_teleop` coroutines. `live.py` gives it the real link; tests give it a fake.

Rules it keeps (rubric + Portal):
- After "go" nothing human drives: the harness only sets Dream-1's task, resumes it, and pauses it. `to_teleop` is
  used only with `practice_teleop=True`, for collecting HG-DAgger corrections in practice, never in a match.
- A step is finished when Dream-1 has moved and then stayed still for `settle_s`, no earlier than `min_s`.
- Not finished by `timeout_s`: pause, then resume on the next prompt, which starts a new Dream-1 episode.
- Done: pause Dream-1 so the arms hold, and tell the team to check the stack and call stop. It never parks or powers
  the arms down.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .stages import MATCH_S, RELAY, Step


@dataclass
class Config:
    practice_teleop: bool = False  # on timeout, hand the arms to the teleoperator (practice only: a DQ in a match)
    dry_run: bool = False  # log only; never call the link


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
        self.attempt_at = 0.0  # when the current prompt started
        self.moved = False  # the arms have moved since the current prompt started
        self.finished: dict[str, float] = {}  # step name -> seconds from go
        self.finished_at: float | None = None

    @property
    def step(self) -> Step:
        return self.steps[self.index]

    @property
    def done(self) -> bool:
        return self.finished_at is not None

    def prompt(self) -> str:
        return self.step.prompts[self.attempt % len(self.step.prompts)]

    async def start(self, now: float) -> None:
        """"Go": the match clock starts and Dream-1 gets the first step."""
        self.started = now
        self._emit(now, "go", {})
        await self._drive(now)

    async def tick(self, now: float, still_for_s: float) -> None:
        """Called a few times a second with how long the arms have been still."""
        if self.done or self.started is None:
            return
        if now - self.started >= MATCH_S:
            self._finish(now, "time")
            return
        if still_for_s < now - self.attempt_at:
            self.moved = True  # they moved after this prompt started
        on_step = now - self.attempt_at
        if self.moved and on_step >= self.step.min_s and still_for_s >= self.step.settle_s:
            await self._advance(now)
        elif on_step >= self.step.timeout_s:
            await self._retry(now)

    async def _advance(self, now: float) -> None:
        step = self.step
        self.finished[step.name] = round(now - self.started, 1)
        self._emit(now, "finished_step", {"step": step.name, "attempt": self.attempt})
        if self.index + 1 >= len(self.steps):
            if not self.cfg.dry_run:
                await self.link.pause()  # the arms hold where they are; no park, no power-down
            self._finish(now, "done")
            return
        self.index += 1
        self.attempt = 0
        await self._drive(now)

    async def _retry(self, now: float) -> None:
        self.attempt += 1
        self._emit(now, "retry", {"step": self.step.name, "attempt": self.attempt, "teleop": self.cfg.practice_teleop})
        if self.cfg.practice_teleop:
            self.attempt_at, self.moved = now, False
            if not self.cfg.dry_run:
                await self.link.to_teleop()
            self.announce(f"TIMEOUT on {self.step.name}: teleoperator has the arms; correct, then resume")
            return
        if not self.cfg.dry_run:
            await self.link.pause()
        await self._drive(now)

    async def _drive(self, now: float) -> None:
        """Set Dream-1's task and resume it: each resume is a new Dream-1 episode."""
        self.attempt_at, self.moved = now, False
        self._emit(now, "task", {"step": self.step.name, "prompt": self.prompt()})
        if self.cfg.dry_run:
            return
        await self.link.set_task(self.prompt())
        await self.link.resume()

    def _finish(self, now: float, why: str) -> None:
        self.finished_at = now
        t = now - self.started
        self._emit(now, "finished", {"why": why, "t_s": round(t, 1), "steps": self.finished})
        if why == "done":
            self.announce("\a\n" + "=" * 60 + f"\n   STACK DONE at {t:.0f}s: once it has stood 3 s, CALL STOP\n"
                          + "=" * 60)
        else:
            self.announce(f"time is up at {t:.0f}s; finished steps: {json.dumps(self.finished)}")

    def _emit(self, t: float, kind: str, data: dict) -> None:
        if self.started is not None:
            data = {"elapsed_s": round(t - self.started, 1), **data}
        self.log(Event(t, kind, data))


class JsonlLog:
    """Events as JSON lines in `folder/log.jsonl`, and printed."""

    def __init__(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.file = (folder / "log.jsonl").open("a")

    def __call__(self, event: Event) -> None:
        line = json.dumps({"kind": event.kind, **event.data})
        self.file.write(line + "\n")
        self.file.flush()
        print(f"[harness] {line}")
