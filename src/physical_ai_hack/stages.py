"""The Hill Climb relay as data: what Dream-1 is told, and for how long. Edit this file to tune a run.

Rubric (hillclimb_rubric_v2):
- Stage 1 passes with at least one cable that started plugged into the hub unplugged (+10).
- Stage 2 passes with a free-standing stack of at least 3 layers / 6 inches, standing 3 s after the gripper lets go (+20).
- Touching a later stage's object before the current stage passes is a DQ, so a step's prompt never names a later
  stage's objects.

v0 is a timeline: each step's prompt runs for `seconds`, then the next one takes over in the same run (no homing in
between). Set the seconds from your zero-shot tries (`harness try`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    name: str
    stage: int  # rubric stage: 1 unplug, 2 stack
    prompt: str  # what Dream-1 is told
    seconds: float  # how long it gets before the next step's prompt


RELAY: tuple[Step, ...] = (
    Step(name="unplug", stage=1, prompt="unplug the usb cable", seconds=45.0),
    Step(name="stack", stage=2, prompt="stack the blocks on top of each other", seconds=120.0),
)

MATCH_S = 300.0  # the 5-minute match clock


def score(stage1: bool, stage2: bool, stop_at_s: float | None, extra_unplugs: int = 0, extra_blocks: int = 0) -> float:
    """The rubric's points for one run, without balls. The time bonus needs both stages and a stop call."""
    points = 0.0
    if stage1:
        points += 10 + 5 * extra_unplugs
    if stage2:
        points += 20 + 10 * extra_blocks
    if stage1 and stage2 and stop_at_s is not None:
        points += max(0.0, MATCH_S - stop_at_s) / 2
    return points
