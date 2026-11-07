"""The Hill Climb relay as data: what Dream-1 is told, and how long to give it.

Edit this file to tune the run. Rubric (hillclimb_rubric_v2):
- Stage 1 passes with at least one cable that started plugged into the hub unplugged (+10).
- Stage 2 passes with a free-standing stack of at least 3 layers / 6 inches, standing 3 s after the gripper lets go (+20).
- Touching a later stage's object before the current stage passes is a DQ, so a step's prompts never name a later
  stage's objects.

There is no camera judge: a step counts as finished when Dream-1 has moved and then stayed still for `settle_s`
(it stops when it thinks it is done). Watch the first runs and set the times from what you see.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    name: str
    stage: int  # rubric stage: 1 unplug, 2 stack
    prompts: tuple[str, ...]  # Dream-1 instructions; each retry moves to the next one
    min_s: float  # never call the step finished before this
    settle_s: float  # arms still this long, after moving: the step is finished
    timeout_s: float  # still not finished after this: pause and retry with the next prompt


RELAY: tuple[Step, ...] = (
    Step(
        name="unplug",
        stage=1,
        prompts=(
            "pull the USB cable out of the hub",
            "grab the USB plug at the hub and pull it straight out",
            "unplug the cable from the hub",
        ),
        min_s=8.0,
        settle_s=4.0,
        timeout_s=60.0,
    ),
    Step(
        name="stack",
        stage=2,
        prompts=(
            "stack the blocks on top of each other",
            "pick up a block and place it on top of the stack",
            "put one block on top of another block",
        ),
        min_s=20.0,
        settle_s=4.0,  # covers the rubric's 3 s free-standing after the gripper lets go
        timeout_s=90.0,
    ),
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
