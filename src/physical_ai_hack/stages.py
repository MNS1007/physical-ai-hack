"""The Hill Climb relay as data: what Dream-1 is told, how long it gets, and what the judge checks. Tune runs here.

Rubric (hillclimb_rubric_v2):
- Stage 1 passes with at least one cable that started plugged into the strip unplugged (+10).
- Stage 2 passes with a free-standing stack of at least 3 layers / 6 inches, standing 3 s after the gripper lets go (+20).
- Touching a later stage's object before the current stage passes is a DQ: the supervised match never starts the
  stack before the judge has seen the cable out, and a step's prompt never names a later stage's objects.
- Time bonus: +1 per 2 s left when both stages pass and a teammate calls stop.

Prompts are the training labels word for word ("unplug the usb cable", "stack the cubes").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    name: str
    stage: int  # rubric stage: 1 unplug, 2 stack
    prompt: str  # what Dream-1 is told (the training label)
    seconds: float  # timeline mode (v0 `relay`): how long it gets before the next prompt
    question: str = ""  # what the judge is asked about the scene after an attempt
    min_s: float = 5.0  # an attempt is never over before this
    settle_s: float = 3.0  # arms still this long, after moving: the attempt is over
    timeout_s: float = 60.0  # the attempt is over after this, moving or not
    max_attempts: int = 3  # supervised match: tries before giving up on the step
    budget_s: float = 150.0  # supervised match: seconds this step may use in all


UNPLUG = Step(
    name="unplug", stage=1, prompt="unplug the usb cable", seconds=45.0,
    question=("Look at the power strip. Has at least one USB cable that was plugged into the power strip been fully "
              "pulled out, so its plug is no longer in any socket? A plug only partly out, or still held in the "
              "socket, does not count."),
    min_s=5.0, settle_s=3.0, timeout_s=45.0, max_attempts=3, budget_s=120.0,
)
STACK = Step(
    name="stack", stage=2, prompt="stack the cubes", seconds=120.0,
    question=("Look at the cubes. Is there a stack of at least 3 cubes, one on top of another, standing on its own "
              "with no gripper touching or holding it? Count the layers of the tallest stack. Also say whether a "
              "cube is still lying free on the table and the stack looks stable enough to take one more."),
    min_s=10.0, settle_s=4.0, timeout_s=90.0, max_attempts=3, budget_s=170.0,
)

RELAY: tuple[Step, ...] = (UNPLUG, STACK)
STEPS = {s.name: s for s in RELAY}

MATCH_S = 300.0  # the 5-minute match clock
EXTRA_BLOCK_MIN_LEFT_S = 60.0  # supervised + --extra-block: only try another cube with this much time left


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
