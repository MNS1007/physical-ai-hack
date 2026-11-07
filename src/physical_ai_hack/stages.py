"""The Hill Climb relay as data: what Dream-1 is told, and what the judge must see before the harness moves on.

Edit this file to tune the run: prompt wording, hold times, timeouts. Rubric (hillclimb_rubric_v2):
- Stage 1 passes with at least one cable that started plugged into the hub unplugged (+10).
- Stage 2 passes with a free-standing stack of at least 3 layers / 6 inches, standing 3 s after the gripper lets go (+20).
- Touching a later stage's object before the current stage passes is a DQ, so a step's prompts never name
  a later stage's objects, and a step only ends on a judge verdict held for `hold_s`.
"""
from __future__ import annotations

from dataclasses import dataclass

STACK_LAYERS = 3  # the rubric's pass bar


@dataclass(frozen=True)
class Step:
    name: str
    stage: int  # rubric stage: 1 unplug, 2 stack
    prompts: tuple[str, ...]  # Dream-1 instructions; each retry moves to the next one
    goal: str  # what the judge must see, in words
    count_means: str  # what the judge's `count` field counts for this step
    target: int  # the step is met when count >= target (and goal_met, and for stacks, gripper_clear)
    hold_s: float  # the met verdicts must span this long before the harness moves on
    timeout_s: float  # no progress for this long: pause and retry with the next prompt
    need_gripper_clear: bool = False

    def met(self, verdict) -> bool:
        if not verdict.goal_met or verdict.count < self.target:
            return False
        return verdict.gripper_clear or not self.need_gripper_clear


RELAY: tuple[Step, ...] = (
    Step(
        name="unplug",
        stage=1,
        prompts=(
            "pull the USB cable out of the hub",
            "grab the USB plug at the hub and pull it straight out",
            "unplug the cable from the hub",
        ),
        goal=("At least one cable that was plugged into the hub at the start is now completely out: its plug is no "
              "longer inside any hub port. A cable lying loose on the table from the start does not count."),
        count_means="cables that started in the hub and are now fully unplugged",
        target=1,
        hold_s=2.0,
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
        goal=(f"A free-standing stack of blocks at least {STACK_LAYERS} layers tall stands on the table, and the "
              "gripper has let go of it and is not touching it."),
        count_means="layers in the tallest standing stack of blocks (one block flat on the table is 1 layer)",
        target=STACK_LAYERS,
        hold_s=3.0,  # the rubric's 3 seconds after the gripper lets go
        timeout_s=75.0,
        need_gripper_clear=True,
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
