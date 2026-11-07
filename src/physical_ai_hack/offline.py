"""Test the judge on photos, no robot needed: take phone photos of your table and see what Claude says.

    uv run harness-judge --step unplug photos/hub_plugged.jpg photos/hub_one_out.jpg
    uv run harness-judge --step stack photos/two_blocks.jpg photos/three_blocks_gripper_away.jpg

Each image is judged on its own, as the live harness judges one overhead frame. Use it to tune the goal wording in
stages.py and to measure latency before match day.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib

import cv2
from dotenv import load_dotenv

from .judge import Judge
from .stages import RELAY


async def main(args) -> None:
    load_dotenv(pathlib.Path.cwd() / ".env")
    steps = {s.name: s for s in RELAY}
    step = steps[args.step]
    later = ", ".join(f"objects of the {s.name} step" for s in RELAY[RELAY.index(step) + 1:])
    judge = Judge(model=args.model)
    print(f"judge {judge.model} (effort {judge.effort}), step {step.name}: met when count >= {step.target}")
    for path in args.images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"{path}: cannot read")
            continue
        verdict, took = await judge(step, {"photo": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)}, later)
        print(f"{path}: met={step.met(verdict)} goal_met={verdict.goal_met} count={verdict.count} "
              f"gripper_clear={verdict.gripper_clear} later_contact={verdict.later_stage_contact} "
              f"[{took:.1f}s] {verdict.reason}")


def run() -> None:
    parser = argparse.ArgumentParser(description="Judge photos with the harness's judge")
    parser.add_argument("--step", choices=[s.name for s in RELAY], required=True)
    parser.add_argument("--model", default=None, help="override HARNESS_MODEL")
    parser.add_argument("images", nargs="+")
    asyncio.run(main(parser.parse_args()))
