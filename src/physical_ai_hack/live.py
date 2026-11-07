"""The harness on the robot's LiveKit room: a Portal observer that drives `uv run dream-1-tasks` over its RPCs.

    uv run --group live harness --dry-run    # judge and log only; Dream-1 is never told anything
    uv run --group live harness              # match mode: starts the clock and drives the relay
    uv run --group live harness --practice   # on stuck, hand the arms to the teleoperator (HG-DAgger data)

Start the portal-makermods peers first (robot, then dream-1-tasks); see README.md. Like the task monitor, this is
an observer: it can never send actions, only call Dream-1's RPCs, so the arms move only under Dream-1.
Linux only: livekit-portal has no wheel for macOS before 26.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import pathlib
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from .judge import Judge
from .relay import Config, JsonlLog, Relay

IDENTITY = "harness"
DREAM_TASKS_IDENTITY = "policy-dream-1-tasks"  # portal-makermods src/utils/common.py
RPC = {"status": "dream.status", "set_task": "dream.set_task", "pause": "dream.pause", "resume": "dream.resume",
       "to_teleop": "dream.to_teleop"}
TICK_HZ = 4
MOVED = 0.02  # a state value changing more than this counts as motion (joints are radians)


def mint_token(identity: str, room: str) -> str:
    """As portal-makermods' mint_token: a LiveKit JWT for `room`."""
    from livekit import api
    from livekit.protocol.room import RoomConfiguration

    grants = api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True,
                             can_update_own_metadata=True)
    return (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity(identity).with_grants(grants)
            .with_room_config(RoomConfiguration(name=room, min_playout_delay=0, max_playout_delay=1))
            .with_ttl(datetime.timedelta(hours=6)).to_jwt())


class DreamLink:
    """Dream-1's RPCs, as the relay's link."""

    def __init__(self, obs) -> None:
        self.obs = obs

    async def call(self, name: str, payload: dict | None = None) -> dict:
        answer = await self.obs.perform_rpc(RPC[name], json.dumps(payload or {}), destination=DREAM_TASKS_IDENTITY,
                                            response_timeout_ms=3000)
        return json.loads(answer or "{}")

    async def status(self) -> dict:
        return await self.call("status")

    async def set_task(self, task: str) -> dict:
        return await self.call("set_task", {"task": task})

    async def pause(self) -> dict:
        return await self.call("pause")

    async def resume(self) -> dict:
        return await self.call("resume")

    async def to_teleop(self) -> dict:
        return await self.call("to_teleop")


def rgb(frame) -> np.ndarray:
    from livekit.portal import frame_bytes_to_numpy_rgb
    return frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height).copy()


async def main(args) -> None:
    from livekit.portal import Observer, ObserverConfig

    for path in (pathlib.Path.cwd() / ".env", pathlib.Path(args.portal) / ".env"):
        if path.exists():
            load_dotenv(path)
    room = os.environ["LIVEKIT_ROOM"]
    cfg = ObserverConfig.from_yaml_file(pathlib.Path(args.portal) / "portal.yaml", room)
    cameras = [c for c in args.cameras.split(",") if c]
    missing = [c for c in cameras if c not in cfg.video_tracks]
    if missing:
        raise SystemExit(f"cameras {missing} are not in portal.yaml's videos: {cfg.video_tracks}")

    folder = pathlib.Path("data/harness_runs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = JsonlLog(folder)
    obs = Observer(cfg)
    link = DreamLink(obs)
    relay = Relay(link, Config(practice_teleop=args.practice, dry_run=args.dry_run), log=log)
    judge = Judge()
    print(f"[harness] judge {judge.model} (effort {judge.effort}); cameras {cameras}; logging to {folder}")

    judging: asyncio.Task | None = None
    asked_at, judged_at = 0.0, -1e9
    last_pose, moved_at = None, time.monotonic()
    try:
        await obs.connect(os.environ["LIVEKIT_URL"], mint_token(IDENTITY, room))  # start the robot first
        status = await link.status()
        print(f"[harness] Dream-1: {json.dumps(status)}")
        if not args.dry_run:
            # off the event loop, so the room stays connected while we wait
            await asyncio.to_thread(input, "[harness] Arms at rest, props placed? Press Enter at GO ...")
        await relay.start(time.monotonic())
        while not relay.done:
            now = time.monotonic()
            state = obs.get_state()
            if state is not None:
                pose = np.array([float(v) for v in state.values.values()])
                if last_pose is None or np.abs(pose - last_pose).max() > MOVED:
                    last_pose, moved_at = pose, now
            if judging is not None and judging.done():
                try:
                    verdict, took = judging.result()
                    await relay.on_verdict(verdict, asked_at, took)
                    print(f"[harness] {relay.step.name} {now - relay.started:5.1f}s: met={verdict.goal_met} "
                          f"count={verdict.count} clear={verdict.gripper_clear} ({took:.1f}s) {verdict.reason}")
                except Exception as exc:
                    print(f"[harness] judge failed: {exc!r}")
                judging = None
            if judging is None and now - judged_at >= args.every and not relay.done:
                frames = {c: rgb(f) for c in cameras if (f := obs.get_video_frame(c)) is not None}
                if frames:
                    asked_at = judged_at = now
                    shot = folder / f"{now - relay.started:07.1f}.jpg"
                    cv2.imwrite(str(shot), cv2.cvtColor(next(iter(frames.values())), cv2.COLOR_RGB2BGR))
                    judging = asyncio.ensure_future(judge(relay.step, frames, relay.later_objects()))
            await relay.tick(now, now - moved_at)
            await asyncio.sleep(1 / TICK_HZ)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[harness] stopping; Dream-1 keeps whatever control it has (press c in its terminal to release)")
    finally:
        if judging is not None:
            await asyncio.gather(judging, return_exceptions=True)
        try:
            await obs.disconnect()
        finally:
            obs.close()


def run() -> None:
    parser = argparse.ArgumentParser(description="Hill Climb relay harness around Dream-1")
    parser.add_argument("--portal", default=os.environ.get("PORTAL_DIR", "../portal-makermods"),
                        help="the portal-makermods checkout (for portal.yaml and its .env)")
    parser.add_argument("--cameras", default=os.environ.get("HARNESS_CAMERAS", "overhead_camera"),
                        help="comma-separated portal.yaml video tracks the judge sees")
    parser.add_argument("--every", type=float, default=float(os.environ.get("HARNESS_JUDGE_EVERY_S", "1.5")),
                        help="seconds between judge calls (one call in flight at a time)")
    parser.add_argument("--dry-run", action="store_true", help="judge and log; never call Dream-1's RPCs")
    parser.add_argument("--practice", action="store_true",
                        help="on stuck, hand control to the teleoperator (practice only: a DQ in a match)")
    asyncio.run(main(parser.parse_args()))
