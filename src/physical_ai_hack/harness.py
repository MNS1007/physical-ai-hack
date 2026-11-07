"""The v0 harness: drive Dream-1 zero-shot through the station agent (`dreamcli robot up`), one prompt per stage.

    uv run harness status                          # what the station agent is doing; moves nothing
    uv run harness try "stack the blocks" -s 60    # one prompt for 60 s, then stop: zero-shot prompt testing
    uv run harness relay --dry-run                 # checks the station and prints the plan; moves nothing
    uv run harness relay                           # match: Enter at GO, unplug -> stack -> stop -> "CALL STOP"

`try` and `relay` (without --dry-run) MOVE THE ARMS: a hand on the emergency stop. Ctrl+C stops the run; the arms
then hold where they are, under power (MakerMods arms have no brakes; the harness never parks or powers them down).
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time
from collections.abc import Callable, Sequence

from .stages import MATCH_S, RELAY, Step
from .station import DEFAULT_URL, Station, StationError

POLL_S = 1.0


class Log:
    """Prints a line and appends it to data/runs/<time>/log.jsonl."""

    def __init__(self, folder: pathlib.Path | None) -> None:
        self.path = None
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            self.path = folder / "log.jsonl"

    def __call__(self, t: float, event: str, **fields) -> None:
        print(f"[{t:6.1f}s] {event} " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps({"t": round(t, 2), "event": event, **fields}) + "\n")


def run_timeline(station, steps: Sequence[Step], log: Callable[..., None], *, checkpoint: str | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 limit_s: float = MATCH_S) -> float:
    """Runs each step's prompt for its seconds in one run, then stops. Returns the seconds from start to stop.

    The first step starts the run; each later one only changes the prompt (no stop, no homing in between).
    A run that ends on its own (error, someone pressed Stop) ends the timeline early.
    """
    t0 = clock()
    try:
        for i, step in enumerate(steps):
            if i == 0:
                station.start(step.prompt, checkpoint)
            else:
                station.prompt(step.prompt)
            log(clock() - t0, "step", name=step.name, prompt=step.prompt, seconds=step.seconds)
            end = min(clock() + step.seconds, t0 + limit_s)
            while clock() < end:
                sleep(POLL_S)
                status = station.status()
                if status.get("state") not in ("starting", "running"):
                    log(clock() - t0, "run ended", state=status.get("state"), error=status.get("last_error"))
                    return clock() - t0
                log(clock() - t0, "tick", state=status["state"], chunks=status.get("chunks"),
                    stalls=status.get("stalls"))
            if clock() - t0 >= limit_s:
                log(clock() - t0, "match clock")
                break
    finally:
        station.stop()  # arms hold their last command, under power
    elapsed = clock() - t0
    log(elapsed, "stopped")
    return elapsed


def check(station: Station) -> bool:
    """Prints whether the station is ready for a run: agent idle, both arms answering, cameras live."""
    status = station.status()
    arms = status.get("arms") or {}
    cameras = status.get("cameras") or {}
    print(f"state={status.get('state')} checkpoint={status.get('checkpoint')} prompt={status.get('prompt')!r}")
    print(f"arms={arms} camera_age_s={cameras}")
    problems = []
    if status.get("state") not in ("idle", "error"):
        problems.append(f"station is {status.get('state')}, not idle")
    problems += [f"arm {name} not answering" for name, ok in arms.items() if not ok]
    problems += [f"camera {name} has no recent frame" for name, age in cameras.items() if age is None or age > 1.0]
    for p in problems:
        print(f"  ✗ {p}")
    return not problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="the station agent's API (default: %(default)s)")
    parser.add_argument("--checkpoint", help="model to run, e.g. dream-1-ft:...; default: the station's")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print the station's status")
    t = sub.add_parser("try", help="run one prompt, then stop (MOVES THE ARMS)")
    t.add_argument("prompt")
    t.add_argument("-s", "--seconds", type=float, default=60.0)
    r = sub.add_parser("relay", help="the match: unplug, stack, stop (MOVES THE ARMS)")
    r.add_argument("--dry-run", action="store_true", help="check the station and print the plan; move nothing")
    args = parser.parse_args(argv)

    station = Station(args.url)
    try:
        if args.command == "status":
            print(json.dumps(station.status(), indent=2))
            return 0
        if not check(station):
            return 1
        folder = pathlib.Path("data/runs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.command == "try":
            steps = [Step(name="try", stage=0, prompt=args.prompt, seconds=args.seconds)]
        else:
            steps = list(RELAY)
            print("plan: " + " -> ".join(f"{s.name} {s.seconds:.0f}s {s.prompt!r}" for s in steps) + " -> stop")
            if args.dry_run:
                return 0
            input("Arms at rest, props placed, hand on the e-stop. Press Enter at GO ... ")
        elapsed = run_timeline(station, steps, Log(folder), checkpoint=args.checkpoint)
        if args.command == "relay":
            print(f"\n>>> STACK DONE at {elapsed:.0f}s: CHECK IT STANDS 3 s, THEN CALL STOP <<<")
        print(f"log: {folder}/log.jsonl")
        return 0
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping the run (arms hold, under power)")
        try:
            station.stop()
        except StationError as error:
            print(f"stop failed: {error}. Press Stop in Live View or Ctrl+C in the agent's terminal.")
        return 130
    except StationError as error:
        print(f"station: {error}", file=sys.stderr)
        return 1


def run() -> None:
    sys.exit(main())
