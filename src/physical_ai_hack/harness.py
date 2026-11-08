"""The harness: drive Dream-1 through the station agent, judge each task with Claude, announce progress.

Start the agent as `station-plus` (Dream's agent plus camera/arm routes; README). Then:

    uv run harness status                                  # what the agent is doing; moves nothing
    uv run --group judge harness judge-now unplug          # Claude's verdict on the scene right now; moves nothing
    uv run --group judge harness judge-image stack a.jpg   # Claude's verdict on saved images (offline)
    uv run harness try "unplug the usb cable" -s 45        # one prompt for 45 s, then stop
    uv run --group judge harness eval unplug --trials 10   # per-task rollouts: success rate and time
    uv run --group judge harness match                     # the rubric relay, supervised: verify, retry, gate
    uv run --group livekit harness match --judge voice     # the operator says done / retry / fail (LiveKit room)
    uv run harness match --trust                           # baseline: no judge, a step ends when the arms settle
    uv run harness relay                                   # v0: fixed timeline
    uv run --group livekit harness listen                  # voice test: prints what you say -> command; no station

`try`, `eval`, `match`, `relay` MOVE THE ARMS: a hand on the emergency stop. Ctrl+C stops the run; the arms then
hold where they are, under power. Add `--voice` for spoken announcements, `--livekit` to speak them in a LiveKit
room. `--checkpoint dream-1-ft:...` runs a fine-tune.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import sys
import time
from collections.abc import Callable, Sequence

from .stages import MATCH_S, RELAY, STEPS, Step
from .station import DEFAULT_URL, Station, StationError

POLL_S = 1.0


class Log:
    """Prints a line and appends it to data/runs/<time>/log.jsonl; images go next to it as files."""

    def __init__(self, folder: pathlib.Path | None) -> None:
        self.folder, self.path, self.t0, self.count = folder, None, time.monotonic(), 0
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            self.path = folder / "log.jsonl"

    def __call__(self, *args, **fields) -> None:
        # timeline mode passes (t, event); the supervisor passes (event,)
        t, event = (args[0], args[1]) if len(args) == 2 else (time.monotonic() - self.t0, args[0])
        images = fields.pop("images", None)
        if images and self.folder is not None:
            self.count += 1
            saved = []
            for name, jpeg in images.items():
                path = self.folder / f"{self.count:03d}-{fields.get('step', 'x')}-{name}.jpg"
                path.write_bytes(jpeg)
                saved.append(path.name)
            fields["images"] = saved
        print(f"[{t:6.1f}s] {event} " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps({"t": round(t, 2), "event": event, **fields}, default=str) + "\n")


def run_timeline(station, steps: Sequence[Step], log: Callable[..., None], *, checkpoint: str | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 limit_s: float = MATCH_S) -> float:
    """v0: each step's prompt for its seconds in one run, then stop. Returns the seconds from start to stop."""
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


def _judge(model: str):
    from dotenv import load_dotenv

    from .judge import ClaudeJudge

    load_dotenv()
    return ClaudeJudge(model=model)


def listen() -> int:
    """Joins the LiveKit room and prints each thing said with the command it maps to. Moves nothing."""
    from .announce import Announcer
    from .judge import is_start, parse_word

    announce = Announcer(livekit=True)
    try:
        announce.livekit.load_whisper()
        announce("Voice test. Say start, done, next, retry, or fail.")
        print("listening; Ctrl+C to end")
        while True:
            heard = announce.livekit.heard(timeout=5.0)
            if heard is None:
                continue
            word = parse_word(heard) or ("start" if is_start(heard) else None)
            print(f"heard {heard!r} -> {word or '(no command)'}", flush=True)
            announce(f"Heard {word}" if word else "Not a command")
    except KeyboardInterrupt:
        return 0
    finally:
        announce.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="the station agent's API (default: %(default)s)")
    parser.add_argument("--checkpoint", help="model to run, e.g. dream-1-ft:...; default: the station's")
    parser.add_argument("--judge", choices=("claude", "voice", "keys"), default="claude",
                        help="who decides done/retry: Claude on the cameras, the operator's voice in the LiveKit "
                             "room (needs --livekit), or typed (testing)")
    parser.add_argument("--judge-model", default="claude-opus-5-5", help="e.g. claude-sonnet-5-5 for speed")
    parser.add_argument("--cameras", help="comma-separated cameras the judge sees (default: all)")
    parser.add_argument("--voice", action="store_true", help="speak announcements on this machine")
    parser.add_argument("--livekit", action="store_true", help="speak announcements in the LiveKit room")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print the station's status")
    sub.add_parser("listen", help="voice test: join the LiveKit room, print what you say and the command it means "
                                  "(no station, nothing moves; Ctrl+C to end)")
    j = sub.add_parser("judge-now", help="Claude's verdict on the current scene (moves nothing)")
    j.add_argument("step", choices=sorted(STEPS))
    ji = sub.add_parser("judge-image", help="Claude's verdict on saved images (offline; no station)")
    ji.add_argument("step", choices=sorted(STEPS))
    ji.add_argument("images", nargs="+")
    t = sub.add_parser("try", help="run one prompt, then stop (MOVES THE ARMS)")
    t.add_argument("prompt")
    t.add_argument("-s", "--seconds", type=float, default=60.0)
    e = sub.add_parser("eval", help="per-task rollouts with Claude's verdicts (MOVES THE ARMS)")
    e.add_argument("step", choices=sorted(STEPS))
    e.add_argument("--trials", type=int, default=5)
    e.add_argument("--no-judge", action="store_true", help="time only, no verdicts")
    m = sub.add_parser("match", help="the rubric relay, supervised by Claude (MOVES THE ARMS)")
    m.add_argument("--trust", action="store_true", help="baseline: no judge, no retries")
    m.add_argument("--extra-block", action="store_true", help="try one more cube when the judge says it can")
    m.add_argument("--dry-run", action="store_true", help="check the station and judge; move nothing")
    r = sub.add_parser("relay", help="v0 fixed timeline (MOVES THE ARMS)")
    r.add_argument("--dry-run", action="store_true", help="check the station and print the plan; move nothing")
    args = parser.parse_args(argv)

    if args.command == "judge-image":
        from .judge import Verdict  # noqa: F401

        verdict = _judge(args.judge_model).check(
            STEPS[args.step], {pathlib.Path(p).stem: pathlib.Path(p).read_bytes() for p in args.images})
        print(json.dumps(verdict.__dict__, indent=2))
        return 0 if verdict.ok else 1

    if args.command == "listen":
        return listen()

    station = Station(args.url)
    folder = pathlib.Path("data/runs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cameras = args.cameras.split(",") if args.cameras else None
    announce = None
    try:
        if args.command == "status":
            print(json.dumps(station.status(), indent=2))
            return 0
        if args.command == "judge-now":
            from .supervisor import Supervisor

            sup = Supervisor(station, _judge(args.judge_model), log=Log(folder), cameras=cameras)
            verdict = sup.verify(STEPS[args.step])
            print(json.dumps(verdict.__dict__, indent=2))
            return 0
        if not check(station):
            return 1
        if args.command == "try":
            run_timeline(station, [Step(name="try", stage=0, prompt=args.prompt, seconds=args.seconds)], Log(folder),
                         checkpoint=args.checkpoint)
            print(f"log: {folder}/log.jsonl")
            return 0
        if args.command == "relay":
            print("plan: " + " -> ".join(f"{s.name} {s.seconds:.0f}s {s.prompt!r}" for s in RELAY) + " -> stop")
            if args.dry_run:
                return 0
            input("Arms at rest, props placed, hand on the e-stop. Press Enter at GO ... ")
            elapsed = run_timeline(station, RELAY, Log(folder), checkpoint=args.checkpoint)
            print(f"\n>>> STACK DONE at {elapsed:.0f}s: CHECK IT STANDS 3 s, THEN CALL STOP <<<")
            return 0

        from .announce import Announcer
        from .supervisor import Supervisor

        use_judge = not (args.command == "match" and args.trust) and not (args.command == "eval" and args.no_judge)
        announce = Announcer(voice=args.voice, livekit=args.livekit or args.judge == "voice")
        judge = None
        if use_judge:
            from .judge import KeyboardJudge, VoiceJudge

            judge = {"claude": lambda: _judge(args.judge_model), "voice": lambda: VoiceJudge(announce),
                     "keys": KeyboardJudge}[args.judge]()
        log = Log(folder)
        sup = Supervisor(station, judge, checkpoint=args.checkpoint, log=log, announce=announce, cameras=cameras)

        if args.command == "match":
            mode = "trusting (no judge)" if args.trust else "supervised by " + (
                args.judge_model if args.judge == "claude" else f"the operator ({args.judge})")
            print("plan: " + " -> ".join(f"{s.name} {s.prompt!r} (≤{s.max_attempts} tries, {s.budget_s:.0f}s)"
                                         for s in RELAY) + f" -> stop; {mode}")
            if args.dry_run:
                if judge:
                    print("judge on the current scene:", sup.verify(RELAY[0]))
                return 0
            if args.judge == "voice":
                from .judge import wait_for_start

                wait_for_start(announce)  # "start" / "go" in the room
            else:
                input("Arms at rest, props placed, hand on the e-stop. Press Enter at GO ... ")
            announce("Go")
            sup.match(RELAY, trust=args.trust, extra_block=args.extra_block)
        else:
            step = STEPS[args.step]
            results = sup.evaluate(step, args.trials,
                                   lambda i: input(f"Trial {i}/{args.trials}: reset the scene, arms at rest. Enter ... "),
                                   judge=judge is not None)
            ok = [r for r in results if r["success"]]
            print(f"\n{step.name} with {args.checkpoint or 'the station model'}: "
                  f"{len(ok)}/{len(results)} success" + (f", median {statistics.median(r['seconds'] for r in ok):.1f} s"
                                                       if ok else ""))
            with (folder / "eval.json").open("w") as f:
                json.dump({"step": step.name, "checkpoint": args.checkpoint, "results": results}, f, indent=2)
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
    finally:
        if announce is not None:
            announce.close()


def run() -> None:
    sys.exit(main())
