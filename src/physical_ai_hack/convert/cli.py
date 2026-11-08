"""`convert inspect` and `convert run`: see README.md, "Converting teleop recordings"."""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from . import episode as ep
from . import mapping as mp
from . import rrd
from .writer import DatasetError, Writer

# The kit's raw MakerMods layout as a Dream embodiment, for the structural checks (before add_eef.py adds poses).
# Built from the mapping's own parts and cameras, so a different mapping is checked against itself.


def _span(times: np.ndarray) -> str:
    if len(times) < 2:
        return f"{len(times)} row(s)"
    s = (times[-1] - times[0]) / 1e9
    return f"{len(times)} rows over {s:.1f} s ({(len(times) - 1) / s:.1f} Hz)" if s > 0 else f"{len(times)} rows"


def inspect(path: str, timeline: str) -> None:
    rec = rrd.read(path, timeline)
    print(f"{path}\n  application: {rec.application_id}   timelines: {sorted(rec.timelines)}")
    if not rec.series and not rec.videos:
        print("  (no scalar series or video on this timeline: empty or aborted recording)")
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    for entity, s in sorted(rec.series.items()):
        print(f"  scalars {entity}: {_span(s.times)}\n    names {s.names}\n    min {s.values.min(0)}\n"
              f"    max {s.values.max(0)}")
    for entity, v in sorted(rec.videos.items()):
        size = ""
        try:
            t, bgr = next(ep._decoded(v))
            size = f", frames {bgr.shape[1]}x{bgr.shape[0]}"
        except Exception as error:  # noqa: BLE001 - inspect reports, never fails
            size = f", cannot decode: {error}"
        print(f"  video {entity}: codec {v.codec}, {_span(v.times)}, {int(v.keyframes.sum())} keyframes{size}")
    for entity, comps in sorted(rec.other.items()):
        print(f"  other {entity}: {sorted(comps)}")


def raw_embodiment(m: mp.Mapping) -> dict:
    spans, start = {}, 0
    for p in m.parts:
        spans[p.name] = [start, start + p.width]
        start += p.width
    roles = {}
    for cam in m.cameras:
        role = cam.key.removeprefix("observation.images.")
        roles[{"left_wrist": "wrist_l", "right_wrist": "wrist_r"}.get(role, role)] = cam.key
    head = next((r for r in roles if r not in ("wrist_l", "wrist_r")), next(iter(roles)))
    return {"state": spans, "action": dict(spans), "video": roles, "head_camera": head}


def validate(out: pathlib.Path, m: mp.Mapping) -> bool:
    try:
        from dream import datasets
    except ImportError:
        print("dream library not installed here: skipped Dream's dataset checks (run them on the station)")
        return True
    report = datasets.validate_dataset(out, raw_embodiment(m))
    print(f"Dream's checks: {'OK' if report.ok else 'FAILED'}: {report.episodes} episodes, {report.frames} frames, "
          f"state {report.state_dim}, action {report.action_dim}, {report.fps} fps" +
          (f"\n  {report.error}" if report.error else ""))
    for note in report.notes:
        print(f"  note: {note}")
    return report.ok


def run_files(files: list[str], out: pathlib.Path, m: mp.Mapping, task: str | None) -> int:
    writer = Writer(out, robot_type=m.robot_type, fps=m.fps, names=m.vector_names,
                    video_keys=[c.key for c in m.cameras], width=m.width, height=m.height)
    converted = 0
    for path in files:
        try:
            rec = rrd.read(path, m.timeline)
            plan = ep.plan(rec, m)
            state, action = ep.vectors(rec, m, plan.ticks)
            stats: dict[str, dict[str, int]] = {c.key: {} for c in m.cameras}
            cameras = {c.key: ep.frames(rec.videos.get(c.entity) if c.entity else None, plan.ticks, m.width,
                                        m.height, stats[c.key]) for c in m.cameras}
            index = writer.add(task or m.task, state, action, cameras)
        except (ep.EpisodeError, DatasetError) as error:
            print(f"SKIP {path}: {error}")
            continue
        converted += 1
        stale = ", ".join(f"{k.removeprefix('observation.images.')} {s['stale']} stale"
                          for k, s in stats.items() if s.get("stale"))
        print(f"episode {index}: {path} -> {len(plan.ticks)} frames ({len(plan.ticks) / m.fps:.1f} s); "
              f"trimmed {plan.head_s:.1f} s head, {plan.tail_s:.1f} s tail"
              + (f"; filled {plan.filled}" if plan.filled else "") + (f"; {stale}" if stale else ""))
    print(f"{converted} of {len(files)} recordings converted into {out}")
    return converted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="convert", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inspect", help="list what a recording holds: entities, names, rates, ranges, cameras")
    i.add_argument("files", nargs="+")
    i.add_argument("--timeline", default="time")
    r = sub.add_parser("run", help="convert recordings (one .rrd = one episode) into a LeRobot v2.1 folder")
    r.add_argument("files", nargs="+", help=".rrd files, then the output folder last")
    r.add_argument("--map", required=True, help="the mapping YAML (configs/convert/...)")
    r.add_argument("--task", help="task text for these episodes (default: the mapping's)")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        for f in args.files:
            inspect(f, args.timeline)
        return 0
    if len(args.files) < 2:
        parser.error("give one or more .rrd files and then the output folder")
    *files, out = args.files
    try:
        m = mp.load(args.map)
    except (mp.MappingError, KeyError, TypeError) as error:
        print(f"mapping {args.map}: {error}", file=sys.stderr)
        return 2
    try:
        converted = run_files(files, pathlib.Path(out).expanduser(), m, args.task)
    except DatasetError as error:
        print(f"output {out}: {error}", file=sys.stderr)
        return 2
    if converted == 0:
        return 1
    return 0 if validate(pathlib.Path(out).expanduser(), m) else 1


def run() -> None:
    sys.exit(main())
