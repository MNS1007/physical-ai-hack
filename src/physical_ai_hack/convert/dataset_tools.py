"""Edits on a finished LeRobot v2.1 dataset: drop camera streams, e.g. a wrist camera the model should not use.

Dream's checks refuse a dataset with more video streams than its embodiment names, so a camera left out of the
embodiment must be left out of the dataset too. Runs on any v2.1 dataset, the kit's `-eef` output included.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil


def drop_cameras(source: pathlib.Path, target: pathlib.Path, drop: list[str]) -> int:
    """Writes `target`: `source` without the video streams `drop` (keys like observation.images.left_wrist).
    Videos are hard-linked where the disk allows; parquet and meta are copied. Returns the episodes kept."""
    info = json.loads((source / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{source} is LeRobot {info.get('codebase_version')}, not v2.1")
    videos = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    unknown = [k for k in drop if k not in videos]
    if unknown:
        raise ValueError(f"{source} has no camera {unknown}; it has {videos}")
    if target.exists():
        raise ValueError(f"{target} exists; pick another folder")

    def copy(src: str, dst: str) -> None:
        if src.endswith(".mp4"):
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        shutil.copy2(src, dst)

    shutil.copytree(source, target, copy_function=copy, ignore=lambda d, names: [
        n for n in names if n in drop or n.startswith(".")])
    for key in drop:
        info["features"].pop(key)
    kept = [k for k in videos if k not in drop]
    info["total_videos"] = int(info["total_episodes"]) * len(kept)
    (target / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    stats = target / "meta" / "episodes_stats.jsonl"
    rows = [json.loads(line) for line in stats.read_text().splitlines() if line.strip()]
    for row in rows:
        for key in drop:
            row["stats"].pop(key, None)
    stats.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return int(info["total_episodes"])
