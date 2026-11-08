"""A LeRobot v2.1 writer laid out exactly as the Dream hackathon kit's recorder writes (its dataset.py), so the kit's
tools/add_eef.py and tools/finetune.py read the result unchanged.

    meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, meta/episodes_stats.jsonl
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/<video key>/episode_000000.mp4      (H.264, yuv420p, a keyframe every 10 frames)

Episodes are appended; an existing folder must match (version, fps, vector width, cameras, frame size).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
from collections.abc import Iterable
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
IMAGE_STATS_EVERY = 10


class DatasetError(RuntimeError):
    pass


def features(names: list[str], video_keys: list[str], width: int, height: int, fps: int) -> dict[str, Any]:
    vector = {"dtype": "float32", "shape": [len(names)], "names": list(names)}
    out: dict[str, Any] = {"action": dict(vector), "observation.state": dict(vector)}
    for key in video_keys:
        out[key] = {
            "dtype": "video", "shape": [height, width, 3], "names": ["height", "width", "channels"],
            "info": {"video.fps": fps, "video.height": height, "video.width": width, "video.channels": 3,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "has_audio": False},
        }
    out["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for column in ("frame_index", "episode_index", "index", "task_index"):
        out[column] = {"dtype": "int64", "shape": [1], "names": None}
    return out


def vector_stats(values: np.ndarray) -> dict[str, list]:
    values = values.astype(np.float64)
    return {"min": values.min(0).tolist(), "max": values.max(0).tolist(), "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(), "count": [len(values)]}


def scalar_stats(values: np.ndarray) -> dict[str, list]:
    return vector_stats(np.asarray(values).reshape(-1, 1))


def encode(path: pathlib.Path, frames: Iterable[np.ndarray], fps: int, width: int, height: int) -> dict[str, Any]:
    """Writes BGR frames as H.264 and returns LeRobot's per-episode image statistics ([3,1,1] per RGB channel)."""
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23", "g": "10"}
    samples = []
    count = 0
    try:
        for bgr in frames:
            frame = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p")
            frame.pts = count
            for packet in stream.encode(frame):
                container.mux(packet)
            if count % IMAGE_STATS_EVERY == 0:
                samples.append(bgr[::4, ::4, ::-1].astype(np.float32) / 255.0)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    pixels = np.concatenate([s.reshape(-1, 3) for s in samples])
    stats: dict[str, Any] = {k: v.reshape(3, 1, 1).tolist() for k, v in
                             {"min": pixels.min(0), "max": pixels.max(0), "mean": pixels.mean(0),
                              "std": pixels.std(0)}.items()}
    stats["count"] = [len(samples)]
    return stats


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


def _append_jsonl(path: pathlib.Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


class Writer:
    def __init__(self, root: pathlib.Path, *, robot_type: str, fps: int, names: list[str], video_keys: list[str],
                 width: int, height: int) -> None:
        self.root, self.fps, self.video_keys, self.size = root, fps, list(video_keys), (width, height)
        self.features = features(names, self.video_keys, width, height, fps)
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            self.info = json.loads(info_path.read_text())
            self._check(self.info)
            tasks = root / "meta" / "tasks.jsonl"
            rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()] if tasks.exists() \
                else []
            self.tasks = {r["task"]: int(r["task_index"]) for r in rows}
        else:
            self.info = {"codebase_version": CODEBASE_VERSION, "robot_type": robot_type, "total_episodes": 0,
                         "total_frames": 0, "total_tasks": 0, "total_videos": 0, "total_chunks": 0,
                         "chunks_size": CHUNKS_SIZE, "fps": fps, "splits": {"train": "0:0"}, "data_path": DATA_PATH,
                         "video_path": VIDEO_PATH, "features": self.features}
            self.tasks = {}

    def _check(self, info: dict[str, Any], where: pathlib.Path | None = None) -> None:
        """Refuses a dataset laid out another way than this writer writes: version, fps, widths, cameras, sizes."""
        where = where or self.root
        if info.get("codebase_version") != CODEBASE_VERSION:
            raise DatasetError(f"{where} is LeRobot {info.get('codebase_version')}, not {CODEBASE_VERSION}")
        if info.get("fps") != self.fps:
            raise DatasetError(f"{where} is at {info.get('fps')} fps; the mapping writes {self.fps}")
        for key, feature in self.features.items():
            if info["features"].get(key, {}).get("shape") != feature["shape"]:
                raise DatasetError(f"{where}: {key} is shaped {info['features'].get(key, {}).get('shape')}; "
                                   f"the mapping writes {feature['shape']}")
        recorded = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
        if recorded != sorted(self.video_keys):
            raise DatasetError(f"{where} has cameras {recorded}; the mapping writes {sorted(self.video_keys)}")

    def add(self, task: str, state: np.ndarray, action: np.ndarray,
            cameras: dict[str, Iterable[np.ndarray]]) -> int:
        """Encodes the cameras and writes one episode; returns its index. Nothing lands in the dataset on failure."""
        scratch = self._scratch()
        try:
            stats = {key: encode(scratch / f"{key}.mp4", cameras[key], self.fps, *self.size)
                     for key in self.video_keys}
            frame_index = np.arange(len(state), dtype=np.int64)
            timestamps = (frame_index / self.fps).astype(np.float32)
            table = pa.table({
                "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1)), state.shape[1]),
                "action": pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1)), action.shape[1]),
                "timestamp": pa.array(timestamps),
                "frame_index": pa.array(frame_index),
            })
            stats.update({"observation.state": vector_stats(state), "action": vector_stats(action),
                          "timestamp": scalar_stats(timestamps), "frame_index": scalar_stats(frame_index)})
            return self._commit(scratch, table, stats, task)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def append_dataset(self, source: pathlib.Path, task: str | None = None,
                       labels: dict[int, str] | None = None) -> list[int]:
        """Copies every episode of another LeRobot v2.1 dataset of the same layout (e.g. one dhk-collect recorded)
        into this one, renumbered; returns the new indices. Refuses a dataset recorded another way. `task`, when given,
        replaces every episode's task text. `labels`, when given, maps source episode index to task text, and only those
        episodes are copied (in source order)."""
        info_path = source / "meta" / "info.json"
        if not info_path.exists():
            raise DatasetError(f"{source} is not a LeRobot dataset (no meta/info.json)")
        info = json.loads(info_path.read_text())
        if info.get("codebase_version") != CODEBASE_VERSION:
            raise DatasetError(f"{source} is LeRobot {info.get('codebase_version')}; only {CODEBASE_VERSION} is read")
        self._check(info, source)
        meta = source / "meta"
        episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line.strip()]
        stats_rows = [json.loads(line) for line in (meta / "episodes_stats.jsonl").read_text().splitlines() if line]
        stats_by = {int(r["episode_index"]): r["stats"] for r in stats_rows}
        tasks = {int(r["task_index"]): r["task"] for r in
                 (json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines() if line.strip())}
        added = []
        if labels is not None:
            unknown = sorted(set(labels) - {int(r["episode_index"]) for r in episodes})
            if unknown:
                raise DatasetError(f"{source} has no episodes {unknown}")
        for row in episodes:
            old = int(row["episode_index"])
            if labels is not None and old not in labels:
                continue
            chunk = old // int(info.get("chunks_size", CHUNKS_SIZE))
            table = pq.read_table(source / info["data_path"].format(episode_chunk=chunk, episode_index=old))
            text = (labels or {}).get(old) or task or (tasks[int(table["task_index"][0].as_py())] if "task_index" in table.column_names
                            else row["tasks"][0])
            table = table.select(["observation.state", "action", "timestamp", "frame_index"])
            scratch = self._scratch()
            try:
                for key in self.video_keys:
                    shutil.copy2(source / info["video_path"].format(episode_chunk=chunk, video_key=key,
                                                                    episode_index=old), scratch / f"{key}.mp4")
                stats = {k: v for k, v in stats_by[old].items() if k not in ("episode_index", "index", "task_index")}
                added.append(self._commit(scratch, table, stats, text))
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        return added

    def _scratch(self) -> pathlib.Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return pathlib.Path(tempfile.mkdtemp(dir=self.root, prefix=".episode-"))

    def _commit(self, scratch: pathlib.Path, table: pa.Table, stats: dict[str, Any], task: str) -> int:
        """Numbers the episode, moves its parquet and videos in (all written first), and updates meta/."""
        index = int(self.info["total_episodes"])
        chunk = index // CHUNKS_SIZE
        frames = table.num_rows
        first = int(self.info["total_frames"])
        new_task = task not in self.tasks
        task_index = self.tasks.setdefault(task, len(self.tasks))
        table = table.append_column("episode_index", pa.array(np.full(frames, index, dtype=np.int64)))
        table = table.append_column("index", pa.array(np.arange(first, first + frames, dtype=np.int64)))
        table = table.append_column("task_index", pa.array(np.full(frames, task_index, dtype=np.int64)))
        stats = {**stats, "episode_index": scalar_stats(np.full(frames, index)),
                 "index": scalar_stats(np.arange(first, first + frames)),
                 "task_index": scalar_stats(np.full(frames, task_index))}
        try:
            pq.write_table(table, scratch / "data.parquet")
            moves = [(scratch / f"{key}.mp4",
                      self.root / VIDEO_PATH.format(episode_chunk=chunk, video_key=key, episode_index=index))
                     for key in self.video_keys]
            moves.append((scratch / "data.parquet",
                          self.root / DATA_PATH.format(episode_chunk=chunk, episode_index=index)))
            for source, target in moves:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
        except BaseException:
            if new_task:
                del self.tasks[task]
            raise
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        if new_task:
            _append_jsonl(meta / "tasks.jsonl", {"task_index": task_index, "task": task})
        _append_jsonl(meta / "episodes.jsonl", {"episode_index": index, "tasks": [task], "length": frames})
        _append_jsonl(meta / "episodes_stats.jsonl", {"episode_index": index, "stats": stats})
        self.info.update(total_episodes=index + 1, total_frames=first + frames, total_tasks=len(self.tasks),
                         total_videos=(index + 1) * len(self.video_keys), total_chunks=chunk + 1,
                         splits={"train": f"0:{index + 1}"})
        _write_json(meta / "info.json", self.info)
        return index
