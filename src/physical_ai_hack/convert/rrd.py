"""Read a Rerun recording into plain arrays, per entity: scalar series and video streams, on one timeline.

Only the parts a converter needs: `Scalars:scalars` rows (with the static `SeriesLines:names`), and `VideoStream`
samples (with the static `VideoStream:codec`). Read chunk by chunk with `rerun.chunk.RrdReader`, so no query engine
is needed. Needs rerun-sdk at the version that wrote the file (0.42 for these recordings).
"""
from __future__ import annotations

import dataclasses
import pathlib
from collections import defaultdict

import numpy as np
import pyarrow as pa

SCALARS = "Scalars:scalars"
SERIES_NAMES = "SeriesLines:names"
SAMPLE = "VideoStream:sample"
KEYFRAME = "VideoStream:is_keyframe"
CODEC = "VideoStream:codec"


@dataclasses.dataclass
class Series:
    """One entity's scalar rows: `values[i]` at `times[i]` (ns), columns named by `names`."""

    times: np.ndarray  # int64 ns, ascending
    values: np.ndarray  # float64 [rows, columns]
    names: list[str]


@dataclasses.dataclass
class Video:
    """One entity's encoded video samples, one frame each, in time order."""

    times: np.ndarray  # int64 ns, ascending
    samples: list[bytes]
    keyframes: np.ndarray  # bool
    codec: str  # fourcc, e.g. "avc1" (H.264)


@dataclasses.dataclass
class Recording:
    path: pathlib.Path
    application_id: str | None
    timelines: set[str]
    series: dict[str, Series]
    videos: dict[str, Video]
    other: dict[str, set[str]]  # entity -> components we do not read, for inspect


def fourcc(code: int) -> str:
    return int(code).to_bytes(4, "big").decode("ascii", errors="replace")


def _cell(column: pa.Array, row: int):
    value = column[row].as_py()
    return value


def read(path: str | pathlib.Path, timeline: str) -> Recording:
    """Reads every scalar and video entity of the file's first recording, timed on `timeline`."""
    from rerun.chunk import RrdReader

    path = pathlib.Path(path)
    reader = RrdReader(path)
    recordings = reader.recordings()
    app = getattr(recordings[0], "application_id", None) if recordings else None
    timelines: set[str] = set()
    scal_t, scal_v, names = defaultdict(list), defaultdict(list), {}
    vid_t, vid_s, vid_k, codecs = defaultdict(list), defaultdict(list), defaultdict(list), {}
    other: dict[str, set[str]] = defaultdict(set)
    if not recordings:
        return Recording(path, None, set(), {}, {}, {})
    for chunk in reader.stream():
        batch = chunk.to_record_batch()
        cols = dict(zip(batch.schema.names, batch.columns))
        entity = chunk.entity_path
        timelines.update(chunk.timeline_names)
        if chunk.is_static:
            if SERIES_NAMES in cols:
                names[entity] = [str(n) for n in _cell(cols[SERIES_NAMES], 0)]
            if CODEC in cols:
                codecs[entity] = fourcc(_cell(cols[CODEC], 0)[0])
            continue
        if timeline not in cols:
            other[entity].add(f"(not on timeline {timeline!r})")
            continue
        times = cols[timeline].cast(pa.int64()).to_numpy(zero_copy_only=False)
        if SCALARS in cols:
            for i, row in enumerate(cols[SCALARS].to_pylist()):
                if row is not None:
                    scal_t[entity].append(times[i])
                    scal_v[entity].append(row)
        if SAMPLE in cols:
            keys = cols[KEYFRAME].to_pylist() if KEYFRAME in cols else [None] * len(times)
            for i, row in enumerate(cols[SAMPLE].to_pylist()):
                if row:
                    vid_t[entity].append(times[i])
                    vid_s[entity].append(bytes(row[0]))
                    vid_k[entity].append(bool(keys[i][0]) if keys[i] else False)
        for name in cols:
            if name not in (SCALARS, SAMPLE, KEYFRAME, "rerun.controls.RowId") and name not in chunk.timeline_names:
                other[entity].add(name)

    series = {}
    for entity, rows in scal_v.items():
        order = np.argsort(np.asarray(scal_t[entity]), kind="stable")
        values = np.asarray(rows, dtype=np.float64)[order]
        series[entity] = Series(np.asarray(scal_t[entity], dtype=np.int64)[order], values,
                                names.get(entity, [str(i) for i in range(values.shape[1])]))
    videos = {}
    for entity, samples in vid_s.items():
        order = np.argsort(np.asarray(vid_t[entity]), kind="stable")
        videos[entity] = Video(np.asarray(vid_t[entity], dtype=np.int64)[order], [samples[i] for i in order],
                               np.asarray(vid_k[entity], dtype=bool)[order], codecs.get(entity, "?"))
    return Recording(path, app, timelines, series, videos, dict(other))
