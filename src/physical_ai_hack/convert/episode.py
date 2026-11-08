"""One recording to one episode: a common clock, the state/action vectors, and each camera's frames on that clock."""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np

from .mapping import Mapping, Part
from .rrd import Recording, Video

MIN_EPISODE_S = 1.6  # the shortest episode Dream trains on (the kit's MIN_EPISODE_S)
STALE_TICKS = 3  # a camera frame older than this many ticks counts as stale


class EpisodeError(ValueError):
    """This recording cannot become an episode under this mapping; the message says why."""


@dataclasses.dataclass
class Plan:
    ticks: np.ndarray  # int64 ns, one per output frame
    head_s: float  # seconds trimmed at the start (streams not all running yet)
    tail_s: float  # seconds trimmed at the end
    filled: list[str]  # parts and cameras filled, not recorded


def clock(starts: list[int], ends: list[int], fps: int) -> np.ndarray:
    """Ticks every 1/fps from the latest start to the earliest end, so every stream has a value at every tick."""
    t0, t1 = max(starts), min(ends)
    if t1 <= t0:
        return np.zeros(0, dtype=np.int64)
    step = 1e9 / fps
    count = int((t1 - t0) // step) + 1
    return (t0 + np.round(np.arange(count) * step)).astype(np.int64)


def latest_at(times: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """For each tick, the index of the last row at or before it (-1 if none)."""
    return np.searchsorted(times, ticks, side="right") - 1


def plan(rec: Recording, m: Mapping) -> Plan:
    """Checks the recording against the mapping and builds the clock over every stream that will be used."""
    missing, filled, starts, ends = [], [], [], []
    for part in m.parts:
        for which, src in (("state", part.state), ("action", part.action)):
            if src is None:
                continue
            series = rec.series.get(src.entity)
            if series is None:
                continue
            absent = [s for s in src.scalars if s not in series.names]
            if absent:
                raise EpisodeError(f"part {part.name!r} {which}: {src.entity} has no scalars {absent} "
                                   f"(it has {series.names})")
            starts.append(int(series.times[0]))
            ends.append(int(series.times[-1]))
        sourced = part.state is not None and all(rec.series.get(s.entity) for s in (part.state, part.action))
        if not sourced:
            if part.fill is None:
                entity = part.state.entity if part.state else "(none)"
                missing.append(f"part {part.name!r} (entity {entity})")
            else:
                filled.append(part.name)
    for cam in m.cameras:
        video = rec.videos.get(cam.entity) if cam.entity else None
        if video is None or len(video.times) < 2:
            if cam.fill is None:
                missing.append(f"camera {cam.key} (entity {cam.entity})")
            else:
                filled.append(cam.key)
            continue
        if video.codec not in ("avc1", "h264"):
            raise EpisodeError(f"camera {cam.key}: codec {video.codec!r} is not H.264")
        starts.append(int(video.times[0]))
        ends.append(int(video.times[-1]))
    if missing:
        raise EpisodeError("not recorded, and no fill in the mapping: " + "; ".join(missing))
    if not starts:
        raise EpisodeError("nothing recorded to align on")
    ticks = clock(starts, ends, m.fps)
    if len(ticks) < round(MIN_EPISODE_S * m.fps) + 1:
        raise EpisodeError(f"only {len(ticks) / m.fps:.2f} s where every stream is running; "
                           f"Dream needs at least {MIN_EPISODE_S} s")
    return Plan(ticks, (max(starts) - min(starts)) / 1e9, (max(ends) - min(ends)) / 1e9, filled)


def _part_values(rec: Recording, part: Part, which: str, ticks: np.ndarray) -> np.ndarray:
    src = part.state if which == "state" else part.action
    series = rec.series.get(src.entity) if src else None
    if series is None or (part.action and rec.series.get(part.action.entity) is None) \
            or (part.state and rec.series.get(part.state.entity) is None):
        return np.tile(np.asarray(part.fill, dtype=np.float64), (len(ticks), 1))
    columns = [series.names.index(s) for s in src.scalars]
    rows = latest_at(series.times, ticks)
    return series.values[rows][:, columns] * part.factor


def vectors(rec: Recording, m: Mapping, ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The state and action vectors at every tick, float32 [ticks, width], parts in mapping order."""
    state = np.concatenate([_part_values(rec, p, "state", ticks) for p in m.parts], axis=1)
    action = np.concatenate([_part_values(rec, p, "action", ticks) for p in m.parts], axis=1)
    return state.astype(np.float32), action.astype(np.float32)


def crop_resize(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-crops to width:height, then resizes; BGR uint8 in and out."""
    import av

    h, w = bgr.shape[:2]
    target = width / height
    if w / h > target:
        cw = round(h * target)
        x = (w - cw) // 2
        bgr = bgr[:, x:x + cw]
    elif w / h < target:
        ch = round(w / target)
        y = (h - ch) // 2
        bgr = bgr[y:y + ch]
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format="bgr24")
    return frame.reformat(width=width, height=height, format="bgr24").to_ndarray()


def _decoded(video: Video) -> Iterator[tuple[int, np.ndarray]]:
    """(time, BGR frame) for each sample, decoding the stream in order; frames pair with sample times by order."""
    import av

    codec = av.CodecContext.create("h264", "r")
    times = iter(video.times.tolist())
    for sample in video.samples:
        for frame in codec.decode(av.Packet(sample)):
            yield next(times), frame.to_ndarray(format="bgr24")
    for frame in codec.decode(None):
        yield next(times), frame.to_ndarray(format="bgr24")


def frames(video: Video | None, ticks: np.ndarray, width: int, height: int,
           stats: dict[str, int] | None = None) -> Iterator[np.ndarray]:
    """One BGR frame per tick: the latest decoded frame at or before it; black when there is no video (fill)."""
    if video is None:
        black = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in ticks:
            yield black
        return
    period = (ticks[1] - ticks[0]) if len(ticks) > 1 else 1
    source = _decoded(video)
    current_t, current = None, None
    pending = next(source, None)
    for tick in ticks.tolist():
        while pending is not None and pending[0] <= tick:
            current_t, raw = pending
            current = None  # resize lazily: only frames that are used
            pending_raw = raw
            pending = next(source, None)
        if current is None:
            if current_t is None:
                raise EpisodeError("a camera has no frame at the first tick")
            current = crop_resize(pending_raw, width, height)
        if stats is not None:
            stats["frames"] = stats.get("frames", 0) + 1
            if tick - current_t > STALE_TICKS * period:
                stats["stale"] = stats.get("stale", 0) + 1
        yield current
