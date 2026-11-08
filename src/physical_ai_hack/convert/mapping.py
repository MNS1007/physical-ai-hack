"""The mapping file: which recorded data fills each part of the output, by entity and scalar name, never by position.

```yaml
robot_type: metal          # info.json's robot_type
fps: 30                    # output rate; every stream is sampled latest-at on this clock
timeline: time             # the recording's timeline to align on
frame: {width: 640, height: 360}   # every camera: center-cropped to this aspect, then resized
task: "stack the blocks on top of each other"   # default task text (--task overrides)
parts:                     # in vector order; each part is `names` wide
  - name: right_arm
    names: [right_joint_0, right_joint_1, ...]
    state:  {entity: /right/joint_pos, scalars: [joint1, joint2, ...]}
    action: {entity: /right/joint_cmd, scalars: [joint1, joint2, ...]}
    unit: rad              # rad (default) or deg (converted to rad)
    scale: 1.0             # multiplied after the unit conversion (a 0-1 gripper to mm: 100)
    fill: [...]            # optional: the value, in output units, when the source is absent or not given
cameras:
  observation.images.scene: {entity: /cameras/scene}
  observation.images.left_wrist: {entity: /cameras/wrist_left, fill: black}   # fill only when absent
```

A part or camera whose source is missing from a recording, and that has no `fill`, refuses that recording.
"""
from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Any

UNITS = {"rad": 1.0, "deg": math.pi / 180.0}


class MappingError(ValueError):
    """The mapping file is malformed."""


@dataclasses.dataclass(frozen=True)
class Source:
    entity: str
    scalars: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Part:
    name: str
    names: tuple[str, ...]
    state: Source | None
    action: Source | None
    factor: float  # unit conversion times scale
    fill: tuple[float, ...] | None

    @property
    def width(self) -> int:
        return len(self.names)


@dataclasses.dataclass(frozen=True)
class Camera:
    key: str  # output video key, observation.images.<role>
    entity: str | None
    fill: str | None  # "black" or None


@dataclasses.dataclass(frozen=True)
class Mapping:
    robot_type: str
    fps: int
    timeline: str
    width: int
    height: int
    task: str
    parts: tuple[Part, ...]
    cameras: tuple[Camera, ...]

    @property
    def vector_names(self) -> list[str]:
        return [n for p in self.parts for n in p.names]


def _source(raw: Any, part: str, which: str, width: int) -> Source | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or "entity" not in raw or "scalars" not in raw:
        raise MappingError(f"part {part!r}: {which} needs entity and scalars")
    scalars = tuple(str(s) for s in raw["scalars"])
    if len(scalars) != width:
        raise MappingError(f"part {part!r}: {which} names {len(scalars)} scalars for {width} outputs")
    return Source(str(raw["entity"]), scalars)


def parse(raw: dict[str, Any]) -> Mapping:
    for key in ("robot_type", "fps", "timeline", "frame", "parts", "cameras"):
        if key not in raw:
            raise MappingError(f"mapping needs {key!r}")
    parts = []
    for p in raw["parts"]:
        name = str(p["name"])
        names = tuple(str(n) for n in p["names"])
        unit = p.get("unit", "rad")
        if unit not in UNITS:
            raise MappingError(f"part {name!r}: unit {unit!r} is not one of {sorted(UNITS)}")
        fill = p.get("fill")
        if fill is not None:
            fill = tuple(float(v) for v in fill)
            if len(fill) != len(names):
                raise MappingError(f"part {name!r}: fill has {len(fill)} values for {len(names)} outputs")
        state = _source(p.get("state"), name, "state", len(names))
        action = _source(p.get("action"), name, "action", len(names))
        if (state is None) != (action is None):
            raise MappingError(f"part {name!r}: give both state and action, or neither (then fill)")
        if state is None and fill is None:
            raise MappingError(f"part {name!r}: no state/action source and no fill")
        parts.append(Part(name, names, state, action, UNITS[unit] * float(p.get("scale", 1.0)), fill))
    if len({n for p in parts for n in p.names}) != sum(p.width for p in parts):
        raise MappingError("output names repeat across parts")
    cameras = []
    for key, c in raw["cameras"].items():
        c = c or {}
        fill = c.get("fill")
        if fill not in (None, "black"):
            raise MappingError(f"camera {key!r}: fill must be 'black' or absent")
        if c.get("entity") is None and fill is None:
            raise MappingError(f"camera {key!r}: no entity and no fill")
        cameras.append(Camera(str(key), c.get("entity"), fill))
    frame = raw["frame"]
    return Mapping(str(raw["robot_type"]), int(raw["fps"]), str(raw["timeline"]), int(frame["width"]),
                   int(frame["height"]), str(raw.get("task") or "do the demonstrated task"), tuple(parts),
                   tuple(cameras))


def load(path: str | pathlib.Path) -> Mapping:
    import yaml

    return parse(yaml.safe_load(pathlib.Path(path).read_text()))
