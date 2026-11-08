"""Converter logic on in-memory recordings: mapping checks, the clock, latest-at, units, fills. No .rrd needed.

    uv run --group convert python -m unittest discover tests
"""
import math
import unittest

import numpy as np

from physical_ai_hack.convert import episode as ep
from physical_ai_hack.convert import mapping as mp
from physical_ai_hack.convert.rrd import Recording, Series, Video

S = 1_000_000_000


def mapping(**over):
    raw = {
        "robot_type": "metal", "fps": 10, "timeline": "time", "frame": {"width": 64, "height": 36},
        "parts": [
            {"name": "left_arm", "names": ["l0", "l1"], "fill": [0.5, -0.5]},
            {"name": "right_arm", "names": ["r0", "r1"], "unit": "deg",
             "state": {"entity": "/r/pos", "scalars": ["j2", "j1"]},
             "action": {"entity": "/r/cmd", "scalars": ["j2", "j1"]}},
            {"name": "right_gripper", "names": ["rg"], "scale": 100,
             "state": {"entity": "/r/pos", "scalars": ["grip"]},
             "action": {"entity": "/r/cmd", "scalars": ["grip"]}},
        ],
        "cameras": {"observation.images.scene": {"entity": "/cam/scene"},
                    "observation.images.left_wrist": {"entity": "/cam/left", "fill": "black"}},
    }
    raw.update(over)
    return mp.parse(raw)


def recording(seconds=3.0, video_start=0.5):
    t = (np.arange(0, seconds, 0.05) * S).astype(np.int64)  # 20 Hz
    values = np.stack([np.full(len(t), 90.0), np.arange(len(t), dtype=float), np.full(len(t), 0.25)], axis=1)
    names = ["j1", "j2", "grip"]
    vt = (np.arange(video_start, seconds, 1 / 30) * S).astype(np.int64)
    video = Video(vt, [b""] * len(vt), np.zeros(len(vt), bool), "avc1")
    return Recording(None, "test", {"time"}, {"/r/pos": Series(t, values, names), "/r/cmd": Series(t, values, names)},
                     {"/cam/scene": video}, {})


class MappingTest(unittest.TestCase):
    def test_part_without_source_or_fill_is_refused(self):
        with self.assertRaises(mp.MappingError):
            mapping(parts=[{"name": "x", "names": ["a"]}])

    def test_scalar_count_must_match_outputs(self):
        with self.assertRaises(mp.MappingError):
            mapping(parts=[{"name": "x", "names": ["a", "b"], "state": {"entity": "/e", "scalars": ["s"]},
                            "action": {"entity": "/e", "scalars": ["s"]}}])

    def test_camera_without_entity_or_fill_is_refused(self):
        with self.assertRaises(mp.MappingError):
            mapping(cameras={"observation.images.scene": {}})

    def test_shipped_mapping_parses(self):
        m = mp.load("configs/convert/metal_right_arm.yaml")
        self.assertEqual(len(m.vector_names), 14)
        self.assertEqual(m.vector_names[:7][-1], "left_gripper")


class EpisodeTest(unittest.TestCase):
    def test_clock_spans_where_every_stream_runs(self):
        plan = ep.plan(recording(), mapping())
        self.assertEqual(plan.ticks[0], int(0.5 * S))  # video starts last
        self.assertIn("left_arm", plan.filled)
        self.assertIn("observation.images.left_wrist", plan.filled)

    def test_vectors_select_by_name_convert_units_and_fill(self):
        m = mapping()
        rec = recording()
        plan = ep.plan(rec, m)
        state, action = ep.vectors(rec, m, plan.ticks)
        self.assertEqual(state.shape, (len(plan.ticks), 5))
        np.testing.assert_allclose(state[:, :2], [[0.5, -0.5]] * len(plan.ticks))  # fill
        np.testing.assert_allclose(state[0, 3], math.radians(90.0), rtol=1e-6)  # j1 in deg -> rad, second
        np.testing.assert_allclose(state[0, 2], math.radians(10.0), rtol=1e-6)  # j2 = row index 10 at t=0.5 s
        np.testing.assert_allclose(state[:, 4], 25.0)  # gripper 0.25 * 100

    def test_latest_at_never_looks_ahead(self):
        times = np.array([0, 10, 20])
        np.testing.assert_array_equal(ep.latest_at(times, np.array([0, 9, 10, 25])), [0, 0, 1, 2])

    def test_missing_required_stream_is_refused(self):
        rec = recording()
        del rec.series["/r/cmd"]
        with self.assertRaises(ep.EpisodeError):
            ep.plan(rec, mapping())

    def test_too_short_is_refused(self):
        with self.assertRaises(ep.EpisodeError):
            ep.plan(recording(seconds=1.0, video_start=0.1), mapping())


if __name__ == "__main__":
    unittest.main()
