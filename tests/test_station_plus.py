"""station-plus routes on Dream's real HTTP server, with a fake controller shaped like StationController.

    uv run --group convert python -m unittest tests.test_station_plus
"""
import asyncio
import json
import types
import unittest
import urllib.error
import urllib.request

try:
    import numpy as np
    from robots import station_api

    from physical_ai_hack import station_plus
except ImportError:  # the convert group is not installed
    station_api = None


class FakeController:
    def __init__(self):
        self.cameras = {"scene": types.SimpleNamespace(data=np.zeros((36, 64, 3), np.uint8))}

    @property
    def frames(self):  # a property, as on Dream's StationController
        return self.cameras

    def arm_states(self):
        return {"right": types.SimpleNamespace(positions_rad=np.zeros(6), gripper=0.5)}

    def status(self):
        return {"state": "idle"}


@unittest.skipIf(station_api is None, "needs the convert group (dreamlabs)")
class RoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        station_plus.install()
        cls.api = station_api.StationApi(FakeController(), asyncio.new_event_loop(), port=0)
        cls.api.start()
        cls.url = f"http://127.0.0.1:{cls.api.port}"

    @classmethod
    def tearDownClass(cls):
        cls.api.stop()

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=5) as r:
            return r.status, r.read()

    def test_frames_lists_cameras(self):
        self.assertEqual(json.loads(self.get("/frames")[1]), {"cameras": ["scene"]})

    def test_frame_is_jpeg(self):
        status, body = self.get("/frame/scene.jpg")
        self.assertEqual((status, body[:2]), (200, b"\xff\xd8"))

    def test_unknown_camera_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/frame/nope.jpg")
        self.assertEqual(caught.exception.code, 404)

    def test_arms(self):
        self.assertEqual(json.loads(self.get("/arms")[1])["right"]["gripper"], 0.5)

    def test_status_is_dreams(self):
        self.assertEqual(json.loads(self.get("/status")[1]), {"state": "idle"})


if __name__ == "__main__":
    unittest.main()
