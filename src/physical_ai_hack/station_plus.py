"""Dream's own station agent (`python -m robots.station`), plus two read-only routes the harness's judge needs.

    uv run --group convert station-plus --config ~/.dreamlabs/station.yaml      # instead of `dreamcli robot up`

Adds to the agent's HTTP API (127.0.0.1:8091):
- `GET /frame/<camera>.jpg`: the latest frame of that camera (e.g. `scene`, `wrist_l`, `wrist_r`), as JPEG.
- `GET /frames`: the camera names available.
- `GET /arms`: each arm's joint positions (rad) and gripper (0-1), as the drivers last reported.

Everything else is Dream's agent unchanged: the same Start/Stop/Prompt routes, safety rules and camera readers. The
agent owns the USB cameras, so this is how another program sees them.
"""
from __future__ import annotations

import io
import sys

JPEG_QUALITY = 85


def _jpeg(rgb) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def install() -> None:
    """Wraps the agent's GET handler; POST routes are untouched."""
    from robots import station_api

    handler = station_api._Handler
    original = handler.do_GET

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        route = self.path.split("?", 1)[0]
        controller = self.server.api.controller
        if route == "/frames":
            self._reply(200, {"cameras": sorted(controller.frames())})
            return
        if route == "/arms":
            arms = {name: {"positions_rad": [float(v) for v in s.positions_rad], "gripper": float(s.gripper)}
                    for name, s in controller.arm_states().items()}
            self._reply(200, arms)
            return
        if route.startswith("/frame/") and route.endswith(".jpg"):
            name = route[len("/frame/"):-len(".jpg")]
            frame = controller.frames().get(name)
            if frame is None:
                self._reply(404, {"error": f"no frame from camera {name!r}; cameras: {sorted(controller.frames())}"})
                return
            body = _jpeg(frame.data)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        original(self)

    handler.do_GET = do_GET


def main() -> None:
    install()
    from robots import station

    station.main(sys.argv[1:])
