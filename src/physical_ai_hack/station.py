"""A client for the Dream station agent's HTTP API (`dreamcli robot up`, the `robots.station_api` module).

| Route | Body | Does |
| --- | --- | --- |
| `GET /status` | | the station's state (`idle`, `starting`, `running`, `stopping`, `homing`, `error`), prompt, ... |
| `POST /start` | `{"prompt", "checkpoint"?}` | starts a run with that task (409 when one is running) |
| `POST /prompt` | `{"prompt"}` | changes the running task, without stopping or homing |
| `POST /stop` | | ends the run; the arms hold their last command, under power |

Standard library only, so it runs in any Python on the station.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8091"  # robots.station.DEFAULT_API_PORT


class StationError(RuntimeError):
    """The station refused a call, or could not be reached."""


class Station:
    def __init__(self, url: str = DEFAULT_URL, timeout_s: float = 65.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s  # a stop waits for the loop to end; the agent's own limit is 60 s

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as answer:
                return json.loads(answer.read() or b"{}")
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read()).get("error", "")
            except ValueError:
                message = ""
            raise StationError(f"{method} {path}: HTTP {error.code} {message}".strip()) from None
        except urllib.error.URLError as error:
            raise StationError(f"cannot reach the station agent at {self.url} ({error.reason}); "
                               "is `dreamcli robot up` running?") from None

    def status(self) -> dict[str, Any]:
        return self._call("GET", "/status")

    def start(self, prompt: str, checkpoint: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"prompt": prompt}
        if checkpoint:
            body["checkpoint"] = checkpoint
        return self._call("POST", "/start", body)

    def prompt(self, prompt: str) -> dict[str, Any]:
        return self._call("POST", "/prompt", {"prompt": prompt})

    def stop(self) -> dict[str, Any]:
        return self._call("POST", "/stop")
