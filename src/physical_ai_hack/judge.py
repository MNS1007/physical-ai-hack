"""Claude as the harness's eyes: one camera frame (or a few) and a step in, a structured verdict out.

The model is `HARNESS_MODEL` (default claude-opus-5-5) at `HARNESS_EFFORT` (default low, for latency). Responses are
parsed into `Verdict` with structured outputs. A refusal or an API error comes back as a not-met verdict with the
reason, so the relay never advances on a failure.
"""
from __future__ import annotations

import base64
import os
import time

import anthropic
import cv2
import numpy as np
from pydantic import BaseModel

from .stages import Step

FRAME_WIDTH = 768  # frames shrink to this width before upload: enough to see a plug in a port, fast to send

SYSTEM = """\
You judge a robot arm's progress in a timed manipulation task from camera images. You answer only about what is \
visible now. Be strict: say the goal is met only when you can clearly see it in the images. If a view is blocked \
or blurry, say the goal is not met and say why in `reason`. Count carefully. Keep `reason` to one short sentence."""


class Verdict(BaseModel):
    goal_met: bool  # the step's goal is clearly visible
    count: int  # what the step's `count_means` counts
    gripper_clear: bool  # the gripper is open and not touching the object it last handled
    later_stage_contact: bool  # the gripper touches an object of a later stage (DQ risk)
    reason: str


def encode(frame_rgb: np.ndarray, width: int = FRAME_WIDTH) -> str:
    """An RGB frame as a base64 JPEG, shrunk to `width`."""
    h, w = frame_rgb.shape[:2]
    if w > width:
        frame_rgb = cv2.resize(frame_rgb, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("could not encode the frame as JPEG")
    return base64.standard_b64encode(jpg.tobytes()).decode()


def question(step: Step, later: str) -> str:
    return (f"Task step: {step.name}.\n"
            f"Goal: {step.goal}\n"
            f"`count`: the number of {step.count_means}.\n"
            f"`gripper_clear`: true when the gripper is open and not touching the object it last moved.\n"
            f"`later_stage_contact`: true when the gripper touches any of these, which belong to later steps: "
            f"{later or 'nothing'}.\n"
            "`goal_met`: true only when the goal above is clearly visible.")


class Judge:
    def __init__(self, model: str | None = None, effort: str | None = None) -> None:
        self.model = model or os.environ.get("HARNESS_MODEL", "claude-opus-5-5")
        self.effort = effort or os.environ.get("HARNESS_EFFORT", "low")
        self.client = anthropic.AsyncAnthropic(max_retries=1, timeout=20.0)

    async def __call__(self, step: Step, frames: dict[str, np.ndarray], later: str = "") -> tuple[Verdict, float]:
        """(verdict, seconds the call took). `frames` maps a camera name to an RGB array."""
        content: list[dict] = []
        for name, frame in frames.items():
            content.append({"type": "text", "text": f"Camera: {name}"})
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": encode(frame)}})
        content.append({"type": "text", "text": question(step, later)})
        started = time.monotonic()
        try:
            response = await self.client.beta.messages.parse(
                model=self.model,
                max_tokens=4000,
                system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_format=Verdict,
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as exc:
            return _failed(f"network error: {exc}"), time.monotonic() - started
        except anthropic.RateLimitError:
            return _failed("rate limited"), time.monotonic() - started
        except anthropic.APIStatusError as exc:
            return _failed(f"API error {exc.status_code}: {exc.message}"), time.monotonic() - started
        took = time.monotonic() - started
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return _failed(f"no verdict (stop_reason {response.stop_reason})"), took
        return response.parsed_output, took


def _failed(reason: str) -> Verdict:
    return Verdict(goal_met=False, count=0, gripper_clear=False, later_stage_contact=False, reason=reason)
