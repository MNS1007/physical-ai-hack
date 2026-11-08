"""The judge: a frontier vision model that looks at the cameras after an attempt and says whether the step is done.

    ClaudeJudge(model="claude-opus-5-5").check(step, {"scene": jpeg, "wrist_r": jpeg}) -> Verdict

Needs the `judge` dependency group (`anthropic`) and ANTHROPIC_API_KEY (environment or .env). Any failure (no key,
timeout, refusal, bad answer) comes back as a Verdict with `done=False, ok=False`: the supervisor never treats an
unanswered question as a pass.
"""
from __future__ import annotations

import base64
import dataclasses
import json
import time
from typing import Protocol

from .stages import Step

DEFAULT_MODEL = "claude-opus-5-5"
TIMEOUT_S = 20.0

SYSTEM = (
    "You are the referee's assistant at a robot-arm competition. You get camera images of the table right after the "
    "robot tried a task, and one question about the scene. Answer only from what the images show. When the images "
    "cannot settle it (blurry, occluded, glare), say done=false and explain. Be strict: a partial result is not done."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "done": {"type": "boolean", "description": "the question's condition is met"},
        "confidence": {"type": "number", "description": "0 to 1"},
        "reason": {"type": "string", "description": "one sentence on what the images show"},
        "layers": {"type": "integer", "description": "stack: layers of the tallest stack; otherwise 0"},
        "can_add": {"type": "boolean", "description": "stack: a free cube remains and the stack looks stable"},
    },
    "required": ["done", "confidence", "reason", "layers", "can_add"],
    "additionalProperties": False,
}


@dataclasses.dataclass(frozen=True)
class Verdict:
    done: bool
    confidence: float = 0.0
    reason: str = ""
    layers: int = 0
    can_add: bool = False
    ok: bool = True  # False: the judge could not answer (error, no key, timeout); never a pass
    seconds: float = 0.0
    give_up: bool = False  # operator judges: "fail" - stop retrying this step


class Judge(Protocol):
    def check(self, step: Step, images: dict[str, bytes]) -> Verdict: ...


class ClaudeJudge:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low") -> None:
        import anthropic

        self.model, self.effort = model, effort
        self.client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=1)

    def check(self, step: Step, images: dict[str, bytes]) -> Verdict:
        t0 = time.monotonic()
        content: list[dict] = []
        for name, jpeg in images.items():
            content.append({"type": "text", "text": f"Camera: {name}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.standard_b64encode(jpeg).decode()}})
        content.append({"type": "text", "text": f"Task the robot was given: {step.prompt!r}.\n{step.question}"})
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=2000, system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            )
            if response.stop_reason in ("refusal", "max_tokens"):
                return Verdict(False, reason=f"judge stopped: {response.stop_reason}", ok=False,
                               seconds=time.monotonic() - t0)
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
            return Verdict(bool(data["done"]), float(data["confidence"]), str(data["reason"]), int(data["layers"]),
                           bool(data["can_add"]), True, time.monotonic() - t0)
        except Exception as error:  # noqa: BLE001 - any failure is "not verified", reported
            return Verdict(False, reason=f"judge error: {type(error).__name__}: {error}", ok=False,
                           seconds=time.monotonic() - t0)


# --- operator judges: a person decides instead of the model -------------------------------------------------------

DONE_WORDS = ("done", "yes", "pass", "passed", "good", "success", "complete")
RETRY_WORDS = ("retry", "again", "redo", "repeat", "one more")
FAIL_WORDS = ("fail", "failed", "skip", "give up", "abort")


def parse_word(text: str) -> str | None:
    """'done', 'retry' or 'fail' from what the operator said or typed; None when it is none of them."""
    words = " " + "".join(c if c.isalnum() else " " for c in text.lower()) + " "
    for verdict, options in (("fail", FAIL_WORDS), ("retry", RETRY_WORDS), ("done", DONE_WORDS)):
        if any(f" {w} " in words for w in options):
            return verdict
    return None


def _verdict(word: str | None, heard: str, t0: float) -> Verdict:
    if word == "done":
        return Verdict(True, 1.0, f"operator: {heard!r}", seconds=time.monotonic() - t0)
    if word == "fail":
        return Verdict(False, 1.0, f"operator: {heard!r}", give_up=True, seconds=time.monotonic() - t0)
    if word == "retry":
        return Verdict(False, 1.0, f"operator: {heard!r}", seconds=time.monotonic() - t0)
    return Verdict(False, reason="operator gave no answer", ok=False, seconds=time.monotonic() - t0)


class KeyboardJudge:
    """The operator types done / retry / fail. For testing the flow; typing is human input in a match."""

    def check(self, step: Step, images: dict[str, bytes]) -> Verdict:
        t0 = time.monotonic()
        while True:
            heard = input(f"[{step.name}] done, retry or fail? ")
            word = parse_word(heard)
            if word:
                return _verdict(word, heard, t0)


class VoiceJudge:
    """The operator says done / retry / fail in the LiveKit room; Whisper on this machine transcribes it.

    The rubric: "Organizers will consider allowing voice input." Ask before using it in a match.
    """

    def __init__(self, announcer, timeout_s: float = 30.0) -> None:
        self.announcer, self.timeout_s = announcer, timeout_s
        if announcer.livekit is None:
            raise RuntimeError("the voice judge listens in the LiveKit room: add --livekit")

    def check(self, step: Step, images: dict[str, bytes]) -> Verdict:
        t0 = time.monotonic()
        self.announcer.livekit.drain()
        self.announcer(f"{step.name}: done, retry, or fail?", step=step.name, ask=True)
        while (left := self.timeout_s - (time.monotonic() - t0)) > 0:
            heard = self.announcer.livekit.heard(timeout=left)
            if heard is None:
                break
            word = parse_word(heard)
            print(f"[voice] heard {heard!r} -> {word}")
            if word:
                return _verdict(word, heard, t0)
        return _verdict(None, "", t0)
