# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["dimos==0.0.15", "moondream", "openai", "python-dotenv"]
# ///
"""Dimensional's vision models (dimos.models.vl) as a second opinion on a step: moves nothing.

    uv run tools/dimos_judge.py stack scene.jpg            # saved image
    uv run tools/dimos_judge.py unplug --live              # the station's scene camera now (station-plus, or a tunnel)

Runs in its own environment (the inline dependencies above), apart from the harness and the Dream SDK. The model:
MOONDREAM_API_KEY -> Moondream hosted (free key at moondream.ai), else OPENAI_API_KEY -> dimos's OpenAI model.
Prints one JSON line: done, what was detected, the model's answer.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time
import urllib.request

QUESTIONS = {
    "unplug": ("Is there a USB cable lying on the table with its plug fully pulled out of the power strip? "
               "Answer yes or no, then one short sentence."),
    "stack": ("How many cubes are stacked on top of each other in the tallest stack? "
              "Answer with a number, then one short sentence."),
}
DETECT = {"unplug": "usb plug", "stack": "cube"}


def model():
    if os.environ.get("MOONDREAM_API_KEY"):
        from dimos.models.vl.moondream_hosted import MoondreamHostedVlModel

        return "moondream-hosted", MoondreamHostedVlModel()
    if os.environ.get("OPENAI_API_KEY"):
        from dimos.models.vl.openai import OpenAIVlModel

        return "openai", OpenAIVlModel()
    sys.exit("set MOONDREAM_API_KEY (free at moondream.ai) or OPENAI_API_KEY in .env")


def boxes(detections) -> list[tuple[float, float, float, float]]:
    items = getattr(detections, "detections", detections) or []
    return [tuple(float(v) for v in d.bbox) for d in items]


def stack_layers(bboxes: list[tuple[float, float, float, float]]) -> int:
    """Layers of the tallest column: boxes that overlap horizontally by half and sit one above the next."""
    best = 0
    for base in bboxes:
        column = [b for b in bboxes if _overlap_x(base, b) >= 0.5]
        column.sort(key=lambda b: b[3], reverse=True)  # bottom first (largest y2)
        layers, top = 1, column[0] if column else base
        for b in column[1:]:
            height = top[3] - top[1]
            if b[3] <= top[1] + 0.5 * height and b[3] >= top[1] - 0.5 * height:  # its bottom near the last top
                layers, top = layers + 1, b
        best = max(best, layers if column else 1)
    return best


def _overlap_x(a, b) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def first_number(text: str) -> int | None:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "zero": 0, "no": 0}
    for token in text.lower().replace(",", " ").replace(".", " ").split():
        if token.isdigit():
            return int(token)
        if token in words:
            return words[token]
    return None


def judge(step: str, image, vl) -> dict:
    answer = vl.query(image, QUESTIONS[step])
    found = boxes(vl.query_detections(image, DETECT[step]))
    result = {"step": step, "answer": answer.strip(), "detected": DETECT[step], "count": len(found)}
    if step == "unplug":
        result["done"] = answer.strip().lower().startswith("yes")
    else:
        said = first_number(answer)
        result["layers_from_boxes"] = stack_layers(found)
        result["layers_said"] = said
        result["done"] = (said or 0) >= 3
    return result


def live_frame(url: str, camera: str) -> pathlib.Path:
    path = pathlib.Path(tempfile.mkdtemp()) / f"{camera}.jpg"
    with urllib.request.urlopen(f"{url}/frame/{camera}.jpg", timeout=10) as r:
        path.write_bytes(r.read())
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=sorted(QUESTIONS))
    p.add_argument("images", nargs="*", help="saved images (the first is judged); or --live")
    p.add_argument("--live", action="store_true", help="grab the station's camera now")
    p.add_argument("--camera", default="scene")
    p.add_argument("--url", default="http://127.0.0.1:8091")
    args = p.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    path = live_frame(args.url, args.camera) if args.live else pathlib.Path(args.images[0]) if args.images else None
    if path is None:
        p.error("give an image or --live")
    from dimos.msgs.sensor_msgs.Image import Image

    name, vl = model()
    t0 = time.monotonic()
    result = judge(args.step, Image.from_file(str(path)), vl)
    print(json.dumps({"judge": f"dimos/{name}", "image": str(path), **result,
                      "seconds": round(time.monotonic() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
