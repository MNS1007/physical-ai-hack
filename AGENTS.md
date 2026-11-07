# Working on the harness

Read `README.md` first: the rubric math, how the processes fit, and the rules. Keep it small and readable.

## Layout

- `src/physical_ai_hack/stages.py`: the relay as data (`Step`: prompts, `min_s`, `settle_s`, `timeout_s`) and the
  rubric's `score()`. Most tuning happens here.
- `src/physical_ai_hack/relay.py`: the state machine, with no LiveKit in it. Inputs: `start(now)`,
  `tick(now, still_for_s)`. Output: calls on a `link` (`set_task`, `resume`, `pause`, `to_teleop`) and `Event`s.
- `src/physical_ai_hack/live.py`: `uv run --group live harness`. A Portal `Observer` on the robot's room; `DreamLink`
  maps the link onto portal-makermods' `dream.*` RPCs (names copied from its `src/utils/common.py`). Motion comes from
  the robot state: any value changing more than 0.02 counts as moving.
- `tests/test_relay.py`: the state machine with `FakeLink`.

## Rules

- Rubric: after "go", match mode may only call `set_task`, `resume`, `pause`. Anything that hands control to a human
  (`to_teleop`) stays behind `Config.practice_teleop`. Never name a later stage's objects in an earlier step's prompts.
- Portal: the harness is an observer and never sends actions. It never parks or powers down the arms (no brakes).
  Control changes only through Dream-1's own RPCs.
- Keep `livekit` imports inside `live.py` functions, so the relay and tests run without LiveKit (and on macOS).
- Never commit `.env` or anything under `data/`.

## Checks

```bash
uv run python -m unittest discover tests
```
