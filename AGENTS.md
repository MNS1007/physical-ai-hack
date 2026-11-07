# Working on the harness

Read `README.md` first: the rubric math, how the processes fit, and the rules. Keep v0 small and readable.

## Layout

- `src/physical_ai_hack/stages.py`: the relay as data (`Step`: prompts, judge goal, `target`, `hold_s`,
  `timeout_s`) and the rubric's `score()`. Most tuning happens here.
- `src/physical_ai_hack/judge.py`: Claude vision judge. `Judge.__call__(step, frames, later)` returns
  `(Verdict, seconds)`. Structured outputs into the pydantic `Verdict`; a refusal or API error returns a not-met
  verdict, never an exception into the relay.
- `src/physical_ai_hack/relay.py`: the state machine, with no LiveKit in it. Inputs: `start(now)`,
  `on_verdict(verdict, asked_at, took_s)`, `tick(now, still_for_s)`. Output: calls on a `link`
  (`set_task`, `resume`, `pause`, `to_teleop`) and `Event`s to a log.
- `src/physical_ai_hack/live.py`: `uv run --group live harness`. A Portal `Observer` on the robot's room; `DreamLink`
  maps the link onto portal-makermods' `dream.*` RPCs (names copied from its `src/utils/common.py`).
- `src/physical_ai_hack/offline.py`: `uv run harness-judge`, the judge on photos.
- `tests/test_relay.py`: the state machine with `FakeLink` and scripted verdicts.

## Rules

- Rubric: after "go", match mode may only call `set_task`, `resume`, `pause`. Anything that hands control to a human
  (`to_teleop`) stays behind `Config.practice_teleop`. Never advance a step on a single verdict; never name a later
  stage's objects in an earlier step's prompts.
- Portal: the harness is an observer and never sends actions. It never parks or powers down the arms (they have no
  brakes). Control changes only through Dream-1's own RPCs.
- The relay must stay importable and testable without LiveKit or an API key: keep `livekit` imports inside
  `live.py` functions.
- Judge model defaults to `claude-opus-5-5` (`HARNESS_MODEL`). Don't use forced `tool_choice` with it (400).
- Never commit `.env` or anything under `data/`.

## Checks

```bash
uv run python -m unittest discover tests
uv run harness-judge --step unplug <photo.jpg>     # needs ANTHROPIC_API_KEY
```

## Next steps (where they go)

- **Voice "stop"/progress:** a LiveKit agent like portal's `src/voice_agent.py`; the relay's `announce` callback
  is the hook. Ask organizers whether voice is allowed in a match.
- **Per-stage checkpoints:** a `checkpoint` field on `Step`, and a way for `dream-1-tasks` to switch it (it reads
  `DREAM_CHECKPOINT` once at start today).
- **Better progress signal:** portal's `task_monitor.py` (TOPReward) as a second opinion on `count`.
- **Stage 3 / plug-ins:** new `Step`s, only if timed runs show they beat the time bonus (2 s = 1 point).
