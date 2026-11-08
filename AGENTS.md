# Working on the harness

Read `README.md` first: the rubric math, how it drives the station, and the rules. Keep it small and readable.

## Where we are

Zero-shot first: the team runs Dream-1 as it is (README, "Run: zero-shot first, and watch"), steps 1–4. Help them:

1. Give the exact command for each step; the person runs anything that moves the arms.
2. Keep the table of tries (prompt, worked?, seconds, how it failed) from what they report or from
   `data/runs/*/log.jsonl`, and suggest the next wording to try.
3. Once a prompt works reliably, write it and its seconds into `stages.py`, run the tests, and commit.

Two fine-tunes are ready (README, "Our fine-tuned models"): `dream-1-ft:355fafe9-all-v2` (unplug + stack, 1000
steps) and `dream-1-ft:355fafe9-unplug-v1`. Compare each against the base model with `harness eval` before the
match; pass the winner with `--checkpoint`.

## Layout

- `src/physical_ai_hack/stages.py`: the relay as data (`Step`: name, stage, prompt, seconds) and the rubric's
  `score()`. Most tuning happens here.
- `src/physical_ai_hack/station.py`: a stdlib client for the Dream station agent's HTTP API (`dreamcli robot up`,
  `robots.station_api` in the `dreamlabs` package): `status`, `start`, `prompt`, `stop`.
- `src/physical_ai_hack/harness.py`: the CLI (`uv run harness status|try|relay`) and `run_timeline()`, which takes
  the station, a clock and a sleep so tests can fake them.
- `tests/test_harness.py`: the timeline with `FakeStation` and a fake clock.
- `src/physical_ai_hack/station_plus.py`: Dream's agent (`robots.station.main`) with `GET /frame/<cam>.jpg`, `/frames`,
  `/arms` added (read-only, from `controller.frames()` / `arm_states()`).
- `src/physical_ai_hack/supervisor.py`: attempts (settle/timeout), `match()` (supervised / trust), `evaluate()`.
- `src/physical_ai_hack/judge.py`: `ClaudeJudge` (structured verdict), `VoiceJudge`, `KeyboardJudge`, `parse_word`.
- `src/physical_ai_hack/announce.py`: console, local TTS, LiveKit room (speech out, operator voice in via Whisper).
- `tests/test_supervisor.py`: the supervisor with a fake station, scripted judge, fake clock.
- `tools/dimos_judge.py`: Dimensional (dimos.models.vl) second opinion, own env via inline deps. `LAPTOP.md`: run from a laptop.
- `src/physical_ai_hack/convert/`: `.rrd` teleop recordings -> the kit's LeRobot v2.1 layout (`uv run --group convert
  convert inspect|run`). `rrd.py` reads, `mapping.py` is the YAML schema, `episode.py` the clock/vectors/frames,
  `writer.py` mirrors the kit's `dataset.py`. Recorder specifics live only in `configs/convert/*.yaml`.
- `tests/test_convert.py`: mapping checks and resampling on in-memory recordings.

## Rules

- Never run `harness try` or `harness relay` without `--dry-run`, `dreamcli robot run/up` without `--dry-run`,
  `dhk-collect` without `--dry-run`, or anything else that moves the arms yourself: give the person the command.
- Before the arms first move on a station, the person checks Dream's `--dry-run --view` page. Never confirm it for
  them and never run `python -m dream.input_check`.
- Rubric: after "go" the harness only calls `/start`, `/prompt`, `/status`, `/stop` and the read-only routes. Never name a later
  stage's objects in an earlier step's prompt.
- Never park, home or power down the arms from the harness; `/stop` leaves them holding.
- The core (`harness status/try/relay/match --trust`) is standard library only; extras live in dependency groups:
  `convert`, `judge` (anthropic), `livekit` (livekit, faster-whisper).
- A judge that cannot answer is never a pass; the supervised match never starts the cubes before the unplug passed.
- Never hardcode a recorder's entity names, units or order in code: put them in the mapping YAML.
- Never commit `.env`, API keys, recordings (`.rrd`) or anything under `data/`.

## Checks

```bash
uv run --group convert python -m unittest discover tests
```
