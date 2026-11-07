# Working on the harness

Read `README.md` first: the rubric math, how it drives the station, and the rules. Keep it small and readable.

## Where we are

Zero-shot first: the team runs Dream-1 as it is (README, "Run: zero-shot first, and watch"), steps 1–4. Help them:

1. Give the exact command for each step; the person runs anything that moves the arms.
2. Keep the table of tries (prompt, worked?, seconds, how it failed) from what they report or from
   `data/runs/*/log.jsonl`, and suggest the next wording to try.
3. Once a prompt works reliably, write it and its seconds into `stages.py`, run the tests, and commit.

Fine-tuning comes later and only if zero-shot falls short (README, "Fine-tuning").

## Layout

- `src/physical_ai_hack/stages.py`: the relay as data (`Step`: name, stage, prompt, seconds) and the rubric's
  `score()`. Most tuning happens here.
- `src/physical_ai_hack/station.py`: a stdlib client for the Dream station agent's HTTP API (`dreamcli robot up`,
  `robots.station_api` in the `dreamlabs` package): `status`, `start`, `prompt`, `stop`.
- `src/physical_ai_hack/harness.py`: the CLI (`uv run harness status|try|relay`) and `run_timeline()`, which takes
  the station, a clock and a sleep so tests can fake them.
- `tests/test_harness.py`: the timeline with `FakeStation` and a fake clock.

## Rules

- Never run `harness try` or `harness relay` without `--dry-run`, `dreamcli robot run/up` without `--dry-run`,
  `dhk-collect` without `--dry-run`, or anything else that moves the arms yourself: give the person the command.
- Before the arms first move on a station, the person checks Dream's `--dry-run --view` page. Never confirm it for
  them and never run `python -m dream.input_check`.
- Rubric: after "go" the harness only calls `/start` once, `/prompt`, `/status` and `/stop`. Never name a later
  stage's objects in an earlier step's prompt.
- Never park, home or power down the arms from the harness; `/stop` leaves them holding.
- Standard library only. Never commit `.env`, API keys, or anything under `data/`.

## Checks

```bash
uv run python -m unittest discover tests
```
