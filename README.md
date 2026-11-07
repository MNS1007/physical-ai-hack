# Hill Climb harness (v0)

A small harness around **Dream-1** for the MakerMods Physical AI Hack (Nov 7–8, 2026). It runs the relay from
`hillclimb_rubric_v2` with no human input after "go": it gives Dream-1 one step at a time, retries with a new prompt
when a step takes too long, and tells your team when the stack is done so they can call **stop**.

```
 go ─▶ 1. UNPLUG ── Dream-1 moved, then still 4 s ──▶ 2. STACK ── moved, then still 4 s ──▶ pause → "CALL STOP"
          │  "pull the USB cable out of the hub"          │  "stack the blocks on top of each other"
          └─ 60 s and not done: pause → next prompt ─┘     └─ 90 s: pause → next prompt
```

There is no camera judge: a step counts as finished when Dream-1 has moved and then stayed still for `settle_s`
(it stops when it thinks it is done). **Your stop-caller is the check**: look at the stack, and call stop once it
has stood 3 s.

## Why this plan (the rubric's math)

If both stages pass, **score = 10 + 20 + (300 − T) / 2**, where T is when your teammate calls stop. Every 2 s costs
1 point, so extra unplugs (+5), extra blocks (+10, and a collapse loses the stack points) and balls (+5, and the
8-ball loses the match) rarely pay. Do the two stages and stop. Reliability first, then speed.

## How it fits

| Process | Where | What |
|---|---|---|
| `uv run robot` | robot host (Linux) | [portal-makermods](https://github.com/livekit-examples/portal-makermods): drives the Metal arms |
| `uv run dream-1-tasks` | robot host | portal-makermods (branch `dc/voice`): Dream-1 with a task set over RPC |
| `uv run --group live harness` | any Linux host in the room | **this repo**: a Portal observer that calls `dream.set_task`, `dream.resume`, `dream.pause` |

The harness is an observer, like portal's task monitor: it never sends actions, so only Dream-1 moves the arms.
Don't also run portal's `task-monitor` (it pauses Dream-1 on its own), except with `--dry-run`.

## Setup

```bash
git clone https://github.com/MNS1007/physical-ai-hack && cd physical-ai-hack
git clone -b dc/voice https://github.com/livekit-examples/portal-makermods ../portal-makermods
uv sync --group live      # Linux only: livekit-portal has no wheel for macOS before 26
cp .env.example .env      # the same LiveKit values portal uses (or rely on portal's .env)
```

| Variable | Default | Meaning |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM` | — | same values as portal's peers |
| `PORTAL_DIR` | `../portal-makermods` | portal checkout, for `portal.yaml` and its `.env` |

## Run

Start `uv run robot`, then `uv run dream-1-tasks` in portal-makermods. Then:

```bash
uv run --group live harness --dry-run    # logs what it would do; Dream-1 is never told anything
uv run --group live harness --practice   # on timeout, hands the arms to the teleoperator (HG-DAgger corrections)
uv run --group live harness              # match mode: arms at rest, props placed, press Enter at "go"
```

Each run logs to `data/harness_runs/<time>/log.jsonl`: every task, retry, finished step and the final time.

## Tuning (`src/physical_ai_hack/stages.py`)

1. **Prompts** (`Step.prompts`). Dream-1 is language-conditioned; wording changes behavior. The first runs first;
   each retry moves to the next. Put the best one first.
2. **Times.** `min_s`: never finish a step before this. `settle_s`: how long Dream-1 must be still to count as
   done. `timeout_s`: when to give up on a prompt. Set them from your first timed runs.
3. **Fine-tuned checkpoint.** Start `dream-1-tasks` with `DREAM_CHECKPOINT=dream-1-ft:<id>-<suffix>`.
4. **Speed.** portal's `DREAM_MAX_JOINT_RAD_S` (default 0.5) caps joint speed: raise it only once runs are reliable.

## Rules the harness keeps

- After "go": no teleop, no keyboard, no human input. Match mode only calls `set_task`, `resume`, `pause`;
  `to_teleop` exists only behind `--practice`.
- No skipped stages: a step's prompts never name a later stage's objects.
- The Metal arms have no brakes: the harness never parks or powers them down. At the end it pauses Dream-1 and the
  robot holds the arms under power.

**Known limit:** without a judge the harness cannot see whether the cable is really out before it moves on to the
blocks. If Dream-1 settles without unplugging, the stack prompt starts anyway, and touching blocks before Stage 1
passes is a DQ. Watch the first runs; if this happens, raise `min_s`/`settle_s` for unplug.

## Tests

```bash
uv run python -m unittest discover tests     # the relay with a fake Dream-1; no robot needed
```
