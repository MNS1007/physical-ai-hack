# Hill Climb harness (v0)

A "System 2" harness around **Dream-1** for the MakerMods Physical AI Hack (Nov 7–8, 2026). It runs the relay from
`hillclimb_rubric_v2` with no human input after "go": it tells Dream-1 one step at a time, watches the overhead
camera with Claude, retries when Dream-1 stalls, and tells your team when to call **stop**.

```
          go
           │
   ┌───────▼────────┐  judge: a cable that started in the hub is out,     ┌──────────────────┐
   │ 1. UNPLUG      │──── held ≥ 2 s (≥ 2 verdicts) ─────────────────────▶ │ 2. STACK         │
   │ "pull the USB  │                                                     │ "stack the blocks│
   │  cable out…"   │◀─┐ stuck: pause → next prompt → resume              │  on top of…"     │
   └────────────────┘  │ (new Dream-1 episode; never teleop in a match)   └────────┬─────────┘
                       │                                                           │ judge: ≥ 3 layers, gripper
                       └───────────────────────────────────────────────────────────┤ clear, held ≥ 3 s
                                                                                   ▼
                                                              pause Dream-1 (arms hold) → "CALL STOP NOW"
```

## Why this plan (the rubric's math)

If both stages pass, **score = 10 + 20 + (300 − T) / 2**, where T is when your teammate calls stop. Every 2 s costs
1 point, so extra unplugs (+5), extra blocks (+10, and a collapse loses the stack points) and balls (+5, and the
8-ball loses the match) rarely pay. The harness does the two stages and stops. Reliability first, then speed:
EV = P(pass both) × (30 + (300 − T) / 2).

## How it fits

| Process | Where | What |
|---|---|---|
| `uv run robot` | robot host (Linux) | [portal-makermods](https://github.com/livekit-examples/portal-makermods): drives the Metal arms |
| `uv run dream-1-tasks` | robot host | portal-makermods (branch `dc/voice`): Dream-1 with a task set over RPC |
| `uv run --group live harness` | any Linux host in the room | **this repo**: a Portal observer that calls `dream.set_task`, `dream.resume`, `dream.pause` |

The harness is an observer, like portal's task monitor: it never sends actions, so only Dream-1 moves the arms.
**Don't also run portal's `task-monitor`** (it pauses Dream-1 on its own verdicts), except with `--dry-run`.

## Setup

```bash
git clone https://github.com/MNS1007/physical-ai-hack && cd physical-ai-hack
git clone -b dc/voice https://github.com/livekit-examples/portal-makermods ../portal-makermods
uv sync --group live                 # live group: Linux only (livekit-portal has no macOS < 26 wheel)
cp .env.example .env                 # ANTHROPIC_API_KEY, plus the LiveKit values portal uses
```

`.env` (this repo, or portal's `.env`, which the harness also reads):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | the judge's key |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM` | — | same values as portal's peers |
| `HARNESS_MODEL` | `claude-opus-5-5` | judge model; try `claude-haiku-5-5` if latency hurts, after checking accuracy |
| `HARNESS_EFFORT` | `low` | judge effort |
| `HARNESS_CAMERAS` | `overhead_camera` | portal.yaml video tracks the judge sees, comma-separated |
| `HARNESS_JUDGE_EVERY_S` | `1.5` | seconds between judge calls (one in flight at a time) |
| `PORTAL_DIR` | `../portal-makermods` | portal checkout, for `portal.yaml` |

## Run

1. **Without the robot (works on a Mac): judge photos.** Take phone photos of your table (hub plugged, one cable
   out, 2 blocks, 3 blocks with the gripper away) and check the verdicts and latency:
   ```bash
   uv run harness-judge --step unplug photos/*.jpg
   uv run harness-judge --step stack photos/*.jpg
   ```
2. **Dry run on the robot.** Start `uv run robot`, then `uv run dream-1-tasks` in portal-makermods, then:
   ```bash
   uv run --group live harness --dry-run
   ```
   Drive Dream-1 yourself (`c` in its terminal) and watch whether the judge's verdicts match what you see.
3. **Practice runs** (collects HG-DAgger corrections: on stuck it hands the arms to the teleoperator, and portal's
   recorder keeps the take):
   ```bash
   uv run --group live harness --practice
   ```
4. **Match mode.** Arms at rest, props placed, press Enter at "go":
   ```bash
   uv run --group live harness
   ```
   When it prints `CALL STOP NOW`, your designated teammate calls stop.

Every run logs to `data/harness_runs/<time>/`: `log.jsonl` (each task, verdict, retry, pass, DQ warning and the
final T and points) and the frames the judge saw. Paste the `passed` and `finished` lines into the eval sheet.

## Tuning (in order of payoff)

1. **Prompts** (`src/physical_ai_hack/stages.py`, `Step.prompts`). Dream-1 is language-conditioned; wording changes
   behavior. The first prompt runs first; each retry moves to the next. Put the best one first.
2. **Judge wording and holds** (`Step.goal`, `hold_s`). A false "cable out" moves the robot on to the blocks early, a
   DQ risk. Keep holds ≥ 2 verdicts. The stack's 3 s hold is the rubric's.
3. **Timeouts** (`Step.timeout_s`, `Config.still_s`). Too short and Dream-1 never finishes a slow grasp; too long
   and a stuck arm wastes the clock.
4. **Fine-tuned checkpoints.** Set `DREAM_CHECKPOINT=dream-1-ft:<id>-<suffix>` for `dream-1-tasks` (see
   `dream/fine_tuning.py`; `lr_schedule="cosine"` for one task).
5. **Speed.** portal's `DREAM_MAX_JOINT_RAD_S` (default 0.5) caps joint speed: raise it only once runs are reliable.

## Rules the harness keeps

- After "go": no teleop, no keyboard, no human input. Match mode only calls `set_task`, `resume` and `pause`;
  `to_teleop` exists only behind `--practice`.
- No skipped stages: a step's prompts never name a later stage's objects, and a step ends only on met verdicts held
  for `hold_s`. The judge flags any contact with a later stage's objects as `dq_risk` in the log.
- The Metal arms have no brakes: the harness never parks or powers them down. At the end it pauses Dream-1, and the
  robot holds the arms under power.

## Tests

```bash
uv run python -m unittest discover tests     # the relay state machine with a fake Dream-1; no robot, no API
```

## Not in v0

Voice (LiveKit agent) to say "stop" or announce progress, per-stage fine-tuned checkpoints swapped mid-run, the
Dimensional OS agent harness, Stage 3 balls, and plug-ins. See `AGENTS.md` for where each would go.
