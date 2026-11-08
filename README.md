# Hill Climb harness (v0)

A small harness around **Dream-1** for the MakerMods Physical AI Hack (Nov 7–8, 2026). It runs the relay from
`hillclimb_rubric_v2` with no human input after "go": it tells Dream-1 to unplug a cable, then to stack the blocks, in
one run, then stops and tells your team to **call stop**.

```
Enter at GO ─▶ unplug (45 s) ─▶ stack (120 s) ─▶ stop: arms hold ─▶ ">>> CALL STOP"
              "pull the USB cable      "stack the blocks on
               out of the power strip"  top of each other"
```

It drives Dream's own station agent (`dreamcli robot up`) over its local HTTP API: `POST /start` with the first
prompt, `POST /prompt` to switch the task mid-run (no homing in between), `POST /stop` at the end. Standard library
only; no LiveKit, no extra install (the converter below has its own dependency group).

Start **zero-shot** (Dream-1 as it is). If you fine-tune later, pass `--checkpoint dream-1-ft:...`.

## Why this plan (the rubric's math)

If both stages pass, **score = 10 + 20 + (300 − T) / 2**, where T is when your teammate calls stop. Every 2 s costs
1 point, so extra unplugs (+5), extra blocks (+10, and a collapse loses the stack points) and balls (+5, and the
8-ball loses the match) rarely pay. Do the two stages and stop. Reliability first, then speed.

## Setup (on the station)

The station is the robot's Linux computer (`ssh -L 8080:localhost:8080 pah@<ip>`). Dream's library is in
`~/dream-env`, set up by the organizers ([Getting started](https://platform.dreamlabs.ai/docs/),
[the kit](https://github.com/dreamlabs-co/physical-ai-hack-2026)).

```bash
git clone https://github.com/MNS1007/physical-ai-hack ~/physical-ai-hack && cd ~/physical-ai-hack
uv run python -m unittest discover tests        # no robot needed
```

## Run: zero-shot first, and watch

We start with Dream-1 as it is (no fine-tune), find the prompts and timings that work on the real props, and only then
run the relay. Every command below runs **on the station**. Anything without `--dry-run` **moves the arms**: a person
runs it, with a hand on the e-stop and everyone out of reach.

**0. Log in** from your laptop. Forward the dashboard port, and the port the `--view` page prints in step 1 (reconnect
with it added if it is not forwarded yet):

```bash
ssh -L 8080:localhost:8080 -L <view-port>:localhost:<view-port> pah@<ip>
```

**1. Dry run with the view page (the arms do not move).** Dream's rule before the arms first move:

```bash
source ~/dream-env/bin/activate
dreamcli robot run --task "pull the USB cable out of the power strip" --dry-run --view
```

Open the page it prints and check, as a person (never an agent):

- every camera under its own name, upright, in natural colours, and changing as things move;
- joints in radians; each gripper 0 closed to 1 open;
- the prompt is the task.

Leave it 20–30 s and watch what the model wants to do, then Ctrl+C.

**2. Real runs, one task at a time.** Stop with Ctrl+C the moment it looks wrong.

```bash
dreamcli robot run --task "pull the USB cable out of the power strip"
dreamcli robot run --task "stack the blocks on top of each other"
```

Try 2–3 wordings per task (e.g. "unplug the USB cable", "pull the cable out of the outlet") and write each try down
in a table like this one:

| Prompt | Worked? | Seconds | How it failed (wrong object / missed grasp / stalled / knocked over) |
|---|---|---|---|
| pull the USB cable out of the power strip | | | |

**3. Repeat the best prompts with the harness.** It is faster to iterate: no homing between prompts. The station agent
owns the arms, so close `dhk-collect` and any `dreamcli robot run` first (one program at a time).

```bash
tmux new -A -s agent                                   # window 1: the station agent
source ~/dream-env/bin/activate && dreamcli robot up --view
```

```bash
cd ~/physical-ai-hack                                  # window 2: the harness
uv run harness status                                  # moves nothing
uv run harness try "<best unplug prompt>" -s 60        # one prompt, then stop
uv run harness try "<best stack prompt>" -s 120
```

**4. Put the winners in `src/physical_ai_hack/stages.py`** (prompt, and seconds = typical time plus a margin), then
the full relay:

```bash
uv run harness relay --dry-run                         # checks arms and cameras, prints the plan; moves nothing
uv run harness relay                                   # the match: Enter at GO
```

Ctrl+C stops any run; the arms then hold where they are, under power. Each harness run logs to
`data/runs/<time>/log.jsonl`.

## Tuning (`src/physical_ai_hack/stages.py`)

1. **Prompts.** Dream-1 is language-conditioned; wording changes behavior. Use `harness try` to compare a few on
   the real props, and write down what each one does.
2. **Seconds.** Time a few `try` runs of each prompt. Give each step its typical time plus a margin: too short and
   the stack prompt starts while the cable is still in (touching blocks before Stage 1 passes is a DQ); too long
   and you lose time bonus.
3. **Fine-tuned model.** `uv run harness --checkpoint dream-1-ft:<id> relay`.

## Rules the harness keeps

- After "go": no teleop, no keyboard, no human input. The run is one `/start`, prompt switches and one `/stop`.
- No skipped stages: a step's prompt never names a later stage's objects (a test checks it).
- The arms are never parked or powered down by the harness: `/stop` leaves them holding, under power (MakerMods arms
  have no brakes).

**Known limit:** v0 is a timeline: it cannot see whether the cable is out before it switches to the blocks. Set the
unplug seconds from your tries, and watch the first runs.

## Harness v1: rollouts, and the match supervised by a judge

Dream-1 never says it is done. The harness ends an attempt when the arms have moved and then stayed still
(`settle_s`, after `min_s`) or at `timeout_s`, stops the run (the arms hold), and asks a **judge** whether the step is
done. Steps, prompts (the training labels), judge questions and times are in `src/physical_ai_hack/stages.py`.

| Command | What |
|---|---|
| `harness eval unplug --trials 10 [--checkpoint dream-1-ft:...]` | per-task rollouts: you reset the scene, it runs one attempt, the judge scores it; success rate + time. Run per model to see whether the fine-tune helped |
| `harness match` | the rubric relay, **supervised**: each step retried until the judge passes it (≤ `max_attempts`, within `budget_s`); the cubes are never touched unless the unplug passed (DQ rule); "Call stop" at the end. `--extra-block` tries a 4th cube when the judge says it can |
| `harness match --trust` | baseline: no judge, no retries; a step ends when the arms settle |

Judges (`--judge`): `claude` (default; `--judge-model claude-sonnet-5-5` for speed), `voice` (the operator says
**done**, **retry** or **fail** in the LiveKit room; Whisper transcribes it on the station; the rubric says organizers
"will consider allowing voice input", so ask first), `keys` (typed, for testing).

Announcements: `--voice` speaks them on the machine; `--livekit` speaks them in a LiveKit room (`LIVEKIT_URL`,
`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM`=hill-climb in `.env`) and sends each as a data message;
join the room from a browser to hear "Task 1 unplug done", "All tasks done. Call stop."

### Run it

```bash
# on the station, window 1: Dream's agent plus camera/arm routes (instead of `dreamcli robot up`).
# Needs ~/.dreamlabs/station.yaml from `dreamcli setup` (check: ls ~/.dreamlabs/). dreamcli itself is in this repo's
# env on z1: `source ~/physical-ai-hack/.venv/bin/activate` (no ~/dream-env there).
uv sync --group convert --group judge --group livekit
uv run --group convert station-plus --config ~/.dreamlabs/station.yaml
# window 2 (ANTHROPIC_API_KEY in .env)
curl -s localhost:8091/frames; curl -s localhost:8091/arms                 # routes work; nothing moves
uv run --group judge harness judge-now unplug                              # verdict on the scene now; nothing moves
uv run --group judge harness match --dry-run                               # checks + one verdict; nothing moves
uv run --group judge harness eval unplug --trials 3                        # MOVES THE ARMS
uv run --group judge --group livekit harness --livekit match               # MOVES THE ARMS
```

Calibrate the judge offline first (no station; a few cents): last frames of finished demos should be "done", first
frames "not done":

```bash
uv run --group judge harness judge-image stack end-scene.jpg end-left_wrist.jpg
```

Each run logs to `data/runs/<time>/`: `log.jsonl`, every image the judge saw, every verdict, and `eval.json`.

## Our fine-tuned models

Both are servable (`dreamlabs/dream-1-beta` base, Metal EEF embodiment, cameras scene + both wrists). Pass one with
`--checkpoint`:

| Checkpoint | Job | Data | Steps | Final loss |
|---|---|---|---|---|
| `dream-1-ft:355fafe9-all-v2` | `ftj_a96b2af23d80eab1` | `all-v2`: 64 clean unplug + stack episodes (no ball) | 1000 | 0.0076 |
| `dream-1-ft:355fafe9-unplug-v1` | `ftj_05b4a80045fb1967` | `unplug-v1`: unplug episodes 1-16 | 500 | 0.0068 |

```bash
uv run --group judge harness --checkpoint dream-1-ft:355fafe9-all-v2 eval unplug --trials 5   # MOVES THE ARMS
uv run --group judge harness --checkpoint dream-1-ft:355fafe9-all-v2 eval stack --trials 5
uv run --group judge harness --checkpoint dream-1-ft:355fafe9-all-v2 match
```

Compare against the base model (no `--checkpoint`) and `unplug-v1` on the same trials. Continue a model with more data:
`finetune ... --resume-from dream-1-ft:355fafe9-all-v2` (same constant schedule, learning rate 1e-5, EEF). On macOS, if
the SDK fails with `CERTIFICATE_VERIFY_FAILED`: `export SSL_CERT_FILE=$(uv run python -m certifi)`.

## Fine-tuning (optional, later)

Follow the kit: collect with `dhk-collect` (Quest teleop, LeRobot v2.1), then on Metal arms `tools/add_eef.py --arm
metal` and `tools/finetune.py --arm metal`. A job is priced by steps ($200 for the default 1,000) and takes hours, so
queue it early (evening of day 1).

## Converting teleop recordings (.rrd) for fine-tuning

Our VR teleop records Rerun files (`.rrd`, Rerun 0.42), **one file per episode**. `convert` turns them into the
LeRobot v2.1 layout the Dream kit's recorder writes, so the kit's `tools/add_eef.py` and `tools/finetune.py` take the
result unchanged. Which recorded data fills which output is in a mapping file, not in the code:
`configs/convert/metal_right_arm.yaml` (right arm teleoperated; the left arm is held at its rest pose; cameras
`scene`, `left_wrist`, `right_wrist`).

```bash
uv sync --group convert                                   # rerun-sdk, PyAV, pyarrow, Dream's dataset checks
uv run --group convert convert inspect rec.rrd            # what a recording holds: entities, names, rates, ranges, cameras
uv run --group convert convert run recs/*.rrd ~/hackathon_data/team-stack \
    --map configs/convert/metal_right_arm.yaml --task "stack the blocks on top of each other"
```

- Inputs can mix both formats: `.rrd` files, folders of `.rrd` files, and datasets `dhk-collect` recorded (LeRobot
  v2.1, the kit's Quest teleop). All become one dataset, renumbered; `--task` replaces every episode's task text.
  Example: `convert run ~/hackathon_data/kit-stack ~/recs/vr-stack/ ~/hackathon_data/team-stack --map ...`
- Each `.rrd` becomes one episode, appended to the output folder (an existing folder must match).
- Every stream is sampled latest-at on a 30 fps clock, over the span where all of them are running.
- A recording missing a stream the mapping needs, or shorter than 1.6 s, is skipped with the reason.
- Frames are center-cropped to 16:9 and resized to 640x360, H.264, as the kit records.
- At the end it runs Dream's own dataset checks on the folder.

A new recorder or arm setup needs only a new mapping: `inspect` a recording, copy the YAML, change the entity and
scalar names, units (`unit: deg`), scales and fills.

## Tests

```bash
uv run python -m unittest tests.test_harness                      # the harness; standard library only
uv run --group convert python -m unittest discover tests          # everything, the converter included
```
