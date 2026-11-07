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
only; no LiveKit, no extra install.

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

## Run

**1. Check what the model sees, with nothing moving** (Dream's rule before the arms first move). In the kit's
`dreamcli`:

```bash
source ~/dream-env/bin/activate
dreamcli robot run --task "stack the blocks on top of each other" --dry-run --view
```

Open the page it prints and check: every camera under its own key, upright, natural colours, live; joints in
radians; each gripper 0 closed to 1 open. A person confirms this, not an agent.

**2. Start the station agent** in its own tmux window (it owns the arms; close `dhk-collect` first; one program at a
time):

```bash
tmux new -A -s agent
source ~/dream-env/bin/activate && dreamcli robot up --view
```

**3. Drive it** from a second window. `try` and `relay` **move the arms**: a hand on the e-stop.

```bash
cd ~/physical-ai-hack
uv run harness status                                              # moves nothing
uv run harness try "pull the USB cable out of the power strip" -s 45   # one prompt, then stop
uv run harness relay --dry-run                                     # checks the station, prints the plan
uv run harness relay                                               # the match: Enter at GO
```

Ctrl+C stops the run at any time; the arms then hold where they are, under power. Each run logs to
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

## Fine-tuning (optional, later)

Follow the kit: collect with `dhk-collect` (Quest teleop, LeRobot v2.1), then on Metal arms `tools/add_eef.py --arm
metal` and `tools/finetune.py --arm metal`. A job is priced by steps ($200 for the default 1,000) and takes hours, so
queue it early (evening of day 1).

## Tests

```bash
uv run python -m unittest discover tests     # the timeline with a fake station; no robot needed
```
