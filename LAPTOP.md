# From a laptop (not the station)

The station (z1) runs only Dream's agent (`station-plus`). Everything else can run from a laptop: the harness talks
to the agent through an SSH tunnel; LiveKit and Dimensional talk to their clouds. Whoever runs a command that moves
the arms stands at the e-stop, whichever machine sends it.

## 0. Once (5 min)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv, if missing
gh repo clone MNS1007/physical-ai-hack && cd physical-ai-hack   # private: ask Nithin to add you
uv sync --group judge --group livekit
cp /dev/null .env && chmod 600 .env && nano .env       # paste the lines below, real values, no quotes
```

```
ANTHROPIC_API_KEY=...          # Claude judge
LIVEKIT_URL=wss://....livekit.cloud
LIVEKIT_API_KEY=API...
LIVEKIT_API_SECRET=...         # the real secret, not the dots the dashboard shows
LIVEKIT_ROOM=hill-climb
MOONDREAM_API_KEY=...          # Dimensional judge (free key at moondream.ai); or OPENAI_API_KEY
```

## 1. Reach the station (one terminal, leave it open)

```bash
ssh -N -L 8091:127.0.0.1:8091 fo@z1
```

Then, in another terminal: `uv run harness status` (nothing moves). Every README command now works from here.

## 2. LiveKit voice (nothing moves)

```bash
uv run --group livekit harness listen
```

Join room `hill-climb` from a phone: meet.livekit.io, tab Custom, the `LIVEKIT_URL` and a token (2 h) from:

```bash
uv run --group livekit python -c "
import os, datetime; from dotenv import load_dotenv; from livekit import api; load_dotenv()
print(api.AccessToken(os.environ['LIVEKIT_API_KEY'], os.environ['LIVEKIT_API_SECRET']).with_identity('phone').with_ttl(datetime.timedelta(hours=2)).with_grants(api.VideoGrants(room_join=True, room=os.environ.get('LIVEKIT_ROOM','hill-climb'))).to_jwt())"
```

Say "Start." / "Done." / "Next task." / "Retry." / "Fail." with a pause; each prints `heard ... -> command`.
Mute between commands: only the last short phrase (4 words or fewer) counts.

## 3. Dimensional second opinion (nothing moves)

```bash
uv run tools/dimos_judge.py stack --live           # the station's scene camera now (needs the tunnel)
uv run tools/dimos_judge.py unplug --live
uv run tools/dimos_judge.py stack some-image.jpg   # a saved image
```

First run installs dimos into its own environment (about 1 minute, 1.7 GB). Prints one JSON line: `done`, what
Dimensional's model detected and counted, its answer. Compare with Claude's: `uv run --group judge harness
judge-now stack`.

## 4. The match (MOVES THE ARMS)

```bash
uv run --group judge harness --checkpoint dreamlabs/dream-1-beta match --dry-run
uv run --group judge --group livekit harness --livekit --checkpoint dreamlabs/dream-1-beta match
uv run --group judge --group livekit harness --checkpoint dreamlabs/dream-1-beta match --judge voice
```
