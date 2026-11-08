"""Announcements: what the harness says out loud and where it says it.

- Console: always.
- Voice (`--voice`): the machine's own text-to-speech (`espeak-ng` on Linux, `say` on macOS), in the background.
- LiveKit (`--livekit`, the `livekit` dependency group): joins the room `LIVEKIT_ROOM` (default `hill-climb`) as
  `harness` and publishes each announcement as speech on an audio track and as a JSON data message (topic
  `harness`). Anyone in the room hears it: open https://meet.livekit.io with your project URL and a token, or any
  LiveKit client. Needs LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.

Announcing never blocks the run: speech is queued and spoken on a worker thread.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import wave

SAMPLE_RATE = 22050
VOICE_RMS = 0.02  # loudness that counts as speech (0-1 full scale)


def synthesize(text: str) -> tuple[bytes, int] | None:
    """Speech as 16-bit mono PCM and its rate, from espeak-ng or macOS say; None without either."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "say.wav")
        if shutil.which("espeak-ng"):
            command = ["espeak-ng", "-w", path, text]
        elif shutil.which("say"):
            command = ["say", "-o", path, "--data-format=LEI16@22050", "--file-format=WAVE", text]
        else:
            return None
        if subprocess.run(command, capture_output=True).returncode != 0:
            return None
        with wave.open(path) as w:
            if w.getsampwidth() != 2:
                return None
            pcm = w.readframes(w.getnframes())
            if w.getnchannels() == 2:  # keep the left channel
                import array

                a = array.array("h", pcm)
                pcm = a[0::2].tobytes()
            return pcm, w.getframerate()


class Announcer:
    def __init__(self, *, voice: bool = False, livekit: bool = False) -> None:
        self.voice = voice
        self._queue: queue.Queue[tuple[str, dict] | None] = queue.Queue()
        self._livekit = _LiveKit() if livekit else None
        self.livekit = self._livekit
        self._thread = threading.Thread(target=self._run, daemon=True, name="announce")
        self._thread.start()

    def __call__(self, text: str, **event) -> None:
        print(f"\n>>> {text}\n", flush=True)
        self._queue.put((text, event))

    def close(self, wait_s: float = 8.0) -> None:
        """Lets queued speech finish (up to wait_s), then leaves the room."""
        self._queue.put(None)
        self._thread.join(wait_s)
        if self._livekit:
            self._livekit.close()

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            text, event = item
            if self._livekit:
                self._livekit.send(text, event)
            elif self.voice:
                if shutil.which("espeak-ng"):
                    subprocess.run(["espeak-ng", text], capture_output=True)
                elif shutil.which("say"):
                    subprocess.run(["say", text], capture_output=True)


class _LiveKit:
    """A LiveKit participant on its own event loop thread: speech on an audio track, events as data messages."""

    def __init__(self) -> None:
        import asyncio

        from dotenv import load_dotenv

        load_dotenv()
        for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
            if not os.environ.get(key):
                raise RuntimeError(f"--livekit needs {key} (environment or .env)")
        self.room_name = os.environ.get("LIVEKIT_ROOM", "hill-climb")
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="livekit").start()
        asyncio.run_coroutine_threadsafe(self._connect(), self.loop).result(timeout=20)
        print(f"[livekit] in room {self.room_name!r} as 'harness'")

    async def _connect(self) -> None:
        from livekit import api, rtc

        token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
                 .with_identity("harness").with_name("Hill Climb harness")
                 .with_grants(api.VideoGrants(room_join=True, room=self.room_name, can_publish=True,
                                              can_publish_data=True)).to_jwt())
        self.room = rtc.Room()

        @self.room.on("track_subscribed")
        def _on_track(track, publication, participant) -> None:
            import asyncio

            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(self._listen(track, participant.identity))

        await self.room.connect(os.environ["LIVEKIT_URL"], token)
        self.source = rtc.AudioSource(SAMPLE_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("announcements", self.source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, options)

    async def _say(self, text: str, event: dict) -> None:
        from livekit import rtc

        payload = json.dumps({"text": text, "t": time.time(), **event}).encode()
        await self.room.local_participant.publish_data(payload, reliable=True, topic="harness")
        speech = synthesize(text)
        if speech is None:
            return
        pcm, rate = speech
        if rate != SAMPLE_RATE:  # resample by nearest sample; good enough for speech
            import array

            src = array.array("h", pcm)
            n = int(len(src) * SAMPLE_RATE / rate)
            pcm = array.array("h", (src[int(i * rate / SAMPLE_RATE)] for i in range(n))).tobytes()
        chunk = SAMPLE_RATE // 50  # 20 ms
        for start in range(0, len(pcm) // 2, chunk):
            piece = pcm[start * 2:(start + chunk) * 2]
            samples = len(piece) // 2
            frame = rtc.AudioFrame.create(SAMPLE_RATE, 1, samples)
            frame.data.cast("B")[:len(piece)] = piece
            await self.source.capture_frame(frame)
        await self.source.wait_for_playout()

    # --- listening: the operator's voice, for the voice judge ---------------------------------------------------

    _segments: "queue.Queue | None" = None
    _heard: "queue.Queue | None" = None

    async def _listen(self, track, who: str) -> None:
        """Cuts the operator's audio into utterances by loudness and hands them to the transcriber."""
        import numpy as np
        from livekit import rtc

        self._start_transcriber()
        print(f"[livekit] listening to {who}")
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        speech, silence_ms, voiced_ms = [], 0, 0
        async for event in stream:
            pcm = np.frombuffer(bytes(event.frame.data), dtype=np.int16)
            loud = np.sqrt(np.mean((pcm.astype(np.float32) / 32768) ** 2)) > VOICE_RMS
            ms = 1000 * len(pcm) // 16000
            if loud:
                speech.append(pcm)
                voiced_ms += ms
                silence_ms = 0
            elif speech:
                speech.append(pcm)
                silence_ms += ms
                if silence_ms >= 600:
                    if voiced_ms >= 250:
                        self._segments.put(np.concatenate(speech).astype(np.float32) / 32768)
                    speech, silence_ms, voiced_ms = [], 0, 0

    def _start_transcriber(self) -> None:
        if self._segments is not None:
            return
        self._segments, self._heard = queue.Queue(), queue.Queue()

        def work() -> None:
            from faster_whisper import WhisperModel

            model = WhisperModel(os.environ.get("WHISPER_MODEL", "base.en"), device="cpu", compute_type="int8")
            while True:
                audio = self._segments.get()
                segments, _ = model.transcribe(audio, language="en", beam_size=1)
                text = " ".join(s.text for s in segments).strip()
                if text:
                    self._heard.put(text)

        threading.Thread(target=work, daemon=True, name="whisper").start()

    def heard(self, timeout: float) -> str | None:
        """The next thing the operator said, or None after timeout."""
        self._start_transcriber()
        try:
            return self._heard.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Forgets what was said before now."""
        self._start_transcriber()
        while not self._heard.empty():
            self._heard.get_nowait()

    def send(self, text: str, event: dict) -> None:
        import asyncio

        try:
            asyncio.run_coroutine_threadsafe(self._say(text, event), self.loop).result(timeout=60)
        except Exception as error:  # noqa: BLE001 - an announcement must never stop the run
            print(f"[livekit] announcement failed: {error}")

    def close(self) -> None:
        import asyncio

        try:
            asyncio.run_coroutine_threadsafe(self.room.disconnect(), self.loop).result(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
