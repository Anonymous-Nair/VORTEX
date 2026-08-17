from __future__ import annotations

import asyncio
import time

from .audio_engine import AudioEngine
from .barge_in import BargeInDetector
from .event_bus import EventBus
from .stt_engine import STTEngine
from .vad_engine import SpeechState, TurnDetector

GRACE_PERIOD_SECONDS = 0.5
MIN_SPEECH_SECONDS = 0.15


class LiveAudioSession:
    def __init__(
        self,
        events: EventBus,
        audio: AudioEngine | None = None,
        stt: STTEngine | None = None,
    ):
        self.events = events
        self.audio = audio or AudioEngine()
        self.stt = stt or STTEngine()
        self.turn = TurnDetector()
        self.barge = BargeInDetector()
        self.running = False
        self.speaking = False
        self._turn_buffer = bytearray()
        self._speech_started = None
        self._speech_seconds = 0.0
        self._pending_end_since: float | None = None

    async def run_calibration(self, tts_engine) -> dict:
        return await self.barge.calibrate(tts_engine, self.audio)

    async def run(self) -> None:
        self.running = True
        await self.events.publish("audio.ready")

        async for raw, energy, capture_time in self.audio.frames():
            if not self.running:
                break

            frame_seconds = len(raw) / 2 / 16000

            if self.speaking:
                try:
                    interrupted = self.barge.update(
                        raw,
                        energy,
                        capture_time,
                    )
                except Exception as exc:
                    print("[BARGE ERROR]", type(exc).__name__, exc)
                    interrupted = False

                if interrupted:
                    self.speaking = False
                    self.barge.reset()
                    self.turn.reset()
                    self._turn_buffer.clear()
                    self._speech_started = None
                    self._speech_seconds = 0.0
                    self._pending_end_since = None
                    await self.events.publish("assistant.interrupted")

                continue

            if self._pending_end_since is not None:
                self._turn_buffer.extend(raw)

                if energy >= self.turn.config.start_threshold:
                    self._pending_end_since = None
                    self.turn.resume()
                    self._speech_seconds += frame_seconds
                    continue

                if time.perf_counter() - self._pending_end_since < GRACE_PERIOD_SECONDS:
                    continue

                await self._finalize_turn()
                continue

            state = self.turn.update(energy, frame_seconds)

            if state == SpeechState.SPEAKING:
                if self._speech_started is None:
                    self._speech_started = time.perf_counter()
                    await self.events.publish("speech.started")
                self._turn_buffer.extend(raw)
                self._speech_seconds += frame_seconds

            elif state == SpeechState.ENDED:
                self._pending_end_since = time.perf_counter()

    async def _finalize_turn(self) -> None:
        audio_data = bytes(self._turn_buffer)
        speech_seconds = self._speech_seconds

        self._turn_buffer.clear()
        self.turn.reset()
        self._speech_started = None
        self._speech_seconds = 0.0
        self._pending_end_since = None

        await self.events.publish("speech.ended")

        if not audio_data or speech_seconds < MIN_SPEECH_SECONDS:
            return

        started = time.perf_counter()
        text = await self.stt.transcribe(audio_data)
        elapsed = time.perf_counter() - started

        if text:
            await self.events.publish(
                "stt.final",
                text=text,
                elapsed=elapsed,
            )

    def set_speaking(self, value: bool) -> None:
        value = bool(value)

        if self.speaking and not value:
            self.turn.reset()
            self.barge.reset()
            self._turn_buffer.clear()
            self._speech_started = None
            self._speech_seconds = 0.0
            self._pending_end_since = None

        if not self.speaking and value:
            self.barge.start_speaking()

        self.speaking = value

    def get_barge_start_callback(self):
        return self.barge.start_speaking

    def stop(self) -> None:
        self.running = False
        self.audio.stop()