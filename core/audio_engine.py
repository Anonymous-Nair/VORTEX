from __future__ import annotations

import asyncio
import math
import time

import numpy as np
import pyaudio

INPUT_RATE = 16000
INPUT_CHANNELS = 1
INPUT_CHUNK = 320
SILENCE_THRESHOLD = 300.0


class AudioEngine:
    """Continuous 16-kHz mono microphone capture."""

    def __init__(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=INPUT_CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=INPUT_CHUNK,
        )
        self.running = False

    async def frames(self):
        self.running = True
        while self.running:
            raw = await asyncio.to_thread(
                self.stream.read,
                INPUT_CHUNK,
                exception_on_overflow=False,
            )
            capture_time = time.perf_counter()
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            energy = float(math.sqrt(float(np.mean(samples * samples))))
            yield raw, energy, capture_time

    async def read_calibration_frames(self, num_frames: int):
        out = []
        first_start = 0.0
        for i in range(num_frames):
            raw = await asyncio.to_thread(
                self.stream.read,
                INPUT_CHUNK,
                exception_on_overflow=False,
            )
            end = time.perf_counter()
            start = end - (INPUT_CHUNK / INPUT_RATE)
            if i == 0:
                first_start = start
            out.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32))

        if not out:
            return np.zeros(0, dtype=np.float32), first_start
        return np.concatenate(out), first_start

    def stop(self) -> None:
        self.running = False
        try:
            self.stream.stop_stream()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass
        try:
            self.audio.terminate()
        except Exception:
            pass

    @staticmethod
    def is_speech(energy: float) -> bool:
        return energy > SILENCE_THRESHOLD