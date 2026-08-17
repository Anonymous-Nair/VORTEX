from __future__ import annotations

import asyncio
import queue
import re
import threading
import time
from typing import Callable, NamedTuple

import numpy as np
import pyaudio
import requests
from scipy.signal import resample_poly

TTS_URL = "http://127.0.0.1:8892/tts"
SAMPLE_RATE = 24000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
PREBUFFER_BYTES = 48000
NETWORK_CHUNK = 16384
PLAYBACK_CHUNK = 8192
REFERENCE_RATE = 16000
RESAMPLE_UP = 2
RESAMPLE_DOWN = 3
HISTORY_SAMPLES = int(REFERENCE_RATE * 0.75)


def sanitize_tts_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[`*_~]+", "", text)
    text = re.sub(
        r"[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]",
        "",
        text,
    )
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    return re.sub(r"\s+", " ", text).strip()


class TimestampedChunk(NamedTuple):
    pcm: np.ndarray
    acoustic_start: float
    acoustic_end: float
    num_samples: int


class PlaybackReferenceBuffer:
    """Chronological 16-kHz TTS reference history."""

    def __init__(self, max_samples: int = HISTORY_SAMPLES):
        self.max_samples = max_samples
        self.chunks: list[TimestampedChunk] = []
        self.total = 0
        self.lock = threading.Lock()

    def push(self, chunk: TimestampedChunk) -> None:
        with self.lock:
            self.chunks.append(chunk)
            self.total += chunk.num_samples
            while self.total > self.max_samples and len(self.chunks) > 1:
                old = self.chunks.pop(0)
                self.total -= old.num_samples

    def get_recent(self, samples: int) -> np.ndarray:
        samples = max(1, int(samples))
        with self.lock:
            if not self.chunks or self.total <= 0:
                return np.zeros(samples, dtype=np.float32)
            parts = []
            remaining = samples
            for chunk in reversed(self.chunks):
                if remaining <= 0:
                    break
                take = min(remaining, chunk.num_samples)
                if take > 0:
                    parts.append(chunk.pcm[-take:])
                    remaining -= take
            if not parts:
                return np.zeros(samples, dtype=np.float32)
            out = np.concatenate(list(reversed(parts)))
        if len(out) < samples:
            out = np.pad(out, (samples - len(out), 0))
        return out[-samples:].copy()

    def clear(self) -> None:
        with self.lock:
            self.chunks.clear()
            self.total = 0


playback_reference = PlaybackReferenceBuffer()


def generate_mls(order: int = 14) -> np.ndarray:
    n = (1 << order) - 1
    out = np.empty(n, dtype=np.float32)
    reg = 1
    taps = (14, 13, 12, 2)
    for i in range(n):
        out[i] = 1.0 if (reg & 1) else -1.0
        feedback = 0
        for tap in taps:
            feedback ^= (reg >> (tap - 1)) & 1
        reg = (reg >> 1) | (feedback << (order - 1))
    return out


class SpeechJob:
    def __init__(self, text: str, generation_id: int, playback_started_cb: Callable[[], None] | None = None):
        self.text = text
        self.generation_id = generation_id
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.playback_started_cb = playback_started_cb


class TTSEngine:
    """Single-worker streaming TTS with a playback-reference buffer."""

    def __init__(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.output = self.audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=2048,
        )
        self.jobs: queue.Queue[SpeechJob | None] = queue.Queue()
        self.running = True
        self.generation_id = 0
        self._cb: Callable[[], None] | None = None
        self._cursor = 0.0
        self._cursor_init = False
        self._hw_delay = 0.0
        self.worker = threading.Thread(
            target=self._worker,
            name="VortexTTSWorker",
            daemon=True,
        )
        self.worker.start()

    def get_reference_buffer(self):
        return playback_reference

    def set_playback_started_callback(self, cb: Callable[[], None]) -> None:
        self._cb = cb

    def set_hardware_delay(self, seconds: float) -> None:
        self._hw_delay = max(0.0, float(seconds))

    def _push_reference(self, chunk: bytes) -> None:
        if len(chunk) < 2:
            return
        samples_24 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        samples_16 = resample_poly(samples_24, RESAMPLE_UP, RESAMPLE_DOWN).astype(np.float32)
        n = len(samples_16)
        if n <= 0:
            return
        duration = n / REFERENCE_RATE
        now = time.perf_counter()
        if not self._cursor_init:
            self._cursor = now
            self._cursor_init = True
        acoustic_start = self._cursor + self._hw_delay
        acoustic_end = acoustic_start + duration
        playback_reference.push(TimestampedChunk(samples_16, acoustic_start, acoustic_end, n))
        self._cursor += duration

    def play_calibration_signal(self, signal: np.ndarray) -> float:
        raw = (np.clip(np.asarray(signal, dtype=np.float32), -0.25, 0.25) * 32767.0).astype(np.int16).tobytes()
        first = None
        offset = 0
        while offset < len(raw):
            end = min(offset + PLAYBACK_CHUNK, len(raw))
            chunk = raw[offset:end]
            if len(chunk) % 2:
                chunk = chunk[:-1]
            if chunk:
                if first is None:
                    first = time.perf_counter()
                self.output.write(chunk)
            offset = end
        return first if first is not None else time.perf_counter()

    async def speak(self, text: str, generation_id: int | None = None) -> bool:
        text = sanitize_tts_text(text)
        if not text:
            return False
        if generation_id is None:
            generation_id = self.generation_id
        job = SpeechJob(text, generation_id, self._cb)
        self.jobs.put(job)
        await asyncio.to_thread(job.done.wait)
        return generation_id == self.generation_id

    def _worker(self) -> None:
        while self.running:
            job = self.jobs.get()
            if job is None:
                break
            try:
                if job.generation_id == self.generation_id:
                    self._run_job(job)
                else:
                    job.cancelled.set()
            except Exception as exc:
                print("[TTS WORKER ERROR]", type(exc).__name__, exc)
            finally:
                job.done.set()
                self.jobs.task_done()

    def _run_job(self, job: SpeechJob) -> None:
        start = time.perf_counter()
        print("[TTS START]", job.text)
        response = None
        reader = None
        q: queue.Queue[bytes | None] = queue.Queue()
        done = threading.Event()
        interrupted = False
        started = False
        try:
            response = requests.post(
                TTS_URL,
                json={"text": job.text, "language": "English"},
                timeout=120,
                stream=True,
            )
            response.raise_for_status()

            def read():
                try:
                    for chunk in response.iter_content(chunk_size=NETWORK_CHUNK):
                        if not chunk:
                            continue
                        if job.cancelled.is_set() or job.generation_id != self.generation_id:
                            break
                        if len(chunk) % 2:
                            chunk = chunk[:-1]
                        if chunk:
                            q.put(chunk)
                except Exception as exc:
                    if not job.cancelled.is_set():
                        print("[TTS STREAM ERROR]", type(exc).__name__, exc)
                finally:
                    done.set()
                    q.put(None)

            reader = threading.Thread(target=read, name="VortexTTSReader", daemon=True)
            reader.start()
            buf = bytearray()

            while len(buf) < PREBUFFER_BYTES:
                if job.cancelled.is_set() or job.generation_id != self.generation_id:
                    interrupted = True
                    break
                x = q.get()
                if x is None:
                    break
                buf.extend(x)

            if buf and not interrupted:
                print(f"[TTS FIRST AUDIO] {time.perf_counter() - start:.3f}s")

            while not interrupted and job.generation_id == self.generation_id:
                if buf:
                    n = min(len(buf), PLAYBACK_CHUNK)
                    n -= n % 2
                    if n > 0:
                        chunk = bytes(buf[:n])
                        del buf[:n]
                        self._push_reference(chunk)
                        if not started:
                            started = True
                            if job.playback_started_cb:
                                try:
                                    job.playback_started_cb()
                                except Exception:
                                    pass
                        try:
                            self.output.write(chunk)
                        except Exception as exc:
                            print("[TTS WRITE ERROR]", type(exc).__name__, exc)
                            interrupted = True
                            break
                        continue

                if done.is_set() and q.empty():
                    break
                try:
                    x = q.get(timeout=0.10)
                except queue.Empty:
                    continue
                if x is not None:
                    buf.extend(x)

            if job.cancelled.is_set() or job.generation_id != self.generation_id:
                interrupted = True
            if interrupted:
                print("[TTS INTERRUPTED]")

        except requests.RequestException as exc:
            if not job.cancelled.is_set():
                print("[TTS REQUEST ERROR]", type(exc).__name__, exc)
        except Exception as exc:
            if not job.cancelled.is_set():
                print("[TTS ERROR]", type(exc).__name__, exc)
        finally:
            job.cancelled.set()
            if response:
                try:
                    response.close()
                except Exception:
                    pass
            if reader and reader.is_alive():
                reader.join(timeout=1.0)
            if interrupted:
                try:
                    self.output.stop_stream()
                    self.output.close()
                    self.output = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        output=True,
                        frames_per_buffer=2048,
                    )
                    playback_reference.clear()
                    self._cursor_init = False
                    print("[TTS] Stream reset complete.")
                except Exception as exc:
                    print("[TTS RESET ERROR]", type(exc).__name__, exc)
            print(f"[TTS COMPLETE] {time.perf_counter() - start:.3f}s")

    def interrupt(self) -> None:
        self.generation_id += 1
        while True:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job.cancelled.set()
                job.done.set()
            self.jobs.task_done()
        playback_reference.clear()
        self._cursor_init = False

    def close(self) -> None:
        self.running = False
        self.generation_id += 1
        while True:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job.cancelled.set()
                job.done.set()
            self.jobs.task_done()
        try:
            self.jobs.put(None)
        except Exception:
            pass
        try:
            self.worker.join(timeout=3)
        except Exception:
            pass
        try:
            self.output.stop_stream()
            self.output.close()
        except Exception:
            pass
        try:
            self.audio.terminate()
        except Exception:
            pass
        playback_reference.clear()