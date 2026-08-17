from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BargeInConfig:
    mic_rate: int = 16000
    mic_frame_samples: int = 320
    history_samples: int = 4800

    min_delay_samples: int = 320
    max_delay_samples: int = 3200
    search_step: int = 8

    mic_floor: float = 450.0
    residual_floor: float = 180.0
    residual_ratio: float = 2.2
    residual_margin: float = 180.0

    confirm_frames: int = 5
    startup_suppression_ms: float = 300.0

    baseline_attack: float = 0.02
    baseline_release: float = 0.10

    diagnostic_only: bool = True


class BargeInDetector:
    """Playback-reference barge-in detector with NCC-gated echo cancellation."""

    def __init__(self, config: BargeInConfig | None = None):
        self.config = config or BargeInConfig()
        self._baseline = 50.0
        self._frames = 0
        self._candidate = False
        self._active = False
        self._started = 0.0
        self._last_log = 0.0
        self._peak = 0.0

    async def calibrate(self, *args, **kwargs) -> dict:
        """Compatibility hook: runtime playback-reference mode."""
        print("[CALIBRATION] Runtime playback-reference mode enabled; no fixed delay calibration required.")
        return {
            "ok": True,
            "mean_delay_ms": 0.0,
            "std_delay_ms": 0.0,
            "mean_correlation": 0.0,
            "mean_gain": 0.0,
        }

    def start_speaking(self) -> None:
        self._active = True
        self._started = time.perf_counter()
        self._frames = 0
        self._candidate = False
        self._peak = 0.0
        self._baseline = 50.0
        self._last_log = 0.0

    def reset(self) -> None:
        self._active = False
        self._started = 0.0
        self._frames = 0
        self._candidate = False
        self._peak = 0.0
        self._baseline = 50.0

    def update(self, mic_raw: bytes, mic_energy: float, capture_time: float | None = None) -> bool:
        del capture_time

        if not self._active:
            return False

        now = time.perf_counter()
        if (now - self._started) * 1000.0 < self.config.startup_suppression_ms:
            return False

        mic = np.frombuffer(mic_raw, dtype=np.int16).astype(np.float32)
        target = self.config.mic_frame_samples
        if len(mic) < target:
            mic = np.pad(mic, (0, target - len(mic)))
        elif len(mic) > target:
            mic = mic[:target]

        if mic_energy < self.config.mic_floor:
            self._frames = 0
            self._candidate = False
            return False

        from .tts_engine import playback_reference

        history = playback_reference.get_recent(self.config.history_samples)
        ref, lag, ncc = self._align(mic, history)
        ref_energy = float(np.dot(ref, ref))

        if ref_energy <= 1.0:
            gain = 0.0
            residual_rms = float(np.sqrt(np.mean(mic * mic)))
        else:
            m = mic - np.mean(mic)
            r = ref - np.mean(ref)
            denom = max(float(np.dot(r, r)), 1.0)
            raw_gain = float(np.dot(m, r) / denom)

            # FIX #1: Gate gain by NCC. When correlation is weak,
            # the least-squares gain is unreliable. Scale it down
            # so weak correlations produce near-zero echo estimates
            # instead of destructive noise amplification.
            ncc_gate = min(max(ncc, 0.0) / 0.5, 1.0)
            gain = max(0.0, min(raw_gain * ncc_gate, 2.0))

            residual = m - gain * r
            residual_rms = float(np.sqrt(np.mean(residual * residual)))

        alpha = self.config.baseline_attack if self._candidate else self.config.baseline_release
        self._baseline = max(
            20.0,
            (1.0 - alpha) * self._baseline + alpha * residual_rms,
        )

        # FIX #2: NCC-modulated residual threshold.
        # When NCC is high (signal correlates with VORTEX playback),
        # require proportionally higher residual to confirm user speech.
        # At ncc=0.8, scale=3.0 -> residual must be 3x baseline thresholds.
        # At ncc=0.0, scale=1.0 -> standard thresholds apply.
        # This rejects pure echo while preserving double-talk detection.
        ncc_scale = 1.0 + min(max(ncc, 0.0), 0.8) * 2.5
        effective_ratio = self.config.residual_ratio * ncc_scale
        effective_margin = self.config.residual_margin * ncc_scale

        candidate = (
            mic_energy >= self.config.mic_floor
            and residual_rms >= self.config.residual_floor
            and residual_rms > self._baseline * effective_ratio
            and residual_rms > self._baseline + effective_margin
        )

        if candidate:
            if not self._candidate:
                print(
                    f"[BARGE] CANDIDATE START mic={mic_energy:.0f} "
                    f"residual={residual_rms:.0f} baseline={self._baseline:.0f} "
                    f"ncc={ncc:.3f} gain={gain:.4f} scale={ncc_scale:.2f}"
                )
            self._candidate = True
            self._frames += 1
            self._peak = max(self._peak, residual_rms)
        else:
            if self._candidate:
                print(
                    f"[BARGE] CANDIDATE RESET residual={residual_rms:.0f} "
                    f"ncc={ncc:.3f} held={self._frames}"
                )
            self._candidate = False
            self._frames = 0
            self._peak = 0.0

        if self._candidate or now - self._last_log >= 0.5:
            mode = "DIAG" if self.config.diagnostic_only else "LIVE"
            tag = "ECHO-LIKELY" if ncc >= 0.65 else "INDEPENDENT-LIKELY"
            print(
                f"[BARGE][{mode}] mic={mic_energy:.0f} ncc={ncc:.3f} "
                f"gain={gain:.4f} residual={residual_rms:.0f} "
                f"baseline={self._baseline:.0f} "
                f"lag_ms={lag / self.config.mic_rate * 1000.0:.1f} "
                f"scale={ncc_scale:.2f} "
                f"{tag} frames={self._frames}/{self.config.confirm_frames}"
            )
            self._last_log = now

        if self._frames >= self.config.confirm_frames:
            peak = self._peak
            print(
                f"[BARGE] USER INTERRUPTION DETECTED peak_residual={peak:.0f} "
                f"ncc={ncc:.3f} gain={gain:.4f} scale={ncc_scale:.2f}"
            )
            self._frames = 0
            self._candidate = False
            self._peak = 0.0
            if self.config.diagnostic_only:
                print("[BARGE] [DIAGNOSTIC MODE — INTERRUPTION SUPPRESSED]")
                return False
            return True

        return False

    def _align(self, mic: np.ndarray, history: np.ndarray) -> tuple[np.ndarray, int, float]:
        n = len(mic)
        if len(history) < n:
            return np.pad(history, (n - len(history), 0))[-n:], 0, 0.0

        mic0 = mic - np.mean(mic)
        mic_std = float(np.std(mic0))
        if mic_std < 1.0:
            return history[-n:].copy(), 0, 0.0

        low = min(self.config.min_delay_samples, len(history) - n)
        high = min(self.config.max_delay_samples, len(history) - n)
        if high < low:
            return history[-n:].copy(), 0, 0.0

        best_ncc = -1.0
        best_lag = low

        for lag in range(low, high + 1, self.config.search_step):
            start = len(history) - n - lag
            if start < 0:
                continue
            r = history[start:start + n]
            r0 = r - np.mean(r)
            r_std = float(np.std(r0))
            if r_std < 1.0:
                continue
            ncc = float(np.dot(mic0, r0) / max(mic_std * r_std * n, 1.0))
            if ncc > best_ncc:
                best_ncc = ncc
                best_lag = lag

        start = max(0, len(history) - n - best_lag)
        ref = history[start:start + n].copy()
        if len(ref) < n:
            ref = np.pad(ref, (0, n - len(ref)))
        return ref, best_lag, max(0.0, best_ncc)