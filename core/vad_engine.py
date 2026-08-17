from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SpeechState(str, Enum):
    SILENCE = "silence"
    SPEAKING = "speaking"
    ENDED = "ended"


@dataclass(slots=True)
class VADConfig:
    start_threshold: float = 350.0
    end_threshold: float = 280.0
    start_frames: int = 2
    end_frames: int = 40
    max_turn_seconds: float = 15.0


class TurnDetector:
    """Lightweight energy-based turn detector with a hard turn-duration cap."""

    def __init__(self, config: VADConfig | None = None):
        self.config = config or VADConfig()
        self.state = SpeechState.SILENCE
        self._speech_frames = 0
        self._silence_frames = 0
        self.elapsed = 0.0

    def update(self, energy: float, frame_seconds: float) -> SpeechState:
        self.elapsed += max(frame_seconds, 0.0)

        if self.state == SpeechState.SILENCE:
            if energy >= self.config.start_threshold:
                self._speech_frames += 1
            else:
                self._speech_frames = 0

            if self._speech_frames >= self.config.start_frames:
                self.state = SpeechState.SPEAKING
                self._silence_frames = 0

        elif self.state == SpeechState.SPEAKING:
            if self.timed_out():
                self.state = SpeechState.ENDED
                self._silence_frames = 0
                return self.state

            if energy <= self.config.end_threshold:
                self._silence_frames += 1
            else:
                self._silence_frames = 0

            if self._silence_frames >= self.config.end_frames:
                self.state = SpeechState.ENDED

        return self.state

    def resume(self) -> None:
        """Resume a provisionally-ended turn during the grace window."""
        self.state = SpeechState.SPEAKING
        self._silence_frames = 0

    def reset(self) -> None:
        self.state = SpeechState.SILENCE
        self._speech_frames = 0
        self._silence_frames = 0
        self.elapsed = 0.0

    def timed_out(self) -> bool:
        return self.elapsed >= self.config.max_turn_seconds
