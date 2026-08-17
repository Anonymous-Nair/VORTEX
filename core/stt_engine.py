from __future__ import annotations

import asyncio
import os
import tempfile
import wave

from faster_whisper import WhisperModel


class STTEngine:
    """
    Local faster-whisper adapter.

    The model remains CPU/int8 to preserve GPU VRAM for
    the VORTEX brain and TTS.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
    ) -> None:
        self.model_size = model_size
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    async def transcribe(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
    ) -> str:

        if not pcm_bytes:
            return ""

        fd, path = tempfile.mkstemp(
            suffix=".wav",
            prefix="vortex_",
        )
        os.close(fd)

        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)

            def run():
                segments, _ = self.model.transcribe(
                    path,
                    language="en",
                    beam_size=5,
                    # Prevents a garbled/repetitive segment from
                    # poisoning the next segment's decoding —
                    # this is what causes loops like "may, may, may".
                    condition_on_previous_text=False,
                    # Biases vocabulary toward the wake word so it
                    # isn't misheard as the more common word "vertex".
                    initial_prompt="Vortex",
                    vad_filter=True,
                )
                return " ".join(
                    segment.text.strip()
                    for segment in segments
                    if segment.text.strip()
                ).strip()

            return await asyncio.to_thread(run)

        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def warmup(self) -> None:
        await self.transcribe(
            b"\x00" * 32000
        )