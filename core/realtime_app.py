from __future__ import annotations

import asyncio

import requests

from core.live_audio import LiveAudioSession
from core.realtime_engine import RealtimeEngine

TTS_HEALTH = "http://127.0.0.1:8892/health"


async def _cancel_tasks(*tasks: asyncio.Task[object]) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def main() -> None:
    print("=" * 60)
    print("VORTEX REALTIME LIVE ENGINE")
    print("=" * 60)

    print("\nChecking TTS...")
    try:
        response = requests.get(TTS_HEALTH, timeout=5)
        response.raise_for_status()
        print("[OK] TTS:", response.json())
    except Exception as exc:
        print("[ERROR] TTS unavailable:", exc)
        print("Start the VORTEX TTS server first.")
        return

    realtime = RealtimeEngine()
    audio = LiveAudioSession(events=realtime.events)

    # TTS implementation remains frozen. This only wires its existing
    # playback-start callback into the barge-in detector.
    realtime.tts.set_playback_started_callback(
        audio.get_barge_start_callback()
    )

    events = realtime.events.subscribe()
    speech_jobs = 0

    async def event_monitor() -> None:
        nonlocal speech_jobs

        while True:
            event = await events.get()

            if event.type == "audio.ready":
                print("[OK] Microphone engine ready")

            elif event.type == "speech.started":
                print("\n[MIC] USER SPEAKING")

            elif event.type == "speech.ended":
                print("[MIC] USER FINISHED SPEAKING")

            elif event.type == "stt.final":
                text = event.data.get("text", "")
                print(f"\n[STT] {text}")
                if text:
                    realtime.submit_text(text)

            elif event.type == "assistant.thinking":
                print("[BRAIN] VORTEX THINKING")

            elif event.type == "llm.token":
                pass

            elif event.type == "tts.started":
                speech_jobs += 1
                audio.set_speaking(True)
                print(
                    "[TTS] VORTEX SPEAKING:",
                    event.data.get("text", ""),
                )

            elif event.type in {"tts.completed", "tts.error"}:
                speech_jobs = max(0, speech_jobs - 1)
                audio.set_speaking(speech_jobs > 0)
                if event.type == "tts.completed":
                    print("[OK] VORTEX SPEECH COMPLETE")
                else:
                    print("[ERROR] TTS ERROR:", event.data.get("error"))

            elif event.type == "assistant.interrupted":
                speech_jobs = 0
                audio.set_speaking(False)
                print("[BARGE-IN] VORTEX INTERRUPTED")

            elif event.type == "assistant.error":
                print("[ERROR] BRAIN ERROR:", event.data.get("error"))

            elif event.type == "assistant.completed":
                print(
                    "[BRAIN] VORTEX COMPLETE:",
                    event.data.get("response", ""),
                )

    async def interruption_bridge() -> None:
        interrupt_events = realtime.events.subscribe()
        while True:
            event = await interrupt_events.get()
            if event.type == "assistant.interrupted":
                # Barge-in publishes the event first; this call performs
                # the actual cancellation of the active brain task.
                realtime.interrupt(notify=False)

    print("\n========================================")
    print(" VORTEX REALTIME ONLINE")
    print("========================================")
    print("[MIC] Continuous microphone")
    print("[BRAIN] Qwen3.5 4B / think:false")
    print("[TTS] Qwen3-TTS / 24 kHz")
    print("[BARGE-IN] Enabled")
    print("\nSpeak normally.")
    print("Press Ctrl+C to stop.\n")

    monitor_task = asyncio.create_task(event_monitor())
    interrupt_task = asyncio.create_task(interruption_bridge())

    try:
        await audio.run()
    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        print("\nStopping VORTEX...")
    finally:
        audio.stop()
        await _cancel_tasks(monitor_task, interrupt_task)
        await realtime.close()
        print("VORTEX realtime engine stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print(" VORTEX SHUTDOWN")
        print("=" * 60)
        print("[OK] VORTEX stopped cleanly.")
