from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"

EXPECTED_CORE = {
    "__init__.py",
    "agent_runtime.py",
    "agent_tools.py",
    "audio_engine.py",
    "barge_in.py",
    "event_bus.py",
    "hud_bridge.py",
    "live_audio.py",
    "llm_engine.py",
    "memory_bridge.py",
    "realtime_app.py",
    "realtime_engine.py",
    "stt_engine.py",
    "tts_engine.py",
    "vad_engine.py",
}


def test_core_layout_is_complete() -> None:
    assert CORE.is_dir()
    assert {p.name for p in CORE.iterdir()} >= EXPECTED_CORE


def test_core_python_is_syntactically_valid() -> None:
    for path in CORE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def test_agent_runtime_uses_public_tool_registry_api() -> None:
    source = (CORE / "agent_runtime.py").read_text(encoding="utf-8-sig")
    assert "self.tools.execute(" in source
    assert "self.tools.execute_tool(" not in source


def test_realtime_entrypoint_targets_core_package() -> None:
    source = (ROOT / "realtime_app.py").read_text(encoding="utf-8-sig")
    assert "from core.live_audio import LiveAudioSession" in source
    assert "from core.realtime_engine import RealtimeEngine" in source
