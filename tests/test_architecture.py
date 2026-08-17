from pathlib import Path
import ast
import unittest

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


class ArchitectureTests(unittest.TestCase):
    def test_core_layout_is_complete(self) -> None:
        self.assertTrue(CORE.is_dir())
        self.assertTrue(EXPECTED_CORE <= {p.name for p in CORE.iterdir()})

    def test_core_python_is_syntactically_valid(self) -> None:
        for path in CORE.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_agent_runtime_uses_public_tool_registry_api(self) -> None:
        source = (CORE / "agent_runtime.py").read_text(encoding="utf-8-sig")
        self.assertIn("self.tools.execute(", source)
        self.assertNotIn("self.tools.execute_tool(", source)

    def test_realtime_entrypoint_targets_core_package(self) -> None:
        source = (ROOT / "realtime_app.py").read_text(encoding="utf-8-sig")
        self.assertIn("from core.live_audio import LiveAudioSession", source)
        self.assertIn("from core.realtime_engine import RealtimeEngine", source)


if __name__ == "__main__":
    unittest.main()
