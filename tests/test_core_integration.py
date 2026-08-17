from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from core.agent_runtime import AgentRuntime
from core.agent_tools import ToolRegistry
from core.event_bus import EventBus
from core.memory_bridge import VortexMemory


class FakeLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    def definitions(self):
        return []

    async def chat(self, messages, tools=None):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class CoreIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_bus_subscription_lifecycle(self):
        bus = EventBus()
        queue = bus.subscribe()
        self.assertEqual(bus.subscriber_count(), 1)
        await bus.publish("test", value=42)
        event = await asyncio.wait_for(queue.get(), 0.2)
        self.assertEqual(event.type, "test")
        self.assertEqual(event.data["value"], 42)
        self.assertTrue(bus.unsubscribe(queue))
        self.assertFalse(bus.unsubscribe(queue))
        self.assertEqual(bus.subscriber_count(), 0)

    async def test_memory_write_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = VortexMemory(vortex_root=tmp, cache_ttl_seconds=0)
            self.assertFalse((Path(tmp) / "Memory").exists())
            path = memory.save_long_term("Python Runtime", "Cancellation safety is important.")
            self.assertTrue(path.exists())
            results = memory.search("cancellation safety")
            self.assertEqual(len(results), 1)
            self.assertIn("Cancellation", results[0]["content"])

    async def test_tool_registry_executes_registered_read_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "hello.txt"
            target.write_text("hello vortex", encoding="utf-8")
            tools = ToolRegistry(vortex_root=str(root))
            result = await tools.execute("read_file", {"path": "hello.txt"})
            self.assertTrue(result["ok"])
            self.assertEqual(result["result"]["content"], "hello vortex")

    async def test_agent_cancellation_propagates_to_llm(self):
        llm = FakeLLM()
        tools = ToolRegistry(vortex_root=tempfile.gettempdir())
        agent = AgentRuntime(llm=llm, tools=tools)
        task = asyncio.create_task(agent.run([{"role": "user", "content": "hello"}]))
        await asyncio.wait_for(llm.started.wait(), 0.5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(llm.cancelled.is_set())


class SourceArchitectureTests(unittest.TestCase):
    def test_llm_transport_is_async(self):
        source = Path("core/llm_engine.py").read_text(encoding="utf-8-sig")
        self.assertIn("httpx.AsyncClient", source)
        self.assertNotIn("requests.post", source)
        self.assertNotIn("asyncio.to_thread", source)

    def test_tts_file_is_not_part_of_this_change_contract(self):
        self.assertTrue(Path("core/tts_engine.py").exists())


if __name__ == "__main__":
    unittest.main()
