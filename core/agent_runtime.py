from __future__ import annotations

import asyncio
import json
from typing import Any

from .agent_tools import ToolRegistry
from .llm_engine import LLMEngine

MAX_TOOL_ROUNDS = 6


class AgentRuntime:
    """Coordinate LLM responses and validated VORTEX tool calls."""

    def __init__(self, llm: LLMEngine, tools: ToolRegistry) -> None:
        self.llm = llm
        self.tools = tools

    async def run(
        self,
        messages: list[dict[str, Any]],
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> str:
        working = [dict(message) for message in messages]

        for _ in range(MAX_TOOL_ROUNDS):
            self._raise_if_cancelled(cancel_event)
            response = await self._request(
                working,
                cancel_event=cancel_event,
            )
            self._raise_if_cancelled(cancel_event)

            content = response.get("content", "")
            if not isinstance(content, str):
                content = ""

            calls = response.get("tool_calls", [])
            if not isinstance(calls, list) or not calls:
                return self._clean_text(content)

            normalized_calls: list[tuple[Any, str, dict[str, Any], str]] = []
            raw_calls: list[dict[str, Any]] = []

            for call in calls:
                name = getattr(call, "name", "")
                arguments = getattr(call, "arguments", {})
                call_id = getattr(call, "call_id", "")

                if not isinstance(name, str) or not name.strip():
                    continue
                name = name.strip()
                if not isinstance(arguments, dict):
                    arguments = {}
                if not isinstance(call_id, str):
                    call_id = ""

                normalized_calls.append((call, name, arguments, call_id))
                raw = {"function": {"name": name, "arguments": arguments}}
                if call_id:
                    raw["id"] = call_id
                raw_calls.append(raw)

            if not normalized_calls:
                return self._clean_text(content)

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": raw_calls,
            }
            working.append(assistant_message)

            seen: set[str] = set()
            for _, name, arguments, call_id in normalized_calls:
                self._raise_if_cancelled(cancel_event)

                try:
                    key = f"{call_id}|{name}|{json.dumps(arguments, sort_keys=True, default=str)}"
                except Exception:
                    key = f"{call_id}|{name}|{repr(arguments)}"
                if key in seen:
                    continue
                seen.add(key)

                result = await self._execute_tool(name, arguments)
                self._raise_if_cancelled(cancel_event)

                try:
                    serialized = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as exc:
                    serialized = json.dumps({"ok": False, "error": str(exc)})

                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "content": serialized,
                    "name": name,
                }
                if call_id:
                    tool_message["tool_call_id"] = call_id
                working.append(tool_message)

        self._raise_if_cancelled(cancel_event)
        return "I couldn't complete that operation within the allowed tool steps, sir."

    async def _request(
        self,
        messages: list[dict[str, Any]],
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        response = await self.llm.chat(
            messages,
            tools=self.tools.definitions(),
            cancel_event=cancel_event,
        )
        self._raise_if_cancelled(cancel_event)
        return response if isinstance(response, dict) else {"content": "", "tool_calls": []}

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self.tools.execute(name, arguments)
            return result if isinstance(result, dict) else {"ok": True, "result": result}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _raise_if_cancelled(cancel_event: asyncio.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError

    @staticmethod
    def _clean_text(text: str) -> str:
        return text.strip() if isinstance(text, str) else ""
