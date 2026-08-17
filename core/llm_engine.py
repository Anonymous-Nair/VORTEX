from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx


@dataclass(slots=True)
class LLMToolCall:
    """
    Normalized Ollama tool-call representation.

    Keeping tool calls separate from ordinary text allows the
    realtime/agent layer to decide what should actually execute.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class LLMStreamEvent:
    """
    Normalized event emitted by the LLM streaming layer.

    kind:
        "text"      -> ordinary generated text
        "tool_call" -> model requested a tool
        "done"      -> model finished the response
    """

    kind: str
    text: str = ""
    elapsed: float = 0.0
    tool_call: LLMToolCall | None = None
    raw: dict[str, Any] | None = None


class LLMEngine:
    """
    Ollama-compatible LLM transport for VORTEX.

    Responsibilities:
        - communicate with the local Ollama chat API
        - stream normal text
        - expose Ollama tool definitions
        - normalize streamed tool calls
        - provide a non-streaming chat operation for agent turns

    This class does NOT execute tools.

    Tool execution belongs exclusively to ToolRegistry.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:11434/api/chat",
        model: str = "vortex:latest",
    ) -> None:
        self.url = url
        self.model = model

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.7,
                "top_k": 30,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }

        if tools:
            payload["tools"] = tools

        return payload

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[tuple[str, float]]:
        async for event in self.stream_events(messages, tools=tools):
            if event.kind == "text" and event.text:
                yield event.text, event.elapsed
            elif event.kind == "done":
                break

    async def stream_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        payload = self._build_payload(messages, stream=True, tools=tools)
        started = time.perf_counter()
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", self.url, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        packet = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    elapsed = time.perf_counter() - started
                    message = packet.get("message", {})

                    piece = message.get("content", "")
                    if piece:
                        yield LLMStreamEvent(
                            kind="text",
                            text=piece,
                            elapsed=elapsed,
                            raw=packet,
                        )

                    tool_calls = message.get("tool_calls", [])
                    if tool_calls:
                        for raw_call in tool_calls:
                            normalized = self._normalize_tool_call(raw_call)
                            if normalized is None:
                                continue
                            yield LLMStreamEvent(
                                kind="tool_call",
                                elapsed=elapsed,
                                tool_call=normalized,
                                raw=packet,
                            )

                    if packet.get("done"):
                        yield LLMStreamEvent(
                            kind="done",
                            elapsed=elapsed,
                            raw=packet,
                        )
                        break

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(messages, stream=False, tools=tools)
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.url, json=payload)
            response.raise_for_status()
            packet = response.json()

        message = packet.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls", [])

        tool_calls: list[LLMToolCall] = []
        for raw_call in raw_tool_calls:
            normalized = self._normalize_tool_call(raw_call)
            if normalized is not None:
                tool_calls.append(normalized)

        return {
            "message": message,
            "content": content,
            "tool_calls": tool_calls,
            "done": bool(packet.get("done")),
            "raw": packet,
        }

    @staticmethod
    def _normalize_tool_call(raw_call: Any) -> LLMToolCall | None:
        if not isinstance(raw_call, dict):
            return None

        function = raw_call.get("function", {})
        if not isinstance(function, dict):
            return None

        name = function.get("name", "")
        if not isinstance(name, str):
            return None
        name = name.strip()
        if not name:
            return None

        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if not isinstance(arguments, dict):
            arguments = {}

        call_id = raw_call.get("id", "")
        if not isinstance(call_id, str):
            call_id = ""

        return LLMToolCall(name=name, arguments=arguments, call_id=call_id)
