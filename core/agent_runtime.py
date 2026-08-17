from __future__ import annotations

import json
from typing import Any

from .agent_tools import ToolRegistry
from .llm_engine import LLMEngine


MAX_TOOL_ROUNDS = 6


class AgentRuntime:
    """
    VORTEX agent runtime.

    The runtime coordinates:
        LLM -> tool request -> ToolRegistry -> LLM -> final answer

    Tool execution is exclusively delegated to ToolRegistry.
    """

    def __init__(
        self,
        llm: LLMEngine,
        tools: ToolRegistry,
    ) -> None:
        self.llm = llm
        self.tools = tools

    async def run(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """
        Run one isolated agent turn.

        Tool state is local to this invocation.
        Nothing from one invocation is retained for another.
        """

        working_messages = [
            dict(message)
            for message in messages
        ]

        for _ in range(MAX_TOOL_ROUNDS):
            response = await self._request(
                working_messages
            )

            content = response.get(
                "content",
                "",
            )

            if not isinstance(content, str):
                content = ""

            tool_calls = response.get(
                "tool_calls",
                [],
            )

            if not tool_calls:
                return self._remove_tool_calls(
                    content
                ).strip()

            # Preserve the assistant's native tool-call message.
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }

            raw_calls: list[dict[str, Any]] = []

            for call in tool_calls:
                name = getattr(
                    call,
                    "name",
                    "",
                )

                arguments = getattr(
                    call,
                    "arguments",
                    {},
                )

                call_id = getattr(
                    call,
                    "call_id",
                    "",
                )

                if not isinstance(name, str):
                    continue

                name = name.strip()

                if not name:
                    continue

                if not isinstance(arguments, dict):
                    arguments = {}

                raw_call: dict[str, Any] = {
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }

                if isinstance(call_id, str) and call_id:
                    raw_call["id"] = call_id

                raw_calls.append(raw_call)

            if not raw_calls:
                return content.strip()

            assistant_message["tool_calls"] = raw_calls

            working_messages.append(
                assistant_message
            )

            # -------------------------------------------------
            # EXECUTE TOOLS
            # -------------------------------------------------

            executed_ids: set[str] = set()

            for call in tool_calls:
                call_id = getattr(
                    call,
                    "call_id",
                    "",
                )

                name = getattr(
                    call,
                    "name",
                    "",
                )

                arguments = getattr(
                    call,
                    "arguments",
                    {},
                )

                if not isinstance(name, str):
                    continue

                name = name.strip()

                if not name:
                    continue

                if not isinstance(arguments, dict):
                    arguments = {}

                # Prevent duplicate execution inside one response.
                dedupe_key = (
                    f"{call_id}|"
                    f"{name}|"
                    f"{json.dumps(arguments, sort_keys=True, default=str)}"
                )

                if dedupe_key in executed_ids:
                    continue

                executed_ids.add(dedupe_key)

                result = await self._execute_tool(
                    name,
                    arguments,
                )

                try:
                    serialized = json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )
                except Exception as exc:
                    serialized = json.dumps(
                        {
                            "ok": False,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )

                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "content": serialized,
                    "name": name,
                }

                if isinstance(call_id, str) and call_id:
                    tool_message[
                        "tool_call_id"
                    ] = call_id

                working_messages.append(
                    tool_message
                )

        return (
            "I couldn't complete that operation "
            "within the allowed tool steps, sir."
        )

    async def _request(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Perform one LLM agent request.

        IMPORTANT:
        Use LLMEngine.chat(), not the legacy stream()
        interface, because stream() intentionally exposes
        text only and discards tool-call events.
        """

        tools = self.tools.definitions()

        response = await self.llm.chat(
            messages,
            tools=tools,
        )

        if not isinstance(response, dict):
            return {
                "content": "",
                "tool_calls": [],
            }

        return response

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute exactly one registered VORTEX tool.

        AgentRuntime never executes tools itself.
        """

        try:
            result = await self.tools.execute_tool(
                name,
                arguments,
            )

            if isinstance(result, dict):
                return result

            return {
                "ok": True,
                "result": result,
            }

        except Exception as exc:
            return {
                "ok": False,
                "tool": name,
                "error": str(exc),
            }

    @staticmethod
    def _extract_tool_calls(
        response: dict[str, Any],
    ) -> list[Any]:
        """
        Extract normalized tool calls from an LLM response.
        """

        calls = response.get(
            "tool_calls",
            [],
        )

        if not isinstance(calls, list):
            return []

        return calls

    @staticmethod
    def _deduplicate(
        tool_calls: list[Any],
    ) -> list[Any]:
        """
        Remove duplicate tool calls while preserving order.
        """

        result: list[Any] = []
        seen: set[str] = set()

        for call in tool_calls:
            name = getattr(
                call,
                "name",
                "",
            )

            arguments = getattr(
                call,
                "arguments",
                {},
            )

            call_id = getattr(
                call,
                "call_id",
                "",
            )

            try:
                key = (
                    f"{call_id}|"
                    f"{name}|"
                    f"{json.dumps(arguments, sort_keys=True, default=str)}"
                )
            except Exception:
                key = (
                    f"{call_id}|"
                    f"{name}|"
                    f"{repr(arguments)}"
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(call)

        return result

    @staticmethod
    def _remove_tool_calls(
        text: str,
    ) -> str:
        """
        Remove accidental tool-call markup from final text.

        Normal Ollama responses should already be clean, but this
        protects the voice layer from provider-specific artifacts.
        """

        if not isinstance(text, str):
            return ""

        return text.strip()