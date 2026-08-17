from __future__ import annotations

import asyncio
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


# ============================================================
# TOOL PERMISSION LEVELS
# ============================================================

READ = "READ"
SAFE_ACTION = "SAFE_ACTION"
DESTRUCTIVE = "DESTRUCTIVE"
PRIVILEGED = "PRIVILEGED"


# ============================================================
# TOOL REGISTRY
# ============================================================


class ToolRegistry:
    """
    Central registry for VORTEX tools.

    The LLM never executes Python directly.

    Instead:

        LLM
          ↓
        tool name + JSON arguments
          ↓
        ToolRegistry
          ↓
        validated Python function
          ↓
        result
          ↓
        AgentRuntime
          ↓
        LLM
    """

    def __init__(
        self,
        memory: Any | None = None,
        vortex_root: str = r"E:\VORTEX",
    ) -> None:
        self.memory = memory
        self.vortex_root = Path(vortex_root).resolve()

        self._tools: dict[str, Callable[..., Any]] = {
            "get_current_time": self.get_current_time,
            "search_memory": self.search_memory,
            "read_file": self.read_file,
            "list_directory": self.list_directory,
            "get_vortex_status": self.get_vortex_status,
        }

        self._permissions: dict[str, str] = {
            "get_current_time": READ,
            "search_memory": READ,
            "read_file": READ,
            "list_directory": READ,
            "get_vortex_status": READ,
        }

    # ========================================================
    # OLLAMA TOOL DEFINITIONS
    # ========================================================

    def definitions(self) -> list[dict[str, Any]]:
        """
        Return the tool definitions supplied to the LLM.

        These follow the function/tool schema expected by
        Ollama-compatible chat APIs.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": (
                        "Get the current local date and time "
                        "of the VORTEX computer."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": (
                        "Search VORTEX's existing Obsidian memory "
                        "for information relevant to the user's request."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The topic or information "
                                    "to search for."
                                ),
                            },
                            "max_results": {
                                "type": "integer",
                                "description": (
                                    "Maximum number of memory "
                                    "results to return."
                                ),
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a text file from the VORTEX "
                        "workspace. Use this when the actual "
                        "file contents are required."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Absolute file path, or a "
                                    "path relative to E:\\VORTEX."
                                ),
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": (
                                    "Maximum number of characters "
                                    "to return."
                                ),
                                "minimum": 1,
                                "maximum": 50000,
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": (
                        "List files and directories in a VORTEX "
                        "workspace directory."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Directory path. Absolute paths "
                                    "are allowed only inside E:\\VORTEX."
                                ),
                            },
                            "max_entries": {
                                "type": "integer",
                                "description": (
                                    "Maximum number of entries "
                                    "to return."
                                ),
                                "minimum": 1,
                                "maximum": 200,
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_vortex_status",
                    "description": (
                        "Get basic local VORTEX runtime information "
                        "such as operating system, Python version, "
                        "and VORTEX workspace status."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]

    # ========================================================
    # PUBLIC TOOL EXECUTION
    # ========================================================

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute one registered tool.

        The model can request a tool, but it cannot directly
        execute arbitrary Python code.
        """

        if name not in self._tools:
            return {
                "ok": False,
                "error": f"Unknown tool: {name}",
            }

        permission = self._permissions.get(
            name,
            PRIVILEGED,
        )

        if permission != READ:
            return {
                "ok": False,
                "error": (
                    f"Tool '{name}' is not permitted in "
                    "the current read-only stage."
                ),
            }

        function = self._tools[name]
        arguments = arguments or {}

        try:
            result = await asyncio.to_thread(
                function,
                **arguments,
            )

            return {
                "ok": True,
                "tool": name,
                "permission": permission,
                "result": result,
            }

        except Exception as exc:
            return {
                "ok": False,
                "tool": name,
                "permission": permission,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    # ========================================================
    # PATH SECURITY
    # ========================================================

    def _resolve_workspace_path(
        self,
        path: str,
    ) -> Path:
        """
        Resolve a path while preventing the Stage 1 tools
        from escaping the VORTEX workspace.

        Examples allowed:

            E:\\VORTEX\\core
            E:\\VORTEX\\core\\llm_engine.py
            core\\llm_engine.py
            core

        Examples rejected:

            C:\\Windows
            C:\\Users
            E:\\OtherFolder
            E:\\VORTEX\\..\\OtherFolder
        """

        if not isinstance(path, str) or not path.strip():
            raise ValueError("A valid path is required.")

        candidate = Path(path.strip())

        if not candidate.is_absolute():
            candidate = self.vortex_root / candidate

        resolved = candidate.resolve()

        try:
            resolved.relative_to(self.vortex_root)
        except ValueError as exc:
            raise PermissionError(
                "Path is outside the VORTEX workspace."
            ) from exc

        return resolved

    # ========================================================
    # TOOL: CURRENT TIME
    # ========================================================

    @staticmethod
    def get_current_time() -> dict[str, str]:
        """
        Return the actual local computer time.

        No LLM-generated time is involved.
        """

        now = datetime.now().astimezone()

        return {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "formatted": now.strftime(
                "%A, %d %B %Y at %H:%M:%S %Z"
            ),
            "timezone": str(
                now.tzinfo
            ),
        }

    # ========================================================
    # TOOL: MEMORY SEARCH
    # ========================================================

    def search_memory(
        self,
        query: str,
        max_results: int = 8,
    ) -> dict[str, Any]:
        """
        Search the existing VortexMemory implementation.

        We deliberately reuse the current memory layer rather
        than creating a second memory system.
        """

        if self.memory is None:
            return {
                "query": query,
                "results": [],
                "message": (
                    "VORTEX memory is not connected "
                    "to the ToolRegistry."
                ),
            }

        max_results = max(
            1,
            min(
                int(max_results),
                10,
            ),
        )

        results = self.memory.search(
            query,
            max_results=max_results,
        )

        cleaned: list[dict[str, Any]] = []

        for result in results:
            cleaned.append(
                {
                    "path": result.get("path", ""),
                    "score": result.get("score", 0),
                    "content": result.get(
                        "content",
                        "",
                    ),
                }
            )

        return {
            "query": query,
            "count": len(cleaned),
            "results": cleaned,
        }

    # ========================================================
    # TOOL: READ FILE
    # ========================================================

    def read_file(
        self,
        path: str,
        max_chars: int = 30000,
    ) -> dict[str, Any]:
        """
        Read a UTF-8 text file inside E:\\VORTEX.

        This is deliberately read-only.
        """

        resolved = self._resolve_workspace_path(
            path
        )

        if not resolved.exists():
            raise FileNotFoundError(
                f"File does not exist: {resolved}"
            )

        if not resolved.is_file():
            raise IsADirectoryError(
                f"Path is not a file: {resolved}"
            )

        max_chars = max(
            1,
            min(
                int(max_chars),
                50000,
            ),
        )

        content = resolved.read_text(
            encoding="utf-8",
            errors="replace",
        )

        truncated = len(content) > max_chars

        if truncated:
            content = content[:max_chars]

        return {
            "path": str(resolved),
            "characters": len(content),
            "truncated": truncated,
            "content": content,
        }

    # ========================================================
    # TOOL: LIST DIRECTORY
    # ========================================================

    def list_directory(
        self,
        path: str,
        max_entries: int = 100,
    ) -> dict[str, Any]:
        """
        List a directory inside E:\\VORTEX.

        No files are modified.
        """

        resolved = self._resolve_workspace_path(
            path
        )

        if not resolved.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {resolved}"
            )

        if not resolved.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {resolved}"
            )

        max_entries = max(
            1,
            min(
                int(max_entries),
                200,
            ),
        )

        entries = []

        for entry in sorted(
            resolved.iterdir(),
            key=lambda item: (
                not item.is_dir(),
                item.name.lower(),
            ),
        ):
            if len(entries) >= max_entries:
                break

            entries.append(
                {
                    "name": entry.name,
                    "type": (
                        "directory"
                        if entry.is_dir()
                        else "file"
                    ),
                }
            )

        return {
            "path": str(resolved),
            "count": len(entries),
            "entries": entries,
        }

    # ========================================================
    # TOOL: VORTEX STATUS
    # ========================================================

    def get_vortex_status(self) -> dict[str, Any]:
        """
        Return safe local runtime information.

        No credentials, environment secrets, API keys,
        or private service configuration are exposed.
        """

        return {
            "workspace": str(
                self.vortex_root
            ),
            "workspace_exists": (
                self.vortex_root.exists()
            ),
            "workspace_is_directory": (
                self.vortex_root.is_dir()
            ),
            "operating_system": (
                platform.system()
            ),
            "platform": platform.platform(),
            "python_version": (
                sys.version.split()[0]
            ),
            "python_executable": (
                sys.executable
            ),
            "username_available": bool(
                os.environ.get("USERNAME")
            ),
            "stage": "AGENT_STAGE_1_READ_ONLY",
        }