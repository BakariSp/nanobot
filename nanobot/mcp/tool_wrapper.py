"""Wrapper to expose MCP tools as nanobot Tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.mcp.session import MCPSession


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a nanobot Tool.

    Registered into ToolRegistry with name ``{prefix}__{mcp_tool_name}``.
    """

    def __init__(
        self,
        server_name: str,
        tool_definition: dict[str, Any],
        session_provider: Callable[[], MCPSession | None],
    ):
        self._server_name = server_name
        self._tool_def = tool_definition
        self._session_provider = session_provider
        # Double-underscore separates prefix from tool name
        self._name = f"{server_name}__{tool_definition['name']}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        desc = self._tool_def.get("description", "")
        return f"[{self._server_name}] {desc}"

    @property
    def parameters(self) -> dict[str, Any]:
        input_schema = self._tool_def.get("inputSchema", {})
        # MCP inputSchema is already JSON Schema object type — use directly
        if input_schema.get("type") == "object":
            return input_schema
        # Rare edge case: non-object schema — wrap it
        return {
            "type": "object",
            "properties": {"input": input_schema},
            "required": ["input"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Forward call to MCP server and return result as string."""
        session = self._session_provider()

        if not session:
            return f"Error: MCP server '{self._server_name}' not available"
        if not session.is_connected:
            return f"Error: MCP server '{self._server_name}' not connected"

        original_name = self._tool_def["name"]

        try:
            result = await session.call_tool(original_name, kwargs)

            # Extract text from MCP CallToolResult
            if hasattr(result, "content") and isinstance(result.content, list):
                if not result.content:
                    return "(empty response)"

                # Concatenate all text content blocks
                parts: list[str] = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    elif hasattr(item, "data"):
                        parts.append(f"[Image: {getattr(item, 'mimeType', 'unknown')}]")
                    elif hasattr(item, "resource"):
                        parts.append(json.dumps(item.resource, indent=2, ensure_ascii=False))

                if parts:
                    return "\n".join(parts)

            # Fallback
            return str(result)

        except Exception as e:
            logger.error(f"MCP tool '{self._name}' execution failed: {e}")
            return f"Error calling {self._server_name} tool '{original_name}': {e}"
