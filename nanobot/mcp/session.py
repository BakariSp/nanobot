"""MCP session wrapper for a single server connection."""

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any

from loguru import logger

from nanobot.mcp.types import MCPServerConfig


def _expand_env_vars(text: str, env_dict: dict[str, str]) -> str:
    """Expand ${VAR} placeholders using env_dict, then os.environ as fallback."""
    import re

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        return env_dict.get(key, os.environ.get(key, m.group(0)))

    return re.sub(r"\$\{(\w+)\}", _replace, text)


class MCPSession:
    """Manages a single MCP server connection."""

    def __init__(self, server_name: str, config: MCPServerConfig):
        self.server_name = server_name
        self.config = config
        self._session: Any = None  # mcp.ClientSession
        self.is_connected = False

    async def connect(self, exit_stack: AsyncExitStack) -> None:
        """Connect to MCP server using configured transport."""
        if self.config.transport == "stdio":
            await self._connect_stdio(exit_stack)
        elif self.config.transport == "http":
            await self._connect_http(exit_stack)
        else:
            raise ValueError(f"Unknown MCP transport: {self.config.transport}")

    async def _connect_stdio(self, exit_stack: AsyncExitStack) -> None:
        """Connect via stdio subprocess."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # Expand ${VAR} placeholders in args
        expanded_args = [_expand_env_vars(arg, self.config.env) for arg in self.config.args]

        # Merge environment: inherit os.environ + overlay config env
        full_env = {**os.environ, **self.config.env}

        server_params = StdioServerParameters(
            command=self.config.command,
            args=expanded_args,
            env=full_env,
        )

        # Enter stdio client context (spawns subprocess)
        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        # Create and enter session context
        self._session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # MCP handshake
        await self._session.initialize()
        self.is_connected = True
        logger.debug(f"MCP server '{self.server_name}' connected via stdio")

    async def _connect_http(self, exit_stack: AsyncExitStack) -> None:
        """Connect via Streamable HTTP transport."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        if not self.config.url:
            raise ValueError(f"MCP server '{self.server_name}': url required for http transport")

        # Expand env vars in headers
        headers = {k: _expand_env_vars(v, self.config.env) for k, v in self.config.headers.items()}

        http_transport = await exit_stack.enter_async_context(
            streamable_http_client(self.config.url, headers=headers)
        )
        read_stream, write_stream, _ = http_transport

        self._session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await self._session.initialize()
        self.is_connected = True
        logger.debug(f"MCP server '{self.server_name}' connected via http")

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover tools from the server."""
        if not self._session:
            raise RuntimeError(f"MCP session '{self.server_name}' not initialized")

        result = await self._session.list_tools()

        tools = []
        for tool in result.tools:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema or {},
                }
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool on the server with timeout."""
        if not self.is_connected or not self._session:
            raise RuntimeError(f"MCP server '{self.server_name}' not connected")

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=self.config.call_timeout_s,
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                f"MCP tool '{name}' on '{self.server_name}' "
                f"timed out after {self.config.call_timeout_s}s"
            )
            raise RuntimeError(f"Tool '{name}' execution timed out")
        except Exception as e:
            # Detect disconnection
            err_lower = str(e).lower()
            if "closed" in err_lower or "broken pipe" in err_lower or "eof" in err_lower:
                logger.warning(f"MCP server '{self.server_name}' disconnected: {e}")
                self.is_connected = False
            raise
