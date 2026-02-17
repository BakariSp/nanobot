"""MCP Manager — lifecycle management for all MCP server connections."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.mcp.session import MCPSession
from nanobot.mcp.tool_wrapper import MCPToolWrapper
from nanobot.mcp.types import MCPConfig

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry


class MCPManager:
    """Manages MCP server connections and dynamic tool registration."""

    def __init__(self, config: MCPConfig, tool_registry: ToolRegistry):
        self.config = config
        self.registry = tool_registry
        self.sessions: dict[str, MCPSession] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._running = False
        # Track which tool names we registered (for cleanup)
        self._registered_tools: list[str] = []

    async def start(self) -> None:
        """Connect to all enabled MCP servers and register their tools."""
        if not self.config.enabled:
            logger.info("MCP integration disabled")
            return

        self._exit_stack = AsyncExitStack()
        total_tools = 0

        for name, server_config in self.config.servers.items():
            if not server_config.enabled:
                logger.debug(f"MCP server '{name}' disabled, skipping")
                continue

            try:
                session = MCPSession(name, server_config)
                await session.connect(self._exit_stack)

                # Discover tools
                tools = await session.list_tools()
                prefix = server_config.tool_prefix or name

                # Wrap and register each tool
                for tool_def in tools:
                    # Capture `name` by value in lambda default arg
                    wrapper = MCPToolWrapper(
                        server_name=prefix,
                        tool_definition=tool_def,
                        session_provider=lambda n=name: self.sessions.get(n),
                    )
                    self.registry.register(wrapper)
                    self._registered_tools.append(wrapper.name)
                    logger.debug(f"  Registered MCP tool: {wrapper.name}")

                self.sessions[name] = session
                total_tools += len(tools)
                logger.info(f"MCP server '{name}' connected ({len(tools)} tools)")

            except Exception as e:
                logger.error(f"Failed to connect MCP server '{name}': {e}")
                # Non-fatal: continue with other servers

        self._running = True
        logger.info(f"MCP Manager started ({len(self.sessions)} servers, {total_tools} tools)")

    async def stop(self) -> None:
        """Disconnect all MCP servers and unregister their tools."""
        if not self._running:
            return

        # Unregister all MCP tools from nanobot registry
        for tool_name in self._registered_tools:
            self.registry.unregister(tool_name)
            logger.debug(f"Unregistered MCP tool: {tool_name}")
        self._registered_tools.clear()

        # Close all sessions via exit stack
        if self._exit_stack:
            try:
                await asyncio.wait_for(
                    self._exit_stack.aclose(),
                    timeout=self.config.shutdown_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning("MCP shutdown timed out, some processes may linger")
            except Exception as e:
                logger.warning(f"MCP shutdown error: {e}")

        self.sessions.clear()
        self._running = False
        logger.info("MCP Manager stopped")

    async def reconnect(self, server_name: str) -> bool:
        """Attempt to reconnect a failed server with retry logic."""
        if server_name not in self.config.servers:
            logger.error(f"Unknown MCP server: {server_name}")
            return False

        server_config = self.config.servers[server_name]
        if not server_config.reconnect_on_failure:
            return False

        for attempt in range(1, server_config.max_reconnect_attempts + 1):
            try:
                logger.info(
                    f"Reconnecting MCP server '{server_name}' "
                    f"(attempt {attempt}/{server_config.max_reconnect_attempts})"
                )
                await asyncio.sleep(server_config.reconnect_delay_ms / 1000)

                # Remove old session
                self.sessions.pop(server_name, None)

                # Create fresh session (reuse exit stack)
                if not self._exit_stack:
                    self._exit_stack = AsyncExitStack()

                session = MCPSession(server_name, server_config)
                await session.connect(self._exit_stack)
                self.sessions[server_name] = session

                logger.info(f"MCP server '{server_name}' reconnected")
                return True

            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} for '{server_name}' failed: {e}")

        logger.error(
            f"Failed to reconnect MCP server '{server_name}' "
            f"after {server_config.max_reconnect_attempts} attempts"
        )
        return False
