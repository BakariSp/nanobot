"""Pydantic models for MCP configuration."""

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    enabled: bool = True
    transport: str = "stdio"  # "stdio" or "http"

    # stdio transport
    command: str | None = None  # e.g., "npx"
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    # http transport (future)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    # Behavior
    reconnect_on_failure: bool = True
    reconnect_delay_ms: int = 2000
    max_reconnect_attempts: int = 5
    call_timeout_s: int = 30

    # Tool naming — prefix for registered tool names (defaults to server key)
    tool_prefix: str | None = None


class MCPConfig(BaseModel):
    """MCP integration configuration."""

    enabled: bool = False
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    shutdown_timeout_s: int = 5
