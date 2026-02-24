"""Tool registry for dynamic tool management."""

from typing import Any

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._modes: dict[str, str] = {}  # tool_name → required mode (e.g. "kevin")
        self._exclude_modes: dict[str, set[str]] = {}  # tool_name → modes where tool is hidden

    def register(self, tool: Tool, mode: str | None = None, exclude_modes: list[str] | None = None) -> None:
        """Register a tool, optionally restricted to or excluded from session modes.

        Args:
            tool: The tool to register.
            mode: If set, tool is ONLY visible in this mode (e.g. "kevin").
            exclude_modes: If set, tool is hidden in these modes (e.g. ["kevin"]).
        """
        self._tools[tool.name] = tool
        if mode:
            self._modes[tool.name] = mode
        if exclude_modes:
            self._exclude_modes[tool.name] = set(exclude_modes)
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_definitions(self, mode: str | None = None) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI format, filtered by session mode.

        Args:
            mode: Current session mode (e.g. "kevin").  When set, mode-restricted
                  tools for *that* mode are included.  Tools restricted to a
                  *different* mode are excluded.  When None, only unrestricted
                  tools are returned.
        """
        return [
            tool.to_schema()
            for name, tool in self._tools.items()
            if (name not in self._modes or self._modes[name] == mode)
            and not (mode and name in self._exclude_modes and mode in self._exclude_modes[name])
        ]
    
    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a tool by name with given parameters.
        
        Args:
            name: Tool name.
            params: Tool parameters.
        
        Returns:
            Tool execution result as string.
        
        Raises:
            KeyError: If tool not found.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            return await tool.execute(**params)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
    
    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
