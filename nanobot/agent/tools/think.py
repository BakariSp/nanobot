"""Think tool — allows the agent to continue working without ending the turn.

When the agent outputs text without a tool call, the loop ends and the text
is sent to the user.  The think tool gives the agent a way to pause, plan,
and then continue calling other tools — without accidentally terminating the
turn by emitting visible text.
"""

from typing import Any

from nanobot.agent.tools.base import Tool


class ThinkTool(Tool):
    """No-op tool for inner monologue / planning within a turn."""

    @property
    def name(self) -> str:
        return "think"

    @property
    def description(self) -> str:
        return (
            "Use this to think through a problem or plan next steps WITHOUT "
            "ending your turn. Your thought will NOT be sent to the user. "
            "After thinking, you can continue calling other tools. "
            "Only output visible text when you are truly done and want to "
            "reply to the user."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your internal reasoning or plan. Not sent to the user.",
                },
            },
            "required": ["thought"],
        }

    async def execute(self, thought: str = "") -> str:
        return "Continue."
