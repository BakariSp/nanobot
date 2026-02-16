"""Tool for querying past worker dispatch tasks."""

from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.task_ledger import TaskLedger


class TaskQueryTool(Tool):
    """Query past worker dispatch tasks by ID or list recent tasks."""

    def __init__(self, ledger: TaskLedger):
        self._ledger = ledger

    @property
    def name(self) -> str:
        return "task_query"

    @property
    def description(self) -> str:
        return (
            "Query past worker dispatch tasks. "
            "Actions: 'get' (by task_id like W-001), 'list' (recent N tasks, default 10)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "list"],
                    "description": "Action: 'get' a specific task, or 'list' recent tasks",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (e.g. W-001) — required for 'get'",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of recent tasks to list (default 10)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self, action: str, task_id: str = "", count: int = 10, **kwargs: Any
    ) -> str:
        if action == "get":
            if not task_id:
                return "Error: task_id is required for 'get' action"
            record = self._ledger.get(task_id)
            if not record:
                return f"Task {task_id} not found"
            return self._ledger.render_markdown(record)

        if action == "list":
            records = self._ledger.list_recent(count)
            if not records:
                return "No worker tasks found."
            lines = []
            for r in records:
                lines.append(
                    f"- **{r.task_id}** [{r.status}] {r.task_preview} "
                    f"({r.duration_s}s, {r.model})"
                )
            return "Recent worker tasks:\n" + "\n".join(lines)

        return f"Unknown action: {action}. Use 'get' or 'list'."
