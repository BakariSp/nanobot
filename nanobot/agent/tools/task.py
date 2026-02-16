"""Task management tool for structured task lifecycle operations."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.memory import MemoryStore


class TaskTool(Tool):
    """Manage the active task pointer and task files with code-enforced structure.

    Actions:
        create  — Create a new task file and set it as active
        switch  — Switch active pointer to an existing task (pause current)
        update  — Update the status line of the active task pointer
        complete — Mark active task done, archive to HISTORY.md, clear pointer
        list    — List all task files in memory/tasks/
    """

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._memory = MemoryStore(workspace)

    @property
    def name(self) -> str:
        return "task"

    @property
    def description(self) -> str:
        return (
            "Manage task lifecycle (create / switch / update / complete / list). "
            "Use this instead of manually editing MEMORY.md's Active Task section."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "switch", "update", "complete", "list"],
                    "description": "The task action to perform.",
                },
                "slug": {
                    "type": "string",
                    "description": "Task slug (kebab-case, e.g. 'teacher-ux-redesign'). Required for create/switch.",
                },
                "goal": {
                    "type": "string",
                    "description": "One-line task goal. Required for create.",
                },
                "status": {
                    "type": "string",
                    "description": "One-line status update. Required for update.",
                },
                "summary": {
                    "type": "string",
                    "description": "One-line completion summary for HISTORY.md. Required for complete.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]

        if action == "create":
            return self._create(kwargs.get("slug", ""), kwargs.get("goal", ""))
        elif action == "switch":
            return self._switch(kwargs.get("slug", ""))
        elif action == "update":
            return self._update(kwargs.get("status", ""))
        elif action == "complete":
            return self._complete(kwargs.get("summary", ""))
        elif action == "list":
            return self._list()
        else:
            return f"Error: Unknown action '{action}'"

    # ── Actions ──────────────────────────────────────────────

    def _create(self, slug: str, goal: str) -> str:
        if not slug:
            return "Error: 'slug' is required for create"
        if not goal:
            return "Error: 'goal' is required for create"

        # Sanitize slug
        slug = re.sub(r"[^a-z0-9\-]", "", slug.lower().replace(" ", "-"))
        if not slug:
            return "Error: slug must contain at least one alphanumeric character"

        task_file = self._memory.tasks_dir / f"{slug}.md"
        rel_path = f"memory/tasks/{slug}.md"

        if task_file.exists():
            return f"Error: Task file already exists: {rel_path}. Use 'switch' to resume it."

        # Create task file with template
        now = datetime.now().strftime("%Y-%m-%d")
        task_file.write_text(
            f"# Task: {goal}\n\n"
            f"Created: {now}\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Plan\n- [ ] (fill in steps)\n\n"
            f"## Key Decisions\n(none yet)\n",
            encoding="utf-8",
        )

        # Pause current task if any
        pause_msg = ""
        current_path = self._memory.get_active_task_path()
        if current_path:
            pause_msg = f"Paused previous task: {current_path.stem}\n"

        # Set pointer
        self._set_pointer(slug, rel_path, f"created — {goal[:60]}")

        return f"{pause_msg}Created task: {rel_path}\nActive task pointer updated."

    def _switch(self, slug: str) -> str:
        if not slug:
            return "Error: 'slug' is required for switch"

        slug = re.sub(r"[^a-z0-9\-]", "", slug.lower().replace(" ", "-"))
        task_file = self._memory.tasks_dir / f"{slug}.md"
        rel_path = f"memory/tasks/{slug}.md"

        if not task_file.exists():
            available = self._list()
            return f"Error: Task file not found: {rel_path}\n\n{available}"

        self._set_pointer(slug, rel_path, "resumed")
        return f"Switched active task to: {slug}"

    def _update(self, status: str) -> str:
        if not status:
            return "Error: 'status' is required for update"

        current = self._memory.read_long_term()
        task_path = self._memory._parse_active_task_file(current)
        if not task_path:
            return "Error: No active task to update. Use 'create' first."

        # Extract current slug from path
        slug = Path(task_path).stem

        self._set_pointer(slug, task_path, status[:80])
        return f"Active task status updated: {status[:80]}"

    def _complete(self, summary: str) -> str:
        if not summary:
            return "Error: 'summary' is required for complete"

        current = self._memory.read_long_term()
        task_path = self._memory._parse_active_task_file(current)
        if not task_path:
            return "Error: No active task to complete."

        slug = Path(task_path).stem

        # Archive to HISTORY.md
        today = datetime.now().strftime("%Y-%m-%d")
        self._memory.append_history(f"{today} | [DONE] {slug}: {summary}")

        # Clear pointer
        self._clear_pointer()

        return f"Task '{slug}' completed and archived to HISTORY.md.\nActive task pointer cleared."

    def _list(self) -> str:
        tasks = sorted(self._memory.tasks_dir.glob("*.md"))
        if not tasks:
            return "No task files found in memory/tasks/"

        current_path = self._memory.get_active_task_path()
        lines = ["Available tasks:"]
        for t in tasks:
            marker = " (active)" if current_path and t.resolve() == current_path.resolve() else ""
            lines.append(f"  - {t.stem}{marker}")
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────

    def _set_pointer(self, slug: str, rel_path: str, status: str) -> None:
        """Rewrite the ## Active Task section in MEMORY.md."""
        current = self._memory.read_long_term()
        new_section = (
            f"## Active Task\n"
            f"- task: {slug}\n"
            f"- file: {rel_path}\n"
            f"- status: {status}\n"
        )

        existing_section = self._memory._extract_section(current, "Active Task")
        if existing_section:
            updated = current.replace(existing_section, new_section)
        else:
            updated = current.rstrip() + "\n\n" + new_section

        self._memory.write_long_term(updated)

    def _clear_pointer(self) -> None:
        """Remove the Active Task section from MEMORY.md."""
        current = self._memory.read_long_term()
        new_section = "## Active Task\n- task: none\n"

        existing_section = self._memory._extract_section(current, "Active Task")
        if existing_section:
            updated = current.replace(existing_section, new_section)
        else:
            updated = current.rstrip() + "\n\n" + new_section

        self._memory.write_long_term(updated)
