"""Memory system for persistent agent memory."""

import re
from pathlib import Path

from nanobot.utils.helpers import ensure_dir


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.tasks_dir = ensure_dir(self.memory_dir / "tasks")

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def get_active_task_path(self) -> Path | None:
        """Return the Path to the active task file, or None."""
        rel = self._parse_active_task_file(self.read_long_term())
        if not rel:
            return None
        p = self.workspace / rel
        return p if p.exists() else None

    def _parse_active_task_file(self, content: str) -> str | None:
        """Extract the 'file:' value from the Active Task section."""
        section = self._extract_section(content, "Active Task")
        if not section:
            return None
        m = re.search(r"^- file:\s*(.+)$", section, re.MULTILINE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_section(content: str, heading: str) -> str | None:
        """Extract the full text of a ## heading section (heading + body)."""
        pattern = rf"(## {re.escape(heading)}\n(?:(?!## ).+\n?)*)"
        m = re.search(pattern, content)
        return m.group(1) if m else None
