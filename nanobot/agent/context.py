"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
    
    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
        
        Returns:
            Complete system prompt.
        """
        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Memory context
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        # Boot recovery context (plugin changelog + crash correlation)
        recovery = self._build_recovery_context()
        if recovery:
            parts.append(f"# Boot Recovery\n\n{recovery}")

        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Runtime context only — personality & instructions come from SOUL.md / AGENTS.md."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# Runtime

## Current Time
{now} ({tz})

## Environment
{runtime}

## Workspace
{workspace_path}"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""

    def _build_recovery_context(self) -> str:
        """Build boot recovery context from plugin changelog and doctor state.

        Injected into the system prompt so the agent knows what happened
        before the last restart (especially crash-related plugin rollbacks).
        """
        lines: list[str] = []

        try:
            from nanobot.agent.tools.plugin_edit import read_changelog
            entries = read_changelog(limit=5)
            if entries:
                lines.append("## Recent plugin edits")
                for e in entries:
                    ts = e.get("ts", "?")[:19]
                    fname = Path(e.get("file", "?")).name
                    reason = e.get("reason", "")
                    status = e.get("status", "?")
                    err = e.get("validation_error", "")
                    line = f"- [{ts}] {fname}: {reason} ({status})"
                    if err:
                        line += f" — {err}"
                    lines.append(line)
        except Exception:
            pass

        try:
            from nanobot.doctor.state import load_state
            state = load_state()
            crash_ts = state.get("last_crash_ts")
            if crash_ts:
                lines.append(f"\n## Last crash: {crash_ts[:19]}")
                total = state.get("total_crashes", 0)
                if total:
                    lines.append(f"Total crashes recorded: {total}")
        except Exception:
            pass

        return "\n".join(lines) if lines else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        override_system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.
            override_system_prompt: If provided, use this instead of building from bootstrap files.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = override_system_prompt or self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History — re-inject media (OSS URLs) into multimodal format
        for h in history:
            if h.get("media") and h["role"] == "user":
                content = self._build_user_content(h["content"], h["media"])
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": h["role"], "content": h["content"]})

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional images.

        Supports two media reference formats:
        - URL (``https://...``) → passed directly as ``image_url``
        - Local path → base64-encoded as data URL (fallback)
        """
        if not media:
            return text

        images = []
        for ref in media:
            if ref.startswith("https://") or ref.startswith("http://"):
                # OSS or other public URL — pass directly
                images.append({"type": "image_url", "image_url": {"url": ref}})
            else:
                # Local file — base64 encode
                p = Path(ref)
                mime, _ = mimetypes.guess_type(ref)
                if not p.is_file() or not mime or not mime.startswith("image/"):
                    continue
                b64 = base64.b64encode(p.read_bytes()).decode()
                images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        
        if tool_calls:
            msg["tool_calls"] = tool_calls
        
        # Thinking models reject history without this
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        
        messages.append(msg)
        return messages
