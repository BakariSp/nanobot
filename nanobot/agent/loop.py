"""Agent loop: the core processing engine."""

import asyncio
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.prompts import followup_nudge_context, zero_system_prompt, jarvis_system_prompt
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


# ── Secretary command parsing ────────────────────────────────────

_COMMANDS = {
    r"^/p$":                      ("mode_switch", lambda m: "jarvis"),
    r"^/jarvis$":                 ("mode_switch", lambda m: "jarvis"),
    r"^/s$":                      ("mode_switch", lambda m: "zero"),
    r"^/status$":                 ("status", lambda m: None),
    r"^/approve\s+(T-\d+)$":     ("approve", lambda m: m.group(1)),
    r"^/pause\s+(T-\d+)$":       ("pause", lambda m: m.group(1)),
    r"^/cancel\s+(T-\d+)$":      ("cancel", lambda m: m.group(1)),
}


def parse_command(text: str) -> tuple[str, Any] | None:
    """Parse secretary commands. Returns (action, arg) or None if not a command."""
    text = text.strip()
    for pattern, (action, extractor) in _COMMANDS.items():
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            return (action, extractor(m))
    return None


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        task_ledger: "TaskLedger | None" = None,
        dual_ledger: "DualLedger | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.task_ledger = task_ledger
        self.dual_ledger = dual_ledger

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Task orchestration tools (secretary architecture)
        from nanobot.agent.tools.worker import WorkerDispatchTool
        from nanobot.agent.tools.task import TaskTool
        from nanobot.agent.tools.task_query import TaskQueryTool

        worker_tool = WorkerDispatchTool(
            default_cwd=str(self.workspace),
            ledger=self.task_ledger,
            dual_ledger=self.dual_ledger,
        )
        self.tools.register(worker_tool)

        self.tools.register(TaskTool(workspace=self.workspace))

        if self.task_ledger:
            self.tools.register(TaskQueryTool(ledger=self.task_ledger))

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    # ── Secretary architecture helpers ──────────────────────────────

    def _get_mode(self, session) -> str:
        """Get current mode (zero or jarvis) from session metadata."""
        return session.metadata.get("mode", "zero")

    def _set_mode(self, session, mode: str) -> None:
        """Set mode and persist to session."""
        session.metadata["mode"] = mode
        self.sessions.save(session)

    def _check_notifications(self) -> str:
        """Read and clear Ralph Loop completion notifications."""
        if not self.dual_ledger:
            return ""
        notifications = self.dual_ledger.read_and_clear_notifications()
        if not notifications:
            return ""
        lines = []
        for n in notifications:
            lines.append(f"[{n['task_id']}/{n['run_id']}] {n['status']}: {n['summary']}")
        return "\n".join(lines)

    def _build_mode_prompt(self, session) -> str:
        """Build mode-specific system prompt (零号 or Jarvis).

        For Zero mode, also loads USER.md, AGENTS.md, and the active task file
        so they are part of the context (these are skipped when override_system_prompt
        bypasses ContextBuilder.build_system_prompt).
        """
        mode = self._get_mode(session)
        memory_context = self.context.memory.get_memory_context()
        if mode == "jarvis":
            status_context = self.dual_ledger.status_summary() if self.dual_ledger else ""
            return jarvis_system_prompt(str(self.workspace), status_context)

        base = zero_system_prompt(str(self.workspace), memory_context)

        # Append workspace bootstrap files that zero_system_prompt doesn't include
        extras = []
        for filename in ("USER.md", "AGENTS.md"):
            path = self.workspace / filename
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        extras.append(f"## {filename}\n{content}")
                except OSError:
                    pass

        # Auto-inject active task file so Zero has task context without manual read_file
        task_path = self.context.memory.get_active_task_path()
        if task_path:
            try:
                task_content = task_path.read_text(encoding="utf-8").strip()
                if task_content:
                    rel = task_path.relative_to(self.workspace)
                    extras.append(f"## Active Task File ({rel})\n{task_content}")
            except (OSError, ValueError):
                pass

        if extras:
            base += "\n\n" + "\n\n".join(extras)

        return base

    async def _handle_command(self, action: str, arg: Any, session, msg: InboundMessage) -> OutboundMessage:
        """Handle secretary commands (bypass LLM, < 100ms)."""
        if action == "mode_switch":
            self._set_mode(session, arg)
            if arg == "jarvis":
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Jarvis here. What do you want to look at?")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="好的，我回来了。")

        if action == "status":
            summary = self.dual_ledger.status_summary() if self.dual_ledger else "Task system not active."
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=summary)

        if action in ("approve", "pause", "cancel"):
            task_id = arg
            if not self.dual_ledger:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Task system not active.")
            task = self.dual_ledger.get_task(task_id)
            if not task:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"{task_id} doesn't exist.")

            if action == "approve":
                if task.status != "pending_approval":
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                          content=f"{task_id} is already {task.status}, no approval needed.")
                self.dual_ledger.update_task_status(task_id, "todo")
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"{task_id} approved, queued for execution.")

            if action == "pause":
                if task.status in ("done", "cancelled"):
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                          content=f"{task_id} is already {task.status}, can't pause.")
                self.dual_ledger.update_task_status(task_id, "paused")
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"{task_id} paused.")

            # cancel
            if task.status in ("done", "cancelled"):
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"{task_id} is already {task.status}, can't cancel.")
            self.dual_ledger.update_task_status(task_id, "cancelled")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"{task_id} cancelled.")

        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                              content=f"Unknown command: {action}")

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                final_content = response.content
                break

        return final_content, tools_used

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True

        # ── Conversation follow-up state ──────────────────────────
        # Tracks which chats have been active this session
        self._active_chats: set[tuple[str, str]] = set()
        # Tracks when each chat last received a user message (monotonic time)
        self._last_user_msg_at: dict[tuple[str, str], float] = {}
        # Tracks when to next *consider* a follow-up per chat (monotonic time)
        self._next_followup_at: dict[tuple[str, str], float] = {}

        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                # Track active chats + schedule follow-up window
                chat_key = (msg.channel, msg.chat_id)
                self._active_chats.add(chat_key)
                now_mono = time.monotonic()
                self._last_user_msg_at[chat_key] = now_mono
                self._next_followup_at[chat_key] = now_mono + random.uniform(120, 600)

                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                # No message arrived — check if any conversation needs a follow-up
                await self._maybe_conversation_followup()
                continue

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    # ── Proactive conversation follow-up ─────────────────────────

    async def _maybe_conversation_followup(self) -> None:
        """Check if Zero should proactively follow up on any silent conversation.

        Uses randomised timing and lets the LLM decide whether there's
        something worth saying — producing natural, personality-consistent
        follow-ups rather than hardcoded nudges.

        Timing (all randomised per-chat):
          - After a user message:  first consideration in 2-10 min
          - After Zero follows up: cooldown 20-45 min
          - After LLM says [SKIP]: cooldown 10-30 min
        """
        now = time.monotonic()

        # Only during reasonable hours (9 AM - 10 PM)
        hour = datetime.now().hour
        if not (9 <= hour <= 22):
            return

        for chat_key in list(self._active_chats):
            next_at = self._next_followup_at.get(chat_key)
            if next_at is None or now < next_at:
                continue

            last_msg_at = self._last_user_msg_at.get(chat_key)
            if last_msg_at is None:
                continue

            channel, chat_id = chat_key
            silent_seconds = now - last_msg_at
            silent_minutes = int(silent_seconds / 60)

            # Need at least ~2 min of silence
            if silent_minutes < 2:
                self._next_followup_at[chat_key] = now + random.uniform(60, 180)
                continue

            # Build full system prompt + session history (same path as normal messages)
            session_key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(session_key)

            # Skip if the session is empty (no prior conversation to follow up on)
            if not session.messages:
                self._next_followup_at[chat_key] = now + random.uniform(600, 1800)
                continue

            # Use 零号's prompt for follow-ups (always in secretary mode)
            memory_context = self.context.memory.get_memory_context()
            system_prompt = zero_system_prompt(str(self.workspace), memory_context)
            max_history = min(self.memory_window, 20)
            history = session.get_history(max_messages=max_history)

            # Include task status in follow-up context
            task_summary = self.dual_ledger.status_summary() if self.dual_ledger else ""
            nudge = followup_nudge_context(silent_minutes, task_summary=task_summary)
            messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": nudge})

            try:
                response = await self.provider.chat(
                    messages=messages,
                    tools=[],  # No tools for follow-up — just text
                    model=self.model,
                    temperature=0.9,  # Slightly more creative for casual chat
                    max_tokens=256,
                )
                reply = (response.content or "").strip()
            except Exception as e:
                logger.warning(f"Follow-up LLM call failed for {session_key}: {e}")
                self._next_followup_at[chat_key] = now + random.uniform(600, 1800)
                continue

            if not reply or "[SKIP]" in reply:
                # LLM decided not to say anything — medium cooldown
                self._next_followup_at[chat_key] = now + random.uniform(600, 1800)
                logger.debug(f"Follow-up skipped for {session_key} (silent {silent_minutes}m)")
                continue

            # Send the follow-up and record it in session history
            logger.info(f"Follow-up to {session_key} (silent {silent_minutes}m): {reply[:80]}")
            session.add_message("assistant", reply)
            self.sessions.save(session)

            # Split on --- like normal messages (multi-bubble)
            parts = [p.strip() for p in re.split(r"\n---\n", reply) if p.strip()]
            for i, part in enumerate(parts):
                await self.bus.publish_outbound(OutboundMessage(
                    channel=channel, chat_id=chat_id, content=part,
                ))
                if i < len(parts) - 1:
                    await asyncio.sleep(0.8)

            # Long cooldown after actually sending something
            self._next_followup_at[chat_key] = now + random.uniform(1200, 2700)

    # ── Message processing ───────────────────────────────────────

    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).

        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/archive":
            if not session.messages:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Nothing to archive — session is empty.")
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _archive_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_archive_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="Conversation archived to memory. Session cleared.")
        if cmd == "/discard":
            if not session.messages:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Nothing to discard — session is empty.")
            count = len(session.messages)
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Discarded {count} messages. Session cleared, nothing saved to memory.")
        if cmd == "/help":
            mode = self._get_mode(session)
            mode_label = "Jarvis (planner)" if mode == "jarvis" else "零号 (secretary)"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"\U0001f408 nanobot commands:\n"
                                          f"/new \u2014 New conversation (archive + clear)\n"
                                          f"/archive \u2014 Archive conversation to memory\n"
                                          f"/discard \u2014 Discard conversation (no memory)\n"
                                          f"/p \u2014 Switch to Jarvis (planner)\n"
                                          f"/s \u2014 Switch to 零号 (secretary)\n"
                                          f"/status \u2014 Task queue status\n"
                                          f"/approve T-xxx \u2014 Approve L3 task\n"
                                          f"/pause T-xxx \u2014 Pause task\n"
                                          f"/cancel T-xxx \u2014 Cancel task\n"
                                          f"\nCurrent mode: {mode_label}")

        # Secretary architecture commands (bypass LLM, < 100ms)
        parsed = parse_command(msg.content)
        if parsed:
            action, arg = parsed
            return await self._handle_command(action, arg, session, msg)

        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        # Check Ralph Loop notifications
        notification_context = self._check_notifications()

        # Build mode-specific system prompt (零号 or Jarvis)
        system_prompt = self._build_mode_prompt(session)
        if notification_context:
            system_prompt += f"\n\n## Recent Task Completions\n{notification_context}"

        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            override_system_prompt=system_prompt,
        )
        final_content, tools_used = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."

        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        # Load SOUL.md and USER.md so the consolidation LLM knows what NOT to duplicate
        soul_content = ""
        user_content = ""
        try:
            soul_path = self.workspace / "SOUL.md"
            if soul_path.exists():
                soul_content = soul_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        try:
            user_path = self.workspace / "USER.md"
            if user_path.exists():
                user_content = user_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass

        dedup_section = ""
        if soul_content or user_content:
            dedup_section = "\n## Already Stored Elsewhere (DO NOT duplicate into memory_update)\n"
            if soul_content:
                dedup_section += f"\n### SOUL.md (agent identity — canonical source)\n{soul_content}\n"
            if user_content:
                dedup_section += f"\n### USER.md (user profile — canonical source)\n{user_content}\n"

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Only add NEW facts discovered in the conversation: evolving user behavior patterns, project runtime discoveries, technical decisions, task context. If nothing new, return the existing content unchanged.

CRITICAL RULES for memory_update:
- DO NOT repeat information already in SOUL.md or USER.md (shown below). Those files are the canonical source for identity and user profile.
- DO NOT store static facts like agent name, age, location, personality, or user name/language/timezone — those belong in SOUL.md / USER.md.
- ONLY store: learned behavior patterns, project-specific discoveries, evolved rules, active task state.
- If existing memory_update contains duplicates of SOUL.md/USER.md content, REMOVE them.
{dedup_section}
## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).

        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )

        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""
