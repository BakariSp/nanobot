"""Agent loop: the core processing engine."""

import asyncio
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.prompts import diary_catchup_prompt, followup_nudge_context, startup_greeting_context, zero_system_prompt, jarvis_system_prompt
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.send_photo import SendPhotoTool
from nanobot.agent.tools.voice_reply import VoiceReplyTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


# ── Output sanitisation ──────────────────────────────────────────

# Matches timestamp prefixes that the LLM sometimes mimics from history,
# e.g. "[2026-02-17T01:00] " or "[2026-02-17 01:00] "
_TS_PREFIX_RE = re.compile(r"^\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}\]?\s*")


def _strip_ts_prefix(text: str) -> str:
    """Remove timestamp prefix from LLM output if the model copied it from history."""
    return _TS_PREFIX_RE.sub("", text).strip()


# ── Conversation archive (append-only, for future model training) ──

_CONVO_LOG_DIR = Path.home() / ".nanobot" / "logs" / "conversations"


def _archive_message(
    direction: str,
    channel: str,
    chat_id: str,
    content: str,
    *,
    sender_id: str = "",
    tools_used: list[str] | None = None,
    metadata: dict | None = None,
    reasoning_content: str | None = None,
) -> None:
    """Append a single message to the daily conversation JSONL file.

    File: ~/.nanobot/logs/conversations/YYYY-MM-DD.jsonl
    Each line is a self-contained JSON object, easy to grep / load for training.
    """
    _CONVO_LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    entry = {
        "ts": now.isoformat(),
        "ts_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": direction,  # "in" | "out" | "system"
        "channel": channel,
        "chat_id": chat_id,
        "sender_id": sender_id,
        "content": content,
    }
    if tools_used:
        entry["tools_used"] = tools_used
    if metadata:
        entry["metadata"] = metadata
    if reasoning_content:
        entry["reasoning_content"] = reasoning_content

    path = _CONVO_LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning(f"Conversation archive write failed: {e}")


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
        mcp_config: "Any | None" = None,  # MCPConfig from nanobot.mcp.types
        notion_config: "Any | None" = None,  # NotionConfig from nanobot.config.schema
        image_gen_config: "Any | None" = None,  # ImageGenConfig
        tts_config: "Any | None" = None,  # TTSToolConfig
        dashscope_api_key: str = "",  # Fallback for TTS if tts_config.dashscope_api_key is empty
    ):
        from nanobot.config.schema import ExecToolConfig, NotionConfig, ImageGenConfig, TTSToolConfig
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
        self.config_notion = notion_config or NotionConfig()
        self.config_image_gen = image_gen_config or ImageGenConfig()
        self.config_tts = tts_config or TTSToolConfig()
        self.dashscope_api_key = dashscope_api_key

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
        self._enabled_channels: set[str] | None = None  # Set after ChannelManager init
        self._register_default_tools()

        # MCP integration (lazy — connects in run())
        self._mcp_manager = None
        if mcp_config and mcp_config.enabled:
            from nanobot.mcp.manager import MCPManager
            self._mcp_manager = MCPManager(mcp_config, self.tools)

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

        # Notion tools (save pages + manage tasks)
        from nanobot.agent.tools.notion_save import SaveToNotionTool
        from nanobot.agent.tools.notion_tasks import NotionTasksTool

        notion_cfg = self.config_notion
        notion_tool: SaveToNotionTool | None = None
        if notion_cfg.api_token and notion_cfg.database_id:
            notion_tool = SaveToNotionTool(
                api_token=notion_cfg.api_token,
                database_id=notion_cfg.database_id,
            )
            self.tools.register(notion_tool)
            self.tools.register(NotionTasksTool(
                api_token=notion_cfg.api_token,
                database_id=notion_cfg.database_id,
            ))

        # Task orchestration tools (secretary architecture)
        from nanobot.agent.tools.worker import WorkerDispatchTool
        from nanobot.agent.tools.task import TaskTool
        from nanobot.agent.tools.task_query import TaskQueryTool
        from nanobot.agent.tools.worker_logs import WorkerLogsTool

        worker_tool = WorkerDispatchTool(
            default_cwd=str(self.workspace),
            ledger=self.task_ledger,
            dual_ledger=self.dual_ledger,
            notion_tool=notion_tool,
        )
        self.tools.register(worker_tool)

        self.tools.register(TaskTool(workspace=self.workspace))

        if self.task_ledger:
            self.tools.register(TaskQueryTool(ledger=self.task_ledger))

        if self.dual_ledger:
            self.tools.register(WorkerLogsTool(dual_ledger=self.dual_ledger))

        # Image generation tool (Volcengine Seedream)
        ig = self.config_image_gen
        if ig.ark_api_key:
            self.tools.register(SendPhotoTool(
                send_callback=self.bus.publish_outbound,
                ark_api_key=ig.ark_api_key,
                ark_base_url=ig.ark_base_url,
                ark_image_model=ig.ark_image_model,
            ))

        # Voice reply tool (TTS)
        tts_cfg = self.config_tts
        tts_api_key = tts_cfg.dashscope_api_key or self.dashscope_api_key
        if tts_cfg.provider == "volcengine" and tts_cfg.volcengine_app_id:
            from nanobot.providers.volcengine_tts import VolcengineTTSProvider
            tts_provider = VolcengineTTSProvider(
                app_id=tts_cfg.volcengine_app_id,
                token=tts_cfg.volcengine_token,
                default_voice=tts_cfg.default_voice or None,
            )
            self.tools.register(VoiceReplyTool(
                send_callback=self.bus.publish_outbound,
                tts_provider=tts_provider,
            ))
        elif tts_api_key:
            from nanobot.providers.tts import DashScopeTTSProvider
            tts_provider = DashScopeTTSProvider(api_key=tts_api_key)
            self.tools.register(VoiceReplyTool(
                send_callback=self.bus.publish_outbound,
                tts_provider=tts_provider,
            ))

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

        if photo_tool := self.tools.get("send_photo"):
            if isinstance(photo_tool, SendPhotoTool):
                photo_tool.set_context(channel, chat_id)

        if voice_tool := self.tools.get("voice_reply"):
            if isinstance(voice_tool, VoiceReplyTool):
                voice_tool.set_context(channel, chat_id)

    # ── Secretary architecture helpers ──────────────────────────────

    def _get_mode(self, session) -> str:
        """Get current mode (zero or jarvis) from session metadata."""
        return session.metadata.get("mode", "zero")

    def _set_mode(self, session, mode: str) -> None:
        """Set mode and persist to session."""
        session.metadata["mode"] = mode
        self.sessions.save(session)

    def _check_notifications(self) -> str:
        """Read and clear Ralph Loop completion notifications.

        For failed runs, includes failure diagnosis and a hint to use worker_logs
        for detailed inspection.
        """
        if not self.dual_ledger:
            return ""
        notifications = self.dual_ledger.read_and_clear_notifications()
        if not notifications:
            return ""
        lines = []
        for n in notifications:
            line = f"[{n['task_id']}/{n['run_id']}] {n['status']}: {n['summary']}"
            # Enrich failed notifications with run details
            if n['status'] == 'failed' and self.dual_ledger:
                run = self.dual_ledger.get_run(n['run_id'])
                if run:
                    if run.failure_diagnosis:
                        line += f" (diagnosis: {run.failure_diagnosis})"
                    line += f" — use worker_logs(get_output, {n['run_id']}) for full log"
            lines.append(line)
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

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str], str | None]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used, reasoning_content).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        last_reasoning: str | None = None

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.reasoning_content:
                last_reasoning = response.reasoning_content

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
                # ── Thinking-model truncation guard ───────────────
                # For models that return reasoning_content (DeepSeek-R1,
                # Kimi, etc.), max_tokens is the *total* budget for both
                # reasoning and visible content.  When the model exhausts
                # the budget on thinking, finish_reason is "length" and
                # the actual reply is truncated or garbled.
                # Retry once with a larger budget so the model can finish.
                if (
                    response.finish_reason == "length"
                    and response.reasoning_content
                ):
                    bumped = min(self.max_tokens * 2, 32768)
                    logger.warning(
                        f"Thinking model truncated (finish_reason=length, "
                        f"reasoning={len(response.reasoning_content)} chars), "
                        f"retrying with max_tokens={bumped}"
                    )
                    retry = await self.provider.chat(
                        messages=messages,
                        tools=self.tools.get_definitions(),
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=bumped,
                    )
                    if retry.reasoning_content:
                        last_reasoning = retry.reasoning_content
                    if retry.finish_reason != "length" and retry.content:
                        response = retry

                final_content = response.content
                break

        return final_content, tools_used, last_reasoning

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

        # ── Start MCP servers ──────────────────────────────────────
        if self._mcp_manager:
            try:
                await self._mcp_manager.start()
            except Exception as e:
                logger.error(f"MCP Manager start failed (non-fatal): {e}")

        # ── Restore active chats from session history & send startup greeting ──
        await self._startup_greet()

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
                        # Split on --- like startup/followup paths (multi-bubble)
                        parts = [p.strip() for p in re.split(r"\n---\n", response.content) if p.strip()]
                        for i, part in enumerate(parts):
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=response.channel,
                                chat_id=response.chat_id,
                                content=part,
                                metadata=response.metadata,
                            ))
                            if i < len(parts) - 1:
                                await asyncio.sleep(0.8)
                    else:
                        # [SILENT] — still notify channel to cancel typing indicator
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="",
                            metadata={"_silent": True},
                        ))
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

    async def shutdown(self) -> None:
        """Async shutdown — stops MCP servers and other async resources."""
        self.stop()
        if self._mcp_manager:
            try:
                await self._mcp_manager.stop()
            except Exception as e:
                logger.error(f"MCP Manager shutdown error: {e}")

    # ── Startup diary catch-up ────────────────────────────────

    async def _maybe_write_diary(self) -> None:
        """Check if yesterday's diary is missing and write it on startup."""
        from datetime import timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        diary_path = self.workspace / "memory" / "diary" / f"{yesterday}.md"
        if diary_path.exists():
            return

        logger.info(f"Diary catch-up: {yesterday}.md missing, triggering write")
        try:
            await self.process_direct(
                diary_catchup_prompt(yesterday),
                session_key="cron:diary-catchup",
            )
            logger.info(f"Diary catch-up: {yesterday} done")
        except Exception as e:
            logger.warning(f"Diary catch-up failed: {e}")

    # ── Startup greeting ───────────────────────────────────────

    async def _startup_greet(self) -> None:
        """Restore active chats from persisted sessions and send a startup greeting.

        Scans all saved sessions to find channel chats (not CLI), restores them
        into _active_chats, then asks the LLM to generate a natural greeting
        for each. The LLM decides whether the current time is appropriate —
        no hardcoded time windows.
        """
        # Catch up on missed diary before greeting
        await self._maybe_write_diary()

        now_mono = time.monotonic()
        restored = []

        for info in self.sessions.list_sessions():
            key = info.get("key", "")
            # Session keys are "channel:chat_id" — skip CLI sessions
            if not key or key.startswith("cli:") or key.startswith("cron:") or key.startswith("heartbeat"):
                continue
            parts = key.split(":", 1)
            if len(parts) != 2:
                continue
            channel, chat_id = parts
            # Skip channels that are not currently enabled
            if self._enabled_channels is not None and channel not in self._enabled_channels:
                logger.debug(f"Startup greet: skipping disabled channel {channel}:{chat_id}")
                continue
            chat_key = (channel, chat_id)
            self._active_chats.add(chat_key)
            self._last_user_msg_at[chat_key] = now_mono
            self._next_followup_at[chat_key] = now_mono + random.uniform(120, 600)
            restored.append((channel, chat_id, key))

        if not restored:
            logger.info("Startup greet: no previous chats to greet")
            return

        logger.info(f"Startup greet: restored {len(restored)} active chat(s), sending greetings...")

        for channel, chat_id, session_key in restored:
            session = self.sessions.get_or_create(session_key)
            if not session.messages:
                continue

            memory_context = self.context.memory.get_memory_context()
            system_prompt = zero_system_prompt(str(self.workspace), memory_context)
            max_history = min(self.memory_window, 20)
            history = session.get_history(max_messages=max_history)

            last_ts = session.messages[-1].get("timestamp", "") if session.messages else ""
            nudge = startup_greeting_context(last_message_ts=last_ts)
            messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": nudge})

            try:
                response = await self.provider.chat(
                    messages=messages,
                    tools=[],
                    model=self.model,
                    temperature=0.9,
                    max_tokens=256,
                )
                reply = _strip_ts_prefix((response.content or "").strip())
            except Exception as e:
                logger.warning(f"Startup greeting failed for {session_key}: {e}")
                continue

            if not reply or "[SKIP]" in reply:
                logger.debug(f"Startup greeting skipped for {session_key} (LLM decided)")
                continue

            logger.info(f"Startup greeting to {session_key}: {reply[:80]}")
            session.add_message("assistant", reply)
            self.sessions.save(session)
            _archive_message("out", channel, chat_id, reply,
                             metadata={"trigger": "startup_greeting"})

            parts = [p.strip() for p in re.split(r"\n---\n", reply) if p.strip()]
            for i, part in enumerate(parts):
                await self.bus.publish_outbound(OutboundMessage(
                    channel=channel, chat_id=chat_id, content=part,
                ))
                if i < len(parts) - 1:
                    await asyncio.sleep(0.8)

    # ── Proactive conversation follow-up ─────────────────────────

    async def _maybe_conversation_followup(self) -> None:
        """Check if Zero should proactively follow up on any silent conversation.

        Uses randomised timing and lets the LLM decide whether there's
        something worth saying — producing natural, personality-consistent
        follow-ups rather than hardcoded nudges.

        No hardcoded time restriction — the LLM decides based on MEMORY.md
        whether the current time is appropriate to reach out.

        Timing (all randomised per-chat):
          - After a user message:  first consideration in 2-10 min
          - After Zero follows up: cooldown 20-45 min
          - After LLM says [SKIP]: cooldown 10-30 min
        """
        now = time.monotonic()

        for chat_key in list(self._active_chats):
            next_at = self._next_followup_at.get(chat_key)
            if next_at is None or now < next_at:
                continue

            last_msg_at = self._last_user_msg_at.get(chat_key)
            if last_msg_at is None:
                continue

            channel, chat_id = chat_key

            # Skip channels that are not currently enabled (avoid wasting LLM calls)
            if self._enabled_channels is not None and channel not in self._enabled_channels:
                continue

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
                reply = _strip_ts_prefix((response.content or "").strip())
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
            _archive_message("out", channel, chat_id, reply,
                             metadata={"trigger": "followup", "silent_minutes": silent_minutes})

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

        # Archive inbound message
        _archive_message("in", msg.channel, msg.chat_id, msg.content,
                         sender_id=msg.sender_id)

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
        final_content, tools_used, reasoning = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Strip timestamp prefixes the LLM may have mimicked from history
        final_content = _strip_ts_prefix(final_content)

        # [SILENT] — Zero chose not to reply (e.g. user hasn't finished typing).
        # Record in session history so she has context, but send nothing.
        is_silent = "[SILENT]" in final_content

        session.add_message("user", msg.content)
        if not is_silent:
            session.add_message("assistant", final_content,
                                tools_used=tools_used if tools_used else None,
                                reasoning_content=reasoning)
        self.sessions.save(session)

        if is_silent:
            logger.info(f"[SILENT] for {msg.channel}:{msg.sender_id} — no outbound")
            _archive_message("out", msg.channel, msg.chat_id, "[SILENT]")
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        # Archive outbound response
        _archive_message("out", msg.channel, msg.chat_id, final_content,
                         tools_used=tools_used if tools_used else None,
                         reasoning_content=reasoning)

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
        final_content, _, _ = await self._run_agent_loop(initial_messages)

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
