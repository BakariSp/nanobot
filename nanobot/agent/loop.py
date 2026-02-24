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
from nanobot.agent.prompts import diary_catchup_prompt, followup_nudge_context, idle_activity_prompt, narrator_summary_prompt, startup_greeting_context, zero_system_prompt, jarvis_system_prompt, kevin_system_prompt, kevin_wakeup_prompt
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.plugin_edit import PluginEditTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.think import ThinkTool
from nanobot.agent.tools.send_photo import SendPhotoTool
from nanobot.agent.tools.voice_reply import VoiceReplyTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


# ── Output sanitisation ──────────────────────────────────────────

# Matches timestamp prefixes that the LLM sometimes mimics from history,
# e.g. "[2026-02-17T01:00] " or "[2026-02-17 01:00] "
_TS_PREFIX_RE = re.compile(r"^\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}\]?\s*")

# Matches tool-call-like wrappers that thinking models sometimes write as
# plain text instead of using the proper function-calling API.
# e.g. message("你好") or message("你好", emotion="happy")
_TOOL_TEXT_RE = re.compile(
    r'^(message|voice_reply)\(\s*"([\s\S]+?)"\s*(?:,\s*.+?)?\)\s*$'
)


def _strip_ts_prefix(text: str) -> str:
    """Remove timestamp prefix from LLM output if the model copied it from history."""
    return _TS_PREFIX_RE.sub("", text).strip()


def _strip_tool_text_wrapper(text: str) -> str:
    """Unwrap tool-call syntax that the LLM wrote as plain text.

    Some thinking models (Qwen, DeepSeek) output ``message("内容")``
    instead of making a proper tool call.  Extract the inner content
    so the user sees clean text.
    """
    m = _TOOL_TEXT_RE.match(text.strip())
    if m:
        return m.group(2)
    return text


# Broad keyword pre-filter for hallucination check.
# Intentionally permissive — the LLM does the actual judgment.
_ACTION_KEYWORDS = ("worker", "派", "dispatch", "T-0", "W-0", "安排", "任务已")


def _needs_hallucination_check(content: str, tools_used: list[str]) -> bool:
    """Quick pre-filter: should we bother running the LLM hallucination check?

    Returns False if dispatch_worker was actually used, or if the response
    doesn't mention any action-related keywords at all.
    """
    if "dispatch_worker" in tools_used:
        return False
    content_lower = content.lower()
    return any(kw in content_lower for kw in _ACTION_KEYWORDS)


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
        memory_window: int = 500,
        brave_api_key: str | None = None,
        google_cse_api_key: str | None = None,
        google_cse_cx: str | None = None,
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
        kevin_config: "Any | None" = None,  # KevinConfig from nanobot.config.schema
    ):
        from nanobot.config.schema import ExecToolConfig, NotionConfig, ImageGenConfig, TTSToolConfig, KevinConfig
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
        self.google_cse_api_key = google_cse_api_key
        self.google_cse_cx = google_cse_cx
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.task_ledger = task_ledger
        self.dual_ledger = dual_ledger
        self.config_notion = notion_config or NotionConfig()
        self.config_image_gen = image_gen_config or ImageGenConfig()
        self.config_tts = tts_config or TTSToolConfig()
        self.dashscope_api_key = dashscope_api_key
        self.config_kevin = kevin_config or KevinConfig()

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self._consolidating: set[str] = set()  # guard against duplicate consolidation tasks
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            google_cse_api_key=google_cse_api_key,
            google_cse_cx=google_cse_cx,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        # Kevin (autonomous trader) — lazy init
        self._kevin_client = None
        self._kevin_state = None

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

        # Shell tool (Kevin doesn't need shell access)
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ), exclude_modes=["kevin"])

        # Web tools — only register web_search when an external API key is
        # actually configured.  When absent, the LLM's built-in inline search
        # (e.g. DashScope qwen3.5-plus enable_search) is used instead.
        if self.brave_api_key or self.google_cse_api_key:
            self.tools.register(WebSearchTool(
                api_key=self.brave_api_key,
                google_cse_api_key=self.google_cse_api_key,
                google_cse_cx=self.google_cse_cx,
            ))
        self.tools.register(WebFetchTool())

        # Message tool (Kevin doesn't message users directly)
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool, exclude_modes=["kevin"])

        # Spawn tool (for subagents — not for Kevin)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool, exclude_modes=["kevin"])

        # Cron tool (Kevin's cron is managed exclusively by end_turn)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service), exclude_modes=["kevin"])

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
            self.tools.register(notion_tool, exclude_modes=["kevin"])
            self.tools.register(NotionTasksTool(
                api_token=notion_cfg.api_token,
                database_id=notion_cfg.database_id,
            ), exclude_modes=["kevin"])

        # Task orchestration tools (secretary architecture)
        from nanobot.agent.tools.worker import WorkerDispatchTool
        from nanobot.agent.tools.task import TaskTool
        from nanobot.agent.tools.task_query import TaskQueryTool
        from nanobot.agent.tools.worker_logs import WorkerLogsTool
        from nanobot.agent.tools.worker_input import WorkerReplyTool

        # Wire SubagentManager ↔ DualLedger for conversational workers
        if self.dual_ledger:
            self.subagents.set_dual_ledger(self.dual_ledger)

        worker_tool = WorkerDispatchTool(
            default_cwd=str(self.workspace),
            ledger=self.task_ledger,
            dual_ledger=self.dual_ledger,
            notion_tool=notion_tool,
            subagent_manager=self.subagents,
        )
        self.tools.register(worker_tool, exclude_modes=["kevin"])

        self.tools.register(TaskTool(workspace=self.workspace), exclude_modes=["kevin"])

        if self.task_ledger:
            self.tools.register(TaskQueryTool(ledger=self.task_ledger), exclude_modes=["kevin"])

        if self.dual_ledger:
            self.tools.register(WorkerLogsTool(dual_ledger=self.dual_ledger), exclude_modes=["kevin"])

        # Worker reply tool — lets Zero answer worker questions
        self.tools.register(WorkerReplyTool(manager=self.subagents), exclude_modes=["kevin"])

        # Think tool (inner monologue — keeps the loop alive without ending the turn)
        self.tools.register(ThinkTool())

        # Plugin self-edit tool (sandboxed modification of providers/peripheral tools)
        self.tools.register(PluginEditTool(), exclude_modes=["kevin"])

        # Image generation tool (Volcengine Seedream)
        ig = self.config_image_gen
        if ig.ark_api_key:
            self.tools.register(SendPhotoTool(
                send_callback=self.bus.publish_outbound,
                ark_api_key=ig.ark_api_key,
                ark_base_url=ig.ark_base_url,
                ark_image_model=ig.ark_image_model,
            ), exclude_modes=["kevin"])

        # Voice reply tool (TTS)
        tts_cfg = self.config_tts
        tts_api_key = tts_cfg.dashscope_api_key or self.dashscope_api_key
        if tts_cfg.provider == "volcengine" and tts_cfg.volcengine_app_id:
            from nanobot.providers.volcengine_tts import VolcengineTTSProvider
            tts_provider = VolcengineTTSProvider(
                app_id=tts_cfg.volcengine_app_id,
                token=tts_cfg.volcengine_token,
                default_voice=tts_cfg.default_voice or "zh_female_vv_uranus_bigtts",
                resource_id=tts_cfg.volcengine_resource_id,
            )
            self.tools.register(VoiceReplyTool(
                send_callback=self.bus.publish_outbound,
                tts_provider=tts_provider,
            ), exclude_modes=["kevin"])
        elif tts_api_key:
            from nanobot.providers.tts import DashScopeTTSProvider
            tts_provider = DashScopeTTSProvider(api_key=tts_api_key)
            self.tools.register(VoiceReplyTool(
                send_callback=self.bus.publish_outbound,
                tts_provider=tts_provider,
            ), exclude_modes=["kevin"])

        # Kevin (autonomous crypto trader — Bybit)
        from nanobot.agent.tools.kevin_tools import KevinStatusTool
        from nanobot.kevin.state import KevinState

        kevin_cfg = self.config_kevin
        if kevin_cfg.enabled and kevin_cfg.bybit_api_key:
            try:
                from nanobot.kevin.client import BybitClient
                from nanobot.kevin.signals import CryptoSignals
                from nanobot.agent.tools.kevin_tools import (
                    BybitTradeTool, BybitBalanceTool, BybitTickerTool,
                    BybitPositionTool, CryptoSentimentTool, KevinEndTurnTool,
                    CalculateTool,
                )
                self._kevin_client = BybitClient(
                    api_key=kevin_cfg.bybit_api_key,
                    api_secret=kevin_cfg.bybit_api_secret,
                    testnet=kevin_cfg.bybit_testnet,
                )
                self._kevin_state = KevinState(self.workspace)
                self._kevin_signals = CryptoSignals(bybit_client=self._kevin_client)

                # Seed initial balance if configured and not yet set
                portfolio = self._kevin_state.get_portfolio()
                if kevin_cfg.initial_balance > 0 and portfolio.get("initial_balance", 0) == 0:
                    try:
                        balance = self._kevin_client.get_balance("USDT")
                        self._kevin_state.update_portfolio(balance, initial_balance=kevin_cfg.initial_balance)
                    except Exception as e:
                        logger.warning(f"Kevin balance seed failed (non-fatal): {e}")
                        self._kevin_state.update_portfolio(0.0, initial_balance=kevin_cfg.initial_balance)

                self.tools.register(BybitTradeTool(self._kevin_client, self._kevin_state), mode="kevin")
                self.tools.register(BybitBalanceTool(self._kevin_client, self._kevin_state), mode="kevin")
                self.tools.register(BybitTickerTool(self._kevin_client), mode="kevin")
                self.tools.register(BybitPositionTool(self._kevin_client), mode="kevin")
                self.tools.register(CryptoSentimentTool(self._kevin_signals), mode="kevin")
                self.tools.register(CalculateTool(self._kevin_client), mode="kevin")
                self.tools.register(KevinStatusTool(self._kevin_state))  # available to Zero too
                if self.cron_service:
                    self.tools.register(KevinEndTurnTool(self._kevin_state, self.cron_service), mode="kevin")
                logger.info("Kevin tools registered (Bybit crypto trading enabled)")
            except ImportError as e:
                logger.warning(f"Kevin enabled but deps missing: {e}. Install with: pip install 'nanobot-ai[kevin]'")
                self._kevin_state = KevinState(self.workspace)
                self.tools.register(KevinStatusTool(self._kevin_state))
            except Exception as e:
                logger.error(f"Kevin init failed: {e}")
                self._kevin_state = KevinState(self.workspace)
                self.tools.register(KevinStatusTool(self._kevin_state))
        else:
            # Kevin disabled — still register status tool so Zero can report it
            self.tools.register(KevinStatusTool(None))

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

        # Worker dispatch needs origin for conversational worker notifications
        from nanobot.agent.tools.worker import WorkerDispatchTool
        if worker_tool := self.tools.get("dispatch_worker"):
            if isinstance(worker_tool, WorkerDispatchTool):
                worker_tool.set_origin(channel, chat_id)

    # ── Plugin crash rollback ─────────────────────────────────────

    def _check_plugin_crash_rollback(self) -> None:
        """On boot, check if the last plugin edit likely caused a crash and auto-rollback."""
        try:
            from nanobot.doctor.state import load_state, save_state
            from nanobot.agent.tools.plugin_edit import _append_changelog, _restore_snapshot

            state = load_state()
            edit_info = state.get("last_plugin_edit")
            crash_ts_str = state.get("last_crash_ts")

            if not edit_info or not crash_ts_str:
                return

            from datetime import datetime, timezone
            edit_ts = datetime.fromisoformat(edit_info["ts"])
            crash_ts = datetime.fromisoformat(crash_ts_str)

            # Both must be tz-aware for comparison
            if edit_ts.tzinfo is None:
                edit_ts = edit_ts.replace(tzinfo=timezone.utc)
            if crash_ts.tzinfo is None:
                crash_ts = crash_ts.replace(tzinfo=timezone.utc)

            # Crash happened after the edit and within 2 minutes → likely causal
            delta = (crash_ts - edit_ts).total_seconds()
            if delta < 0 or delta > 120:
                return

            snapshot_path = Path(edit_info["snapshot_path"])
            target_path = Path(edit_info["file"])

            if not snapshot_path.exists():
                logger.warning(f"Plugin crash rollback: snapshot missing {snapshot_path}")
                return

            if not target_path.exists():
                logger.warning(f"Plugin crash rollback: target missing {target_path}")
                return

            # Perform rollback
            if _restore_snapshot(snapshot_path, target_path):
                logger.warning(
                    f"Plugin crash rollback: restored {target_path.name} from "
                    f"{snapshot_path.name} (crash {delta:.0f}s after edit)"
                )
                _append_changelog({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "file": str(target_path),
                    "reason": f"auto-rollback: crash {delta:.0f}s after edit '{edit_info.get('reason', '?')}'",
                    "status": "auto_rolled_back",
                })

            # Clear the last_plugin_edit so we don't rollback again
            state["last_plugin_edit"] = None
            save_state(state)

        except Exception as e:
            logger.error(f"Plugin crash rollback check failed: {e}")

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

        Returns a rich notification block per task so Zero can read the raw log
        and summarize findings to Cai instead of parroting a one-line summary.
        """
        if not self.dual_ledger:
            return ""
        notifications = self.dual_ledger.read_and_clear_notifications()
        if not notifications:
            return ""
        blocks: list[str] = []
        for n in notifications:
            task_id = n["task_id"]
            run_id = n["run_id"]
            status = n.get("status", "unknown")
            goal = n.get("goal", "")
            duration = n.get("duration_s", 0)
            raw_log = n.get("raw_log_path", "")
            notion_url = n.get("notion_page_url", "")
            diagnosis = n.get("failure_diagnosis", "")

            # Fall back to ledger if notification was written by older code
            if not goal and self.dual_ledger:
                task = self.dual_ledger.get_task(task_id)
                if task:
                    goal = task.goal
            if not raw_log:
                from pathlib import Path
                candidate = Path.home() / ".nanobot" / "data" / "reports" / f"{run_id}.raw.log"
                if candidate.exists():
                    raw_log = str(candidate)

            status_icon = "success" if status == "success" else "FAILED"
            lines = [
                f"Worker Task Update [{task_id}/{run_id}]",
                f"Goal: {goal[:150]}" if goal else "",
                f"Status: {status_icon} | Duration: {duration:.0f}s",
            ]
            if raw_log:
                lines.append(f"Raw log: {raw_log}")
            if notion_url:
                lines.append(f"Notion: {notion_url}")
            if status == "failed":
                if diagnosis:
                    lines.append(f"Failure diagnosis: {diagnosis}")
                lines.append(
                    f"\nAction: read_file the raw log -> understand what went wrong -> "
                    f"tell Cai the failure reason and suggest next step (retry / different approach / Cai decides)"
                )
            else:
                lines.append(
                    f"\nAction: read_file the raw log -> extract 3-5 key findings "
                    f"(files changed, why, decisions, issues) -> summarize to Cai in Chinese"
                )
            blocks.append("\n".join(line for line in lines if line))
        return "\n\n---\n\n".join(blocks)

    def _build_mode_prompt(self, session) -> str:
        """Build mode-specific system prompt (零号, Jarvis, or Kevin).

        For Zero mode, also loads USER.md, AGENTS.md, and the active task file
        so they are part of the context (these are skipped when override_system_prompt
        bypasses ContextBuilder.build_system_prompt).
        """
        mode = self._get_mode(session)
        memory_context = self.context.memory.get_memory_context()
        if mode == "jarvis":
            status_context = self.dual_ledger.status_summary() if self.dual_ledger else ""
            return jarvis_system_prompt(str(self.workspace), status_context)

        if mode == "kevin":
            # Settle daily rent — accrues by calendar day, sleeping doesn't save money
            if self._kevin_state and self.config_kevin.daily_cost > 0:
                self._kevin_state.deduct_daily_rent(self.config_kevin.daily_cost)
            state_summary = self._kevin_state.get_status_summary() if self._kevin_state else ""
            review_context = self._kevin_state.format_review_context() if self._kevin_state else ""
            memo = self._kevin_state.get_memo() if self._kevin_state else ""
            turn_count = self._kevin_state.get_turn_count() if self._kevin_state else 0
            is_sim = self._kevin_state.is_simulation() if self._kevin_state else False
            return kevin_system_prompt(
                str(self.workspace), state_summary, review_context, memo,
                turn_count=turn_count, is_simulation=is_sim,
            )

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

        # Auto-inject active project file (personal interests / idle work)
        project_path = self.context.memory.get_active_project_path()
        if project_path:
            try:
                project_content = project_path.read_text(encoding="utf-8").strip()
                if project_content:
                    rel = project_path.relative_to(self.workspace)
                    extras.append(f"## Active Project ({rel})\n{project_content}")
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

    async def _llm_confirm_hallucination(
        self, content: str, tools_used: list[str]
    ) -> bool:
        """Use a fast LLM call to confirm whether the response hallucinates an action.

        Only called after _needs_hallucination_check() passes (keyword pre-filter).
        Cost: ~200 input tokens, ~5 output tokens. Adds ~0.5-1s latency.

        Returns True if the response claims to have dispatched/completed an action
        that doesn't appear in tools_used.
        """
        tools_str = ", ".join(tools_used) if tools_used else "none"
        prompt = (
            "判断下面这段回复是否声称已经执行了某个操作（派了 worker、创建了任务、"
            "完成了某件事），但实际调用的工具列表里并没有对应的操作。\n\n"
            f"实际调用的工具: [{tools_str}]\n\n"
            f"回复内容:\n{content[:600]}\n\n"
            "这段回复是否声称做了实际没做的事？只回答 YES 或 NO。"
        )
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0,
                max_tokens=10,
            )
            answer = (response.content or "").strip().upper()
            is_hallucination = "YES" in answer
            if is_hallucination:
                logger.info(f"LLM hallucination check: YES (tools={tools_str})")
            return is_hallucination
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return False  # Fail open — don't block on check failure

    async def _run_agent_loop(
        self, initial_messages: list[dict], *, mode: str | None = None,
    ) -> tuple[str | None, list[str], str | None]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.
            mode: Session mode (e.g. "kevin") — controls which tools are visible.

        Returns:
            Tuple of (final_content, list_of_tools_used, reasoning_content).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        last_reasoning: str | None = None
        tool_defs = self.tools.get_definitions(mode=mode)

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=tool_defs,
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
                        tools=tool_defs,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=bumped,
                    )
                    if retry.reasoning_content:
                        last_reasoning = retry.reasoning_content
                    if retry.finish_reason != "length" and retry.content:
                        response = retry

                final_content = response.content

                # ── Action-hallucination guard ────────────────
                # If the model *claims* it dispatched a worker but
                # didn't actually call the tool, inject a correction
                # and let it try again (once).
                if (
                    final_content
                    and iteration < self.max_iterations
                    and _needs_hallucination_check(final_content, tools_used)
                    and await self._llm_confirm_hallucination(final_content, tools_used)
                ):
                    logger.warning("Action hallucination detected — re-prompting")
                    messages = self.context.add_assistant_message(
                        messages, final_content, [],
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "[System] 你刚才说派了 worker 或完成了某个操作，"
                            "但实际上并没有调用对应的工具。请用工具实际执行，"
                            "或者诚实地说你还没做。不要编造已完成的操作。"
                        ),
                    })
                    final_content = None
                    continue

                break

        return final_content, tools_used, last_reasoning

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True

        # ── Boot recovery: auto-rollback plugin edits that caused a crash ──
        self._check_plugin_crash_rollback()

        # ── Boot recovery: mark orphaned 'doing' tasks as done ────
        # On restart, conversational workers (in-memory) are lost.
        # Tasks stuck at 'doing' or 'waiting_for_input' would cause
        # Zero to re-dispatch them via startup_greet. Mark them done
        # so the LLM sees they were already handled.
        if self.dual_ledger:
            orphan_statuses = ("doing", "waiting_for_input")
            for task in self.dual_ledger.list_tasks():
                if task.status in orphan_statuses:
                    logger.info(
                        f"Boot recovery: orphaned task {task.task_id} "
                        f"(was {task.status}). Marking as done."
                    )
                    self.dual_ledger.update_task_status(task.task_id, "done")

        # ── Conversation follow-up state ──────────────────────────
        # Tracks which chats have been active this session
        self._active_chats: set[tuple[str, str]] = set()
        # Tracks when each chat last received a user message (monotonic time)
        self._last_user_msg_at: dict[tuple[str, str], float] = {}
        # Tracks when to next *consider* a follow-up per chat (monotonic time)
        self._next_followup_at: dict[tuple[str, str], float] = {}
        # How many follow-ups sent without a user reply (per chat)
        self._unanswered_followups: dict[tuple[str, str], int] = {}
        # What was said in each unanswered follow-up (for dedup / progression)
        self._followup_history: dict[tuple[str, str], list[str]] = {}

        # ── Idle activity state ───────────────────────────────────
        self._last_idle_activity_at: float = 0.0
        self._next_idle_check_at: float = time.monotonic() + 1800  # first check after 30 min
        self._idle_active: bool = False

        logger.info("Agent loop started")

        # ── Start MCP servers ──────────────────────────────────────
        if self._mcp_manager:
            try:
                await self._mcp_manager.start()
            except Exception as e:
                logger.error(f"MCP Manager start failed (non-fatal): {e}")

        # ── Restore active chats from session history & send startup greeting ──
        await self._startup_greet()

        # ── Kevin boot (autonomous trader) ──
        await self._kevin_boot()

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
                # User replied → reset follow-up progression
                self._unanswered_followups[chat_key] = 0
                self._followup_history[chat_key] = []
                # Clear persisted state (will be saved in _process_message)
                _s = self.sessions.get_or_create(f"{msg.channel}:{msg.chat_id}")
                _s.metadata.pop("followup_history", None)
                _s.metadata.pop("unanswered_followups", None)

                try:
                    response = await self._process_message(msg)
                    if response and response.metadata.get("_silent"):
                        # [SILENT] — forward directly so channel can react
                        await self.bus.publish_outbound(response)
                    elif response:
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
                        # Fallback — cancel typing indicator
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
                await self._maybe_idle_activity()
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

            last_ts = session.messages[-1].get("timestamp", "") if session.messages else ""
            nudge = startup_greeting_context(last_message_ts=last_ts)

            # Route through the unified _process_message pipeline (full tools).
            msg = InboundMessage(
                channel=channel, sender_id="system", chat_id=chat_id,
                content=nudge,
                metadata={"proactive": True, "trigger": "startup_greeting"},
            )
            try:
                response = await self._process_message(msg, session_key=session_key)
            except Exception as e:
                logger.warning(f"Startup greeting failed for {session_key}: {e}")
                continue

            if not response or response.metadata.get("_silent"):
                logger.debug(f"Startup greeting skipped for {session_key} (LLM decided)")
                continue

            logger.info(f"Startup greeting to {session_key}: {response.content[:80]}")

            parts = [p.strip() for p in re.split(r"\n---\n", response.content) if p.strip()]
            for i, part in enumerate(parts):
                await self.bus.publish_outbound(OutboundMessage(
                    channel=channel, chat_id=chat_id, content=part,
                ))
                if i < len(parts) - 1:
                    await asyncio.sleep(0.8)

    # ── Kevin boot (autonomous trader) ─────────────────────────

    async def _kevin_boot(self) -> None:
        """Boot Kevin if enabled. Sets up his session and triggers first wakeup.

        Kevin schedules his own subsequent wakeups via the cron tool.
        """
        if not self.config_kevin.enabled or not self._kevin_client:
            return

        # Sync balance from Bybit before death check (fixes cold-start where
        # portfolio.json doesn't exist yet → balance=0 → instant death)
        if self._kevin_state:
            try:
                live_balance = self._kevin_client.get_balance("USDT")
                init = self.config_kevin.initial_balance or None
                self._kevin_state.update_portfolio(live_balance, initial_balance=init)
                logger.info(f"Kevin boot: synced balance from Bybit → {live_balance:.2f} USDT")
            except Exception as e:
                logger.warning(f"Kevin boot: failed to sync Bybit balance: {e}")

        # Check if Kevin is dead (no funds)
        if self._kevin_state and self._kevin_state.is_dead():
            logger.warning("Kevin boot: no funds (balance ≤ 0.01 USDC). Not starting.")
            return

        logger.info("Kevin boot: starting autonomous trader")

        # Create/restore Kevin's session with mode locked to "kevin"
        session = self.sessions.get_or_create("cron:kevin")
        if session.metadata.get("mode") != "kevin":
            session.metadata["mode"] = "kevin"
            self.sessions.save(session)

        # Trigger first wakeup
        try:
            await self.process_direct(
                kevin_wakeup_prompt(),
                session_key="cron:kevin",
                channel="system",
                chat_id="kevin",
            )
            logger.info("Kevin boot: first wakeup completed")
        except Exception as e:
            logger.error(f"Kevin boot failed: {e}")

        # Safety net: if Kevin's turn didn't set a cron, create a default one
        self._ensure_kevin_cron()

    def _ensure_kevin_cron(self) -> None:
        """Safety net: if no Kevin cron job exists, create a 15-minute default.

        Called after _kevin_boot() and could be called after cron-triggered
        wakeups to guarantee Kevin always has a next wakeup scheduled.
        """
        if not self.cron_service:
            return
        kevin_jobs = [
            j for j in self.cron_service.list_jobs()
            if j.payload.to == "kevin" and j.enabled
        ]
        if kevin_jobs:
            return  # Already has a cron — all good

        from nanobot.cron.types import CronSchedule
        self.cron_service.add_job(
            name="Kevin auto-wakeup (15m)",
            schedule=CronSchedule(kind="every", every_ms=15 * 60 * 1000),
            message="[Auto-wakeup: end_turn was not called. Check what happened last time.]",
            deliver=True,
            channel="system",
            to="kevin",
            delete_after_run=False,
        )
        logger.warning("Kevin: no cron job found after boot — created 15min safety-net wakeup")

    # ── Proactive conversation follow-up ─────────────────────────

    async def _maybe_conversation_followup(self) -> None:
        """Check if Zero should proactively follow up on any silent conversation.

        The system provides context (what was already said, how many times),
        and the LLM decides whether to speak, what to say, or [SKIP].
        No hardcoded caps — Zero decides organically based on her personality.

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

            session_key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(session_key)

            # Skip if the session is empty (no prior conversation to follow up on)
            if not session.messages:
                self._next_followup_at[chat_key] = now + random.uniform(600, 1800)
                continue

            # Include context for the LLM to make an informed decision
            unanswered = self._unanswered_followups.get(chat_key, 0)
            prev_msgs = self._followup_history.get(chat_key, [])
            # Restore from session metadata if in-memory is empty (e.g. after restart)
            if not prev_msgs and session.metadata.get("followup_history"):
                prev_msgs = session.metadata["followup_history"]
                self._followup_history[chat_key] = prev_msgs
                unanswered = session.metadata.get("unanswered_followups", 0)
                self._unanswered_followups[chat_key] = unanswered
            task_summary = self.dual_ledger.status_summary() if self.dual_ledger else ""
            nudge = followup_nudge_context(
                silent_minutes,
                task_summary=task_summary,
                unanswered_count=unanswered,
                previous_followups=prev_msgs,
            )

            # Route through the unified _process_message pipeline (full tools).
            msg = InboundMessage(
                channel=channel, sender_id="system", chat_id=chat_id,
                content=nudge,
                metadata={"proactive": True, "trigger": "followup",
                          "silent_minutes": silent_minutes},
            )
            try:
                response = await self._process_message(msg, session_key=session_key)
            except Exception as e:
                logger.warning(f"Follow-up LLM call failed for {session_key}: {e}")
                self._next_followup_at[chat_key] = now + random.uniform(600, 1800)
                continue

            if not response or response.metadata.get("_silent"):
                # LLM decided not to say anything — cooldown increases with unanswered count
                base_cooldown = 600 + unanswered * 300  # more silence → longer wait
                cooldown = random.uniform(base_cooldown, base_cooldown * 2)
                self._next_followup_at[chat_key] = now + cooldown
                logger.debug(f"Follow-up skipped for {session_key} (silent {silent_minutes}m, unanswered={unanswered})")
                continue

            # Send the follow-up
            logger.info(f"Follow-up to {session_key} (silent {silent_minutes}m, unanswered={unanswered}): {response.content[:80]}")

            # Track this follow-up for progression context
            self._unanswered_followups[chat_key] = unanswered + 1
            if chat_key not in self._followup_history:
                self._followup_history[chat_key] = []
            self._followup_history[chat_key].append(response.content[:200])
            # Persist so it survives restart
            session.metadata["followup_history"] = self._followup_history[chat_key]
            session.metadata["unanswered_followups"] = self._unanswered_followups[chat_key]
            self.sessions.save(session)

            # Split on --- like normal messages (multi-bubble)
            parts = [p.strip() for p in re.split(r"\n---\n", response.content) if p.strip()]
            for i, part in enumerate(parts):
                await self.bus.publish_outbound(OutboundMessage(
                    channel=channel, chat_id=chat_id, content=part,
                ))
                if i < len(parts) - 1:
                    await asyncio.sleep(0.8)

            # Cooldown increases with unanswered count (naturally space out)
            base = 1200 + unanswered * 600  # 20min base, +10min per unanswered
            self._next_followup_at[chat_key] = now + random.uniform(base, base * 1.5)

    # ── Idle activity ─────────────────────────────────────────────

    async def _maybe_idle_activity(self) -> None:
        """Run an idle activity session if all conditions are met.

        Conditions (all must hold):
          1. Past the next scheduled check time
          2. All chats silent > 15 min
          3. No running tasks in dual_ledger
          4. Cooldown: > 45 min since last idle activity
          5. Not already running (_idle_active guard)
        """
        now = time.monotonic()

        if self._idle_active:
            return
        if now < self._next_idle_check_at:
            return

        # Need at least 15 min silence
        if self._last_user_msg_at:
            most_recent = max(self._last_user_msg_at.values())
            if now - most_recent < 900:  # 15 min
                self._next_idle_check_at = now + 60
                return

        # No running tasks
        if self.dual_ledger:
            summary = self.dual_ledger.status_summary()
            if "running" in summary.lower():
                self._next_idle_check_at = now + 300  # re-check in 5 min
                return

        # Cooldown since last idle activity (45 min)
        if self._last_idle_activity_at and now - self._last_idle_activity_at < 2700:
            return

        self._idle_active = True
        try:
            has_project = self.context.memory.get_active_project_path() is not None

            # Calculate how long Cai has been silent
            if self._last_user_msg_at:
                most_recent = max(self._last_user_msg_at.values())
                silent_min = int((now - most_recent) / 60)
            else:
                silent_min = 30  # default if no messages yet

            prompt = idle_activity_prompt(
                has_active_project=has_project,
                silent_minutes=silent_min,
            )
            logger.info(f"Idle activity triggered (silent={silent_min}m, project={has_project})")

            result = await self.process_direct(
                prompt,
                session_key="idle:personal",
                channel="system",
                chat_id="idle",
            )

            if result and "[SKIP]" in result:
                # Short cooldown — didn't want to do anything
                cooldown = random.uniform(1200, 2400)  # 20–40 min
                logger.info("Idle activity: Zero chose to skip")
            else:
                # Long cooldown — actually did something
                cooldown = random.uniform(2700, 5400)  # 45–90 min
                logger.info(f"Idle activity done: {(result or '')[:100]}")

                # ── Narrator: tell Cai what Zero was up to ──
                await self._send_narrator_summary(result or "")

        except Exception:
            logger.warning("Idle activity failed", exc_info=True)
            cooldown = random.uniform(1800, 3600)  # 30–60 min on error
        finally:
            self._idle_active = False
            self._last_idle_activity_at = time.monotonic()
            self._next_idle_check_at = time.monotonic() + cooldown

    async def _send_narrator_summary(self, idle_result: str) -> None:
        """Generate a brief narrator line about Zero's idle activity and send to active chats."""
        if not idle_result.strip():
            return
        try:
            sys_prompt, user_prompt = narrator_summary_prompt(idle_result)
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.9,
                max_tokens=200,
            )
            narrator_text = (response.content or "").strip()
            if not narrator_text:
                return

            # Send to all active chats as a quiet narrator message
            for channel, chat_id in list(self._active_chats):
                if self._enabled_channels is not None and channel not in self._enabled_channels:
                    continue
                await self.bus.publish_outbound(OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=f"_{narrator_text}_",  # Markdown italic
                    metadata={"narrator": True},
                ))
            logger.info(f"Narrator: {narrator_text[:80]}")
        except Exception:
            logger.debug("Narrator summary failed", exc_info=True)

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
            return await self._process_system_message(msg, session_key=session_key)

        is_proactive = msg.metadata.get("proactive", False)
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        # Archive inbound message (skip for proactive triggers — they're internal nudges)
        if not is_proactive:
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

        if len(session.messages) > self.memory_window and session.key not in self._consolidating:
            self._consolidating.add(session.key)
            asyncio.create_task(self._consolidate_memory_guarded(session))

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
        session_mode = self._get_mode(session)
        final_content, tools_used, reasoning = await self._run_agent_loop(
            initial_messages, mode=session_mode if session_mode != "zero" else None,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Strip timestamp prefixes the LLM may have mimicked from history
        final_content = _strip_ts_prefix(final_content)
        # Unwrap message("...") wrappers from thinking models
        final_content = _strip_tool_text_wrapper(final_content)

        # [SILENT] — Zero chose not to reply (e.g. user hasn't finished typing).
        # [SKIP]  — Zero decided not to say anything (proactive triggers only).
        # Record in session history so she has context, but send nothing.
        is_silent = "[SILENT]" in final_content or (is_proactive and "[SKIP]" in final_content)

        # For proactive triggers (greeting / follow-up), the inbound content is a
        # system nudge, not a real user message — don't pollute session history.
        if not is_proactive:
            session.add_message("user", msg.content,
                                media=msg.media if msg.media else None)
        if not is_silent:
            session.add_message("assistant", final_content,
                                tools_used=tools_used if tools_used else None,
                                reasoning_content=reasoning)
        self.sessions.save(session)

        if is_silent:
            logger.info(f"[SILENT] for {msg.channel}:{msg.sender_id} — no outbound")
            _archive_message("out", msg.channel, msg.chat_id, "[SILENT]")
            # Return a silent marker so the channel can acknowledge (e.g. reaction)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="",
                metadata={**(msg.metadata or {}), "_silent": True},
            )

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

    async def _process_system_message(
        self, msg: InboundMessage, session_key: str | None = None,
    ) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce, cron wakeup).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.

        Args:
            msg: The inbound system message.
            session_key: Override session key (e.g. "cron:kevin" from process_direct).
                         When provided, the session is looked up by this key so that
                         Kevin's cron wakeups land in his dedicated session with the
                         correct mode prompt — not in a freshly-derived "cli:kevin".
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

        key = session_key or f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(key)
        self._set_tool_context(origin_channel, origin_chat_id)

        # Use mode prompt so Zero has full context (AGENTS.md, MEMORY.md, etc.)
        system_prompt = self._build_mode_prompt(session)
        notification_context = self._check_notifications()
        if notification_context:
            system_prompt += f"\n\n## Recent Task Completions\n{notification_context}"

        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
            override_system_prompt=system_prompt,
        )
        session_mode = self._get_mode(session)
        final_content, _, _ = await self._run_agent_loop(
            initial_messages, mode=session_mode if session_mode != "zero" else None,
        )

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

    async def _consolidate_memory_guarded(self, session) -> None:
        """Wrapper that ensures only one consolidation runs per session at a time."""
        try:
            await self._consolidate_memory(session)
            self.sessions.save(session)  # persist updated last_consolidated
        finally:
            self._consolidating.discard(session.key)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Append a summary to HISTORY.md for grep-searchable event logging.

        MEMORY.md is NOT touched here — Zero maintains it herself
        (during diary writing or whenever she decides to).

        Args:
            archive_all: If True, process all messages (for /new, /archive).
                       If False, only process new messages since last consolidation.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"History consolidation (archive_all): {len(session.messages)} messages")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"History consolidation: {len(old_messages)} new messages to log")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)

        prompt = f"""刚聊完一段，记一笔。

格式: YYYY-MM-DD HH:MM | 内容
2-3 句话，关键事件、决策、结果。留具体细节（文件名、任务 ID、技术词）。

## 对话内容
{conversation}"""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "我是零号。这是我的事件日志，写给以后的自己翻。简短，带细节，grep 能搜到就好。"},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if text:
                memory.append_history(text)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"History consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"History consolidation failed: {e}")

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
