"""Subagent manager for background task execution.

Two modes:
- Simple subagent: fire-and-forget (spawn)
- Conversational worker: can pause for input, write reports (spawn_worker)
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


# ── Reports directory ────────────────────────────────────────────

REPORTS_DIR = Path.home() / ".nanobot" / "data" / "reports"


# ── Worker state ─────────────────────────────────────────────────

@dataclass
class WorkerState:
    """Mutable state for a conversational worker."""

    worker_id: str          # Same as run_id (W-xxx)
    task_id: str            # T-xxx
    task_goal: str          # Full task description
    status: str = "running"  # running | waiting_for_input | done | failed
    question: str | None = None
    question_context: str | None = None
    input_answer: str | None = None
    input_event: asyncio.Event = field(default_factory=asyncio.Event)
    messages: list[dict] = field(default_factory=list)
    report_path: str = ""
    origin: dict[str, str] = field(default_factory=dict)

    # Injected callback: (worker_id, task_id, question, context) -> None
    notify_callback: Callable[..., Awaitable[None]] | None = field(
        default=None, repr=False
    )


class SubagentManager:
    """
    Manages background subagent and conversational worker execution.

    - Simple subagents (spawn): lightweight, fire-and-forget
    - Conversational workers (spawn_worker): can pause for input,
      generate reports, integrate with DualLedger
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        google_cse_api_key: str | None = None,
        google_cse_cx: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.google_cse_api_key = google_cse_api_key
        self.google_cse_cx = google_cse_cx
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

        # Conversational workers (keyed by worker_id / run_id)
        self._workers: dict[str, WorkerState] = {}

        # Optional DualLedger (set from AgentLoop after init)
        self._dual_ledger: Any | None = None

    def set_dual_ledger(self, dl: Any) -> None:
        """Wire up DualLedger reference (called from AgentLoop)."""
        self._dual_ledger = dl

    # ══════════════════════════════════════════════════════════════
    #  Simple subagent (existing, unchanged)
    # ══════════════════════════════════════════════════════════════

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """Spawn a simple fire-and-forget subagent."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute a simple subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        try:
            tools = self._build_base_tools()
            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            final_result = await self._run_agent_loop(
                task_id, tools, messages, max_iterations=15
            )

            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    # ══════════════════════════════════════════════════════════════
    #  Conversational worker (new)
    # ══════════════════════════════════════════════════════════════

    async def spawn_worker(
        self,
        task: str,
        task_id: str,
        run_id: str,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        model: str | None = None,
    ) -> str:
        """Spawn a conversational worker that can ask questions.

        Args:
            task: Detailed task description.
            task_id: DualLedger task ID (T-xxx).
            run_id: DualLedger run ID (W-xxx).
            origin_channel: Channel to send notifications to.
            origin_chat_id: Chat ID for notifications.
            model: Override model for this worker.

        Returns:
            Status message.
        """
        report_path = str(REPORTS_DIR / f"{task_id}-report.md")

        state = WorkerState(
            worker_id=run_id,
            task_id=task_id,
            task_goal=task,
            report_path=report_path,
            notify_callback=self._notify_zero_question,
            origin={"channel": origin_channel, "chat_id": origin_chat_id},
        )
        self._workers[run_id] = state

        worker_model = model or self.model
        # Bare alias without provider prefix → not resolvable by litellm
        if "/" not in worker_model:
            logger.warning(
                f"Worker model '{worker_model}' has no provider prefix, "
                f"falling back to parent model: {self.model}"
            )
            worker_model = self.model

        bg_task = asyncio.create_task(
            self._run_conversational_worker(state, task, worker_model)
        )
        self._running_tasks[run_id] = bg_task
        bg_task.add_done_callback(lambda _: self._cleanup_worker(run_id))

        logger.info(f"Spawned conversational worker [{run_id}] for {task_id}: {task[:80]}")
        return f"Worker {run_id} started for {task_id}."

    def reply_to_worker(self, worker_id: str, answer: str) -> str:
        """Send an answer to a waiting worker, unblocking its request_input."""
        state = self._workers.get(worker_id)
        if not state:
            # Try matching by task_id as a convenience
            for ws in self._workers.values():
                if ws.task_id == worker_id:
                    state = ws
                    break
        if not state:
            return f"Worker {worker_id} not found. Active workers: {list(self._workers.keys()) or 'none'}"
        if state.status != "waiting_for_input":
            return f"Worker {worker_id} is not waiting for input (status: {state.status})."

        state.input_answer = answer
        state.input_event.set()
        logger.info(f"Reply sent to worker {worker_id}: {answer[:80]}")
        return f"Answer delivered to {worker_id}. Worker will resume."

    def get_worker(self, worker_id: str) -> WorkerState | None:
        """Get a worker's state by ID."""
        return self._workers.get(worker_id)

    def list_workers(self) -> list[dict[str, str]]:
        """List all active workers with their status."""
        result = []
        for wid, state in self._workers.items():
            entry = {
                "worker_id": wid,
                "task_id": state.task_id,
                "status": state.status,
                "goal": state.task_goal[:60],
            }
            if state.question:
                entry["question"] = state.question[:100]
            result.append(entry)
        return result

    async def _run_conversational_worker(
        self,
        state: WorkerState,
        task: str,
        model: str,
    ) -> None:
        """Execute a conversational worker with request_input capability."""
        wid = state.worker_id
        tid = state.task_id
        logger.info(f"Worker [{wid}] starting: {task[:80]}")

        # Update DualLedger status
        if self._dual_ledger:
            self._dual_ledger.update_task_status(tid, "doing")

        try:
            # Build tools (base + request_input)
            from nanobot.agent.tools.worker_input import RequestInputTool
            tools = self._build_base_tools()
            tools.register(RequestInputTool(state))

            # Build messages
            system_prompt = self._build_worker_prompt(state)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]
            state.messages = messages

            # Run agent loop (more iterations than simple subagent)
            final_result = await self._run_agent_loop(
                wid, tools, messages, max_iterations=50, model=model
            )

            state.status = "done"
            logger.info(f"Worker [{wid}] completed successfully")

            # Write report
            report = self._generate_report(state, final_result)
            self._save_report(state, report)

            # Update DualLedger
            self._update_ledger_on_completion(state, final_result, "success")

            # Announce to Zero
            await self._announce_worker_completion(state, final_result, "ok")

        except Exception as e:
            state.status = "failed"
            error_msg = f"Error: {str(e)}"
            logger.error(f"Worker [{wid}] failed: {e}")

            self._update_ledger_on_completion(state, error_msg, "failed")
            await self._announce_worker_completion(state, error_msg, "error")

    def _cleanup_worker(self, run_id: str) -> None:
        """Remove worker from tracking after completion."""
        self._running_tasks.pop(run_id, None)
        # Keep in _workers for a while so Zero can still query it
        state = self._workers.get(run_id)
        if state and state.status not in ("done", "failed"):
            state.status = "failed"

    # ── Notification helpers ─────────────────────────────────────

    async def _notify_zero_question(
        self,
        worker_id: str,
        task_id: str,
        question: str,
        context: str,
    ) -> None:
        """Send a worker's question to Zero via MessageBus."""
        # Update DualLedger status
        if self._dual_ledger:
            self._dual_ledger.update_task_status(task_id, "waiting_for_input")

        content_parts = [
            f"[Worker {worker_id} ({task_id}) needs input]",
            "",
            f"**Question:** {question}",
        ]
        if context:
            content_parts.append(f"\n**Context:** {context}")
        content_parts.append(
            f"\nUse `worker_reply(worker_id='{worker_id}', answer='...')` to reply."
        )

        state = self._workers.get(worker_id)
        origin = state.origin if state else {}
        chat_id = f"{origin.get('channel', 'cli')}:{origin.get('chat_id', 'direct')}"

        msg = InboundMessage(
            channel="system",
            sender_id="worker",
            chat_id=chat_id,
            content="\n".join(content_parts),
        )
        await self.bus.publish_inbound(msg)

    async def _announce_worker_completion(
        self,
        state: WorkerState,
        result: str,
        status: str,
    ) -> None:
        """Announce worker completion to Zero via MessageBus."""
        status_text = "completed" if status == "ok" else "failed"

        content_parts = [
            f"[Worker {state.worker_id} ({state.task_id}) {status_text}]",
            "",
            f"**Task:** {state.task_goal[:120]}",
            f"**Result:** {result[:800]}",
        ]

        if state.report_path and Path(state.report_path).exists():
            content_parts.append(f"\n**Full report:** `{state.report_path}`")

        # Action guidance for Zero
        if status == "ok":
            content_parts.append(
                "\n**Action required:** Read the full report with `read_file`, "
                "then summarize key findings to Cai. Include file paths and evidence."
            )
        else:
            content_parts.append(
                "\n**Action required:** Read the report to understand the failure. Then either:\n"
                "1. Fix the issue and `dispatch_worker` again with more context\n"
                "2. Do the work yourself if it's quick\n"
                "3. Ask Cai for guidance if you're unsure"
            )

        chat_id = f"{state.origin.get('channel', 'cli')}:{state.origin.get('chat_id', 'direct')}"

        msg = InboundMessage(
            channel="system",
            sender_id="worker",
            chat_id=chat_id,
            content="\n".join(content_parts),
        )
        await self.bus.publish_inbound(msg)

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce simple subagent result (unchanged from original)."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )
        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")

    # ── DualLedger integration ───────────────────────────────────

    def _update_ledger_on_completion(
        self,
        state: WorkerState,
        result: str,
        run_status: str,
    ) -> None:
        """Update DualLedger with worker completion."""
        dl = self._dual_ledger
        if not dl:
            return

        # Update run record
        run = dl.get_run(state.worker_id)
        if run:
            run.status = run_status
            run.summary = result[:200]
            run.report_path = state.report_path if hasattr(run, "report_path") else ""
            dl.save_run(run)

        # Update task status
        task_status = "done" if run_status == "success" else "blocked"
        dl.update_task_status(state.task_id, task_status)

        # Append notification (for backup — primary notification is via MessageBus)
        dl.append_notification(
            task_id=state.task_id,
            run_id=state.worker_id,
            status=run_status,
            summary=result[:200],
        )

    # ── Report generation ────────────────────────────────────────

    def _generate_report(self, state: WorkerState, final_result: str) -> str:
        """Generate a detailed markdown report from the worker's conversation."""
        lines = [
            f"# Worker Report: {state.task_id} / {state.worker_id}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Task",
            "",
            state.task_goal,
            "",
            "## Execution Log",
            "",
        ]

        # Extract meaningful entries from messages
        step = 0
        for msg in state.messages:
            role = msg.get("role", "")

            if role == "assistant":
                text = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                if text and text.strip():
                    step += 1
                    lines.append(f"### Step {step}: Analysis")
                    lines.append("")
                    lines.append(text[:2000])
                    lines.append("")

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "unknown")
                    try:
                        tool_args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        tool_args = {}

                    step += 1
                    if tool_name == "request_input":
                        lines.append(f"### Step {step}: Question to Zero")
                        lines.append("")
                        lines.append(f"**Q:** {tool_args.get('question', '?')}")
                        if tool_args.get("context"):
                            lines.append(f"**Context:** {tool_args['context']}")
                    elif tool_name in ("read_file", "list_dir"):
                        target = tool_args.get("path", tool_args.get("directory", "?"))
                        lines.append(f"### Step {step}: {tool_name}(`{target}`)")
                    elif tool_name == "exec":
                        cmd = tool_args.get("command", "?")[:80]
                        lines.append(f"### Step {step}: exec(`{cmd}`)")
                    elif tool_name == "web_search":
                        query = tool_args.get("query", "?")
                        lines.append(f"### Step {step}: web_search(`{query}`)")
                    else:
                        lines.append(f"### Step {step}: {tool_name}")
                    lines.append("")

            elif role == "tool":
                tool_name = msg.get("name", "")
                content = msg.get("content", "")

                if tool_name == "request_input" and "Zero's answer:" in content:
                    answer = content.replace("Zero's answer: ", "")
                    lines.append(f"**A:** {answer[:500]}")
                    lines.append("")
                elif tool_name in ("read_file", "exec", "web_search", "web_fetch"):
                    # Truncate long tool outputs in report
                    preview = content[:800] + ("..." if len(content) > 800 else "")
                    lines.append(f"```\n{preview}\n```")
                    lines.append("")

        lines.extend([
            "## Result",
            "",
            final_result[:2000],
            "",
        ])

        return "\n".join(lines)

    def _save_report(self, state: WorkerState, report: str) -> None:
        """Write the report file to disk."""
        if not state.report_path:
            return
        path = Path(state.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(report, encoding="utf-8")
            logger.info(f"Worker report saved: {path} ({len(report)} chars)")
        except OSError as e:
            logger.warning(f"Failed to save worker report: {e}")

    # ── Shared agent loop ────────────────────────────────────────

    async def _run_agent_loop(
        self,
        label: str,
        tools: ToolRegistry,
        messages: list[dict[str, Any]],
        max_iterations: int = 15,
        model: str | None = None,
    ) -> str:
        """Run the LLM agent loop with tool calling.

        Shared between simple subagents and conversational workers.
        Returns the final text response.
        """
        use_model = model or self.model
        final_result: str | None = None

        for iteration in range(1, max_iterations + 1):
            response = await self.provider.chat(
                messages=messages,
                tools=tools.get_definitions(),
                model=use_model,
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
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": tool_call_dicts,
                })

                for tool_call in response.tool_calls:
                    logger.debug(f"[{label}] tool: {tool_call.name}")
                    result = await tools.execute(tool_call.name, tool_call.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
            else:
                final_result = response.content
                break

        return final_result or "Task completed but no final response was generated."

    # ── Tool registry builder ────────────────────────────────────

    def _build_base_tools(self) -> ToolRegistry:
        """Build the base tool set shared by subagents and workers."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(allowed_dir=allowed_dir))
        tools.register(WriteFileTool(allowed_dir=allowed_dir))
        tools.register(EditFileTool(allowed_dir=allowed_dir))
        tools.register(ListDirTool(allowed_dir=allowed_dir))
        tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        if self.brave_api_key or self.google_cse_api_key:
            tools.register(WebSearchTool(
                api_key=self.brave_api_key,
                google_cse_api_key=self.google_cse_api_key,
                google_cse_cx=self.google_cse_cx,
            ))
        tools.register(WebFetchTool())
        return tools

    # ── Prompt builders ──────────────────────────────────────────

    def _build_subagent_prompt(self, task: str) -> str:
        """Build prompt for simple subagent (unchanged)."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

When you have completed the task, provide a clear summary of your findings or actions."""

    def _build_worker_prompt(self, state: WorkerState) -> str:
        """Build prompt for conversational worker — focused on technical execution."""
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        return f"""# Worker {state.worker_id}

## Current Time
{now} ({tz})

You are a technical worker for the Insight AI project.
Your manager is Zero (零号). She assigned you task {state.task_id}.

## Your Job
1. Complete the assigned task thoroughly
2. Provide evidence: file paths, code snippets, data
3. Your final response is your summary — it goes to Zero

## Asking Questions
You have a `request_input` tool. Use it when you:
- Need clarification about requirements
- Need information you can't find in the codebase
- Need a decision from Zero or Cai (the CEO)
- Are unsure about the approach and want to confirm before proceeding

Don't guess when you can ask. Don't ask about things you can figure out by reading code.

## What You Can Do
- Read, write, edit files
- Execute shell commands (git, npm, pytest, etc.)
- Search the web
- Ask Zero questions via request_input

## Workspace
Project root: d:/Insight-AI
Read CLAUDE.md first for project context.

## Output
When done, provide a clear summary:
- What you found / what you did
- Key evidence (file paths, code references)
- Any issues or recommendations
- Files changed (if any)"""

    def get_running_count(self) -> int:
        """Return the number of currently running subagents + workers."""
        return len(self._running_tasks)
