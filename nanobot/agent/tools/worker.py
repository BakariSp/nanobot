"""Code worker dispatch tool for Boss AI orchestrator.

Two modes:
- Legacy: Dispatches to Claude Code CLI synchronously (for backward compat)
- New:    Non-blocking — writes T-xxx task to DualLedger queue, returns immediately.
          Ralph Loop picks up and executes asynchronously.
"""

import asyncio
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.task_ledger import (
    TaskLedger, TaskRecord,
    DualLedger, TaskDefinition, classify_risk,
)

# Legacy usage log (kept for backward compat)
USAGE_LOG = Path.home() / ".nanobot" / "data" / "worker_usage.jsonl"

# Interval for periodic progress messages (seconds)
PROGRESS_INTERVAL = 120


def _parse_report(output: str) -> dict:
    """Extract JSON report from worker output."""
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', output)

    # Find last balanced JSON with "status" key
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] == '}':
            depth = 0
            for j in range(i, -1, -1):
                if cleaned[j] == '}':
                    depth += 1
                elif cleaned[j] == '{':
                    depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[j:i + 1])
                        if isinstance(parsed, dict) and "status" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        break
                    break

    # Fallback
    tail = output[-800:] if len(output) > 800 else output
    return {
        "status": "unknown",
        "summary": "Could not parse worker report",
        "raw_output_tail": tail,
        "files_changed": [],
        "risk_signals": ["report_parse_failed"],
        "checks": {}
    }


_AUTH_KEYWORDS = re.compile(
    r"(auth\w*\s+(required|expired|failed|error)|"
    r"log\s*in\s+(required|to continue)|"
    r"unauthorized|"
    r"not\s+authenticated|"
    r"session\s+expired|"
    r"oauth\s+(token|expired|error)|"
    r"re-?auth)",
    re.IGNORECASE,
)

_AUTH_URL_PATTERN = re.compile(r"(https?://\S*(?:auth|login|oauth|consent)\S*)", re.IGNORECASE)


def _detect_auth_failure(output: str) -> str | None:
    """Check worker output for Claude Code auth expiry.

    Returns a user-friendly message if auth failure detected, None otherwise.
    """
    if not _AUTH_KEYWORDS.search(output):
        return None

    # Try to extract the auth URL so user can click it directly
    url_match = _AUTH_URL_PATTERN.search(output)
    url_hint = f"\n\nAuth URL: {url_match.group(1)}" if url_match else ""

    return (
        "Claude Code auth expired. "
        "Please open a terminal and run `claude` to re-authorize, "
        "then retry the task." + url_hint
    )


def _log_usage(task: str, model: str, duration_s: float, status: str, exit_code: int) -> None:
    """Append usage record to JSONL log for quota tracking (legacy)."""
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "task_preview": task[:120],
        "duration_s": round(duration_s, 1),
        "status": status,
        "exit_code": exit_code,
    }
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class WorkerDispatchTool(Tool):
    """Dispatch a coding task — non-blocking queue mode (default) or legacy sync.

    New mode (queue):
        Writes a T-xxx TaskDefinition to DualLedger with risk classification.
        Returns immediately. Ralph Loop picks up and executes asynchronously.

    Legacy mode:
        Runs Claude Code CLI synchronously. Used when dual_ledger is None.
    """

    def __init__(
        self,
        default_cwd: str = ".",
        ledger: TaskLedger | None = None,
        dual_ledger: DualLedger | None = None,
        notion_tool: Any | None = None,
    ):
        self.default_cwd = default_cwd
        self._ledger = ledger
        self._dual_ledger = dual_ledger
        self._notion_tool = notion_tool
        self._progress_callback: Callable[[str, dict], Awaitable[None]] | None = None

    def set_notion_tool(self, tool: Any) -> None:
        """Wire up the Notion tool after registration."""
        self._notion_tool = tool

    def set_progress_callback(
        self, cb: Callable[[str, dict], Awaitable[None]] | None
    ) -> None:
        """Inject a callback for sending periodic progress messages."""
        self._progress_callback = cb

    @property
    def name(self) -> str:
        return "dispatch_worker"

    @property
    def description(self) -> str:
        return (
            "Queue a coding task for async execution by the Ralph Loop. "
            "Returns immediately with task ID, risk level, and queue status. "
            "L1-L2 tasks auto-execute. L3 tasks need /approve. "
            "Use for code changes, scans, fixes, features."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Detailed task description for the worker"
                },
                "task_type": {
                    "type": "string",
                    "enum": ["recon", "fix", "feature", "refactor", "plan", "verify"],
                    "description": "Task type. Default: fix"
                },
                "scope_allow": {
                    "type": "string",
                    "description": "Allowed file paths, comma-separated (e.g. 'insigh_ai_frontend/**')"
                },
                "scope_deny": {
                    "type": "string",
                    "description": "Denied file paths, comma-separated (e.g. 'Insight_Backend/**')"
                },
                "model": {
                    "type": "string",
                    "description": "Worker model. Default: opus"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory. Default: d:/Insight-AI"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds for worker. Default: 3600"
                },
                "acceptance": {
                    "type": "string",
                    "description": "Acceptance check command (e.g. 'npm run build'). Optional."
                },
            },
            "required": ["task"]
        }

    # ── Progress helpers ──────────────────────────────────────────

    async def _run_with_progress(
        self,
        proc: asyncio.subprocess.Process,
        timeout: int,
        task_id: str,
    ) -> tuple[bytes, bytes]:
        """Run proc.communicate() while sending periodic progress messages."""
        start = time.monotonic()

        async def _tick() -> None:
            while True:
                await asyncio.sleep(PROGRESS_INTERVAL)
                elapsed = int(time.monotonic() - start)
                if self._progress_callback:
                    try:
                        await self._progress_callback("dispatch_worker", {
                            "_progress": True,
                            "task_id": task_id,
                            "message": f"Worker {task_id} still running ({elapsed}s elapsed)…",
                        })
                    except Exception:
                        pass

        tick_task = asyncio.create_task(_tick())
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return stdout, stderr
        finally:
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass

    # ── Notion sync ───────────────────────────────────────────────

    async def _sync_to_notion(self, record: TaskRecord) -> str | None:
        """Save report to Notion; returns page URL or None."""
        if not self._notion_tool:
            return None
        try:
            md = TaskLedger.render_markdown(record)
            result = await self._notion_tool.execute(
                title=f"Worker Report {record.task_id}",
                content=md,
                tags=["worker-report", record.status],
            )
            # The notion tool returns "Saved to Notion: <url> (page_id: ...)"
            if "notion.so" in result or "notion.site" in result:
                for word in result.split():
                    if "notion" in word and ("http" in word or "//" in word):
                        return word.strip("()")
            return result  # Return raw result if we can't extract URL
        except Exception as e:
            logger.warning(f"Notion sync failed for {record.task_id}: {e}")
            return None

    # ── Main execute ──────────────────────────────────────────────

    async def execute(
        self,
        task: str,
        task_type: str = "fix",
        scope_allow: str = "",
        scope_deny: str = "",
        model: str = "opus",
        cwd: str = "",
        timeout: int = 3600,
        acceptance: str = "",
        **kwargs: Any,
    ) -> str:
        # ── New: non-blocking queue mode ──────────────────────────
        if self._dual_ledger:
            return self._queue_task(
                task=task,
                task_type=task_type,
                scope_allow=scope_allow,
                scope_deny=scope_deny,
                acceptance=acceptance,
            )

        # ── Legacy: synchronous execution ─────────────────────────
        # Generate task ID
        task_id = self._ledger.next_id() if self._ledger else ""
        ts_start = datetime.now(timezone.utc).isoformat()
        work_dir = cwd or self.default_cwd

        cli = shutil.which("claude")
        if not cli:
            for candidate in [
                Path.home() / ".local" / "bin" / "claude.exe",
                Path.home() / ".local" / "bin" / "claude",
                Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd",
            ]:
                if candidate.exists():
                    cli = str(candidate)
                    break
        if not cli:
            return json.dumps({
                "task_id": task_id,
                "status": "failed",
                "summary": "Code worker CLI not found in PATH or known locations",
                "files_changed": [],
                "checks": {},
            }, indent=2)

        prompt = f"""You are a code worker. Complete the following task precisely.

## Task
{task}

## Scope
- ALLOWED paths: {scope_allow or "any"}
- DENIED paths: {scope_deny or "none"}

## Rules
- Keep changes minimal and focused
- Follow existing code patterns
- Do NOT create new documentation files unless asked
- Run relevant checks (lint, build, test) when you make code changes

## Required Output
After completing the task, output EXACTLY one JSON block as your FINAL message:
```json
{{
  "status": "done|failed|blocked",
  "summary": "1-2 sentence summary",
  "files_changed": ["file1", "file2"],
  "risk_signals": [],
  "checks": {{
    "lint": "pass|fail|skipped",
    "typecheck": "pass|fail|skipped",
    "build": "pass|fail|skipped",
    "test": "pass|fail|skipped"
  }}
}}
```"""

        cmd = [
            cli,
            "-p", prompt,
            "--output-format", "text",
            "--permission-mode", "bypassPermissions",
            "--allowedTools", "Bash Edit Read Write Glob Grep",
            "--model", model,
        ]

        start_time = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(work_dir).resolve()),
            )
            try:
                stdout, stderr = await self._run_with_progress(
                    proc, timeout, task_id
                )
            except asyncio.TimeoutError:
                proc.kill()
                duration = time.monotonic() - start_time
                _log_usage(task, model, duration, "timeout", 124)
                record = self._build_record(
                    task_id, ts_start, duration, model, task, work_dir,
                    status="timeout", exit_code=124,
                    summary=f"Worker timed out after {timeout}s",
                )
                await self._save_and_sync(record)
                return json.dumps({
                    "task_id": task_id,
                    "status": "timeout",
                    "summary": f"Worker timed out after {timeout}s",
                    "files_changed": [],
                    "checks": {},
                    "duration_s": round(duration, 1),
                }, indent=2)

            duration = time.monotonic() - start_time
            raw = (stdout.decode("utf-8", errors="replace") +
                   ("\n" + stderr.decode("utf-8", errors="replace") if stderr else ""))

            # Detect Claude Code auth expiry
            if proc.returncode != 0:
                auth_failure = _detect_auth_failure(raw)
                if auth_failure:
                    _log_usage(task, model, duration, "auth_expired", proc.returncode)
                    record = self._build_record(
                        task_id, ts_start, duration, model, task, work_dir,
                        status="auth_expired", exit_code=proc.returncode,
                        summary=auth_failure,
                        risk_signals=["auth_expired"],
                    )
                    await self._save_and_sync(record)
                    return json.dumps({
                        "task_id": task_id,
                        "status": "auth_expired",
                        "summary": auth_failure,
                        "files_changed": [],
                        "checks": {},
                        "duration_s": round(duration, 1),
                    }, indent=2)

            report = _parse_report(raw)
            report["task_id"] = task_id
            report["exit_code"] = proc.returncode
            report["worker_model"] = model
            report["duration_s"] = round(duration, 1)

            _log_usage(task, model, duration, report.get("status", "unknown"), proc.returncode)

            # Save to ledger + Notion
            record = self._build_record(
                task_id, ts_start, duration, model, task, work_dir,
                status=report.get("status", "unknown"),
                exit_code=proc.returncode or 0,
                summary=report.get("summary", ""),
                files_changed=report.get("files_changed", []),
                risk_signals=report.get("risk_signals", []),
                checks=report.get("checks", {}),
            )
            notion_url = await self._save_and_sync(record)
            if notion_url:
                report["notion_url"] = notion_url

            return json.dumps(report, indent=2, ensure_ascii=False)

        except Exception as e:
            duration = time.monotonic() - start_time
            _log_usage(task, model, duration, "error", -1)
            record = self._build_record(
                task_id, ts_start, duration, model, task, work_dir,
                status="error", exit_code=-1,
                summary=f"Worker dispatch error: {str(e)}",
            )
            await self._save_and_sync(record)
            return json.dumps({
                "task_id": task_id,
                "status": "failed",
                "summary": f"Worker dispatch error: {str(e)}",
                "files_changed": [],
                "checks": {},
                "duration_s": round(duration, 1),
            }, indent=2)

    # ── Non-blocking queue ─────────────────────────────────────────

    def _queue_task(
        self,
        task: str,
        task_type: str,
        scope_allow: str,
        scope_deny: str,
        acceptance: str,
    ) -> str:
        """Create a T-xxx task definition and return immediately."""
        dl = self._dual_ledger
        assert dl is not None

        allow_list = [s.strip() for s in scope_allow.split(",") if s.strip()] if scope_allow else []
        deny_list = [s.strip() for s in scope_deny.split(",") if s.strip()] if scope_deny else []

        risk = classify_risk(task_type, task, allow_list)

        task_id = dl.next_task_id()
        status = "pending_approval" if risk == "L3" else "todo"

        acceptance_checks = []
        if acceptance:
            acceptance_checks.append({"type": "command", "value": acceptance})

        td = TaskDefinition(
            task_id=task_id,
            goal=task,
            scope={"allow": allow_list, "deny": deny_list},
            risk_level=risk,
            task_type=task_type,
            acceptance=acceptance_checks,
            status=status,
            created_by="zero",
        )
        dl.save_task(td)

        if status == "pending_approval":
            msg = f"Task {task_id} queued (L3 — needs `/approve {task_id}`)"
        else:
            msg = f"Task {task_id} queued ({risk}, auto-approved, Ralph Loop will execute)"

        logger.info(f"Queued {task_id}: {task[:80]} [{risk}, {status}]")
        return json.dumps({
            "task_id": task_id,
            "risk_level": risk,
            "status": status,
            "task_type": task_type,
            "message": msg,
        }, indent=2)

    # ── Helpers ───────────────────────────────────────────────────

    def _build_record(
        self,
        task_id: str,
        ts_start: str,
        duration: float,
        model: str,
        task: str,
        cwd: str,
        *,
        status: str = "unknown",
        exit_code: int = -1,
        summary: str = "",
        files_changed: list[str] | None = None,
        risk_signals: list[str] | None = None,
        checks: dict[str, str] | None = None,
    ) -> TaskRecord:
        return TaskRecord(
            task_id=task_id,
            ts_start=ts_start,
            ts_end=datetime.now(timezone.utc).isoformat(),
            duration_s=round(duration, 1),
            model=model,
            task=task,
            task_preview=task[:120],
            cwd=cwd,
            status=status,
            exit_code=exit_code,
            summary=summary,
            files_changed=files_changed or [],
            risk_signals=risk_signals or [],
            checks=checks or {},
        )

    async def _save_and_sync(self, record: TaskRecord) -> str | None:
        """Save to ledger and optionally sync to Notion. Returns Notion URL."""
        if self._ledger:
            self._ledger.save(record)
        notion_url = await self._sync_to_notion(record)
        if notion_url:
            record.notion_url = notion_url
        return notion_url
