"""Tool for inspecting worker execution logs, reports, and raw output.

Gives the agent real visibility into what workers actually did/failed/output,
instead of relying on summary-only notifications.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.task_ledger import DualLedger

DATA_DIR = Path.home() / ".nanobot" / "data"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = Path.home() / ".nanobot" / "logs"


class WorkerLogsTool(Tool):
    """Inspect worker execution details: task records, run logs, raw output, and Ralph Loop status.

    Use this to get REAL information about what workers did, why they failed, and what's happening now.
    """

    def __init__(self, dual_ledger: DualLedger):
        self._ledger = dual_ledger

    @property
    def name(self) -> str:
        return "worker_logs"

    @property
    def description(self) -> str:
        return (
            "Inspect worker execution details. Actions: "
            "'get_task' (T-xxx full info + all runs), "
            "'get_run' (W-xxx record + report), "
            "'get_output' (W-xxx raw stdout/stderr — the actual execution log), "
            "'list_active' (all non-done tasks with status), "
            "'ralph_status' (Ralph Loop heartbeat + recent log). "
            "Use this BEFORE answering questions about worker status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_task", "get_run", "get_output",
                        "list_active", "ralph_status",
                    ],
                    "description": (
                        "Action to perform. "
                        "get_task: Full task definition + all worker runs. "
                        "get_run: One worker run record + markdown report. "
                        "get_output: Raw worker stdout/stderr log (the actual execution trace). "
                        "list_active: All tasks not yet done (todo/doing/blocked/pending_approval). "
                        "ralph_status: Ralph Loop process health + recent log entries."
                    ),
                },
                "id": {
                    "type": "string",
                    "description": "Task ID (T-xxx) or Run ID (W-xxx), required for get_task/get_run/get_output",
                },
                "tail": {
                    "type": "integer",
                    "description": "Number of lines to return from end of log (default 80, max 500)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self, action: str, id: str = "", tail: int = 80, **kwargs: Any
    ) -> str:
        tail = min(max(tail, 10), 500)

        if action == "get_task":
            return self._get_task(id)
        elif action == "get_run":
            return self._get_run(id)
        elif action == "get_output":
            return self._get_output(id, tail)
        elif action == "list_active":
            return self._list_active()
        elif action == "ralph_status":
            return self._ralph_status(tail)
        else:
            return f"Unknown action: {action}. Use: get_task, get_run, get_output, list_active, ralph_status."

    # ── get_task ─────────────────────────────────────────────────

    def _get_task(self, task_id: str) -> str:
        if not task_id:
            return "Error: 'id' is required. Provide a T-xxx task ID."
        task = self._ledger.get_task(task_id)
        if not task:
            return f"Task {task_id} not found."

        lines = [
            f"# Task {task.task_id}",
            "",
            f"**Goal:** {task.goal}",
            f"**Status:** {task.status}",
            f"**Risk:** {task.risk_level}",
            f"**Type:** {task.task_type}",
            f"**Created:** {task.created_at}",
            f"**Updated:** {task.updated_at}",
        ]

        if task.scope.get("allow"):
            lines.append(f"**Scope allow:** {', '.join(task.scope['allow'])}")
        if task.scope.get("deny"):
            lines.append(f"**Scope deny:** {', '.join(task.scope['deny'])}")
        if task.blocked_by:
            lines.append(f"**Blocked by:** {task.blocked_by}")

        if task.worker_runs:
            lines += ["", "## Worker Runs"]
            for run_id in task.worker_runs:
                run = self._ledger.get_run(run_id)
                if run:
                    status_icon = {"success": "OK", "failed": "FAIL", "running": "RUN"}.get(run.status, run.status)
                    lines.append(
                        f"- **{run.run_id}** [{status_icon}] {run.duration_s}s — {run.summary[:100]}"
                    )
                    if run.failure_diagnosis:
                        lines.append(f"  Diagnosis: {run.failure_diagnosis}")
                    if run.lesson_learned:
                        lines.append(f"  Lesson: {run.lesson_learned.strip()}")
                else:
                    lines.append(f"- **{run_id}** (record not found)")
        else:
            lines.append("\n*No worker runs yet.*")

        return "\n".join(lines)

    # ── get_run ──────────────────────────────────────────────────

    def _get_run(self, run_id: str) -> str:
        if not run_id:
            return "Error: 'id' is required. Provide a W-xxx run ID."
        run = self._ledger.get_run(run_id)
        if not run:
            # Try reading the markdown report directly
            report_path = REPORTS_DIR / f"{run_id}.md"
            if report_path.exists():
                return report_path.read_text(encoding="utf-8")
            return f"Run {run_id} not found."

        # Read the markdown report if available
        report_path = REPORTS_DIR / f"{run_id}.md"
        report_md = ""
        if report_path.exists():
            report_md = report_path.read_text(encoding="utf-8")

        if report_md:
            return report_md

        # Fallback: render from record
        lines = [
            f"# Worker Run {run.run_id}",
            "",
            f"**Task:** {run.task_id}",
            f"**Status:** {run.status}",
            f"**Model:** {run.model}",
            f"**Duration:** {run.duration_s}s",
            f"**Exit code:** {run.exit_code}",
            f"**Started:** {run.created_at}",
            "",
            "## Summary",
            run.summary or "(no summary)",
        ]
        if run.files_changed:
            lines += ["", "## Files Changed"]
            for f in run.files_changed:
                lines.append(f"- `{f}`")
        if run.checks:
            lines += ["", "## Checks"]
            for k, v in run.checks.items():
                lines.append(f"- {k}: {v}")
        if run.failure_diagnosis:
            lines.append(f"\n**Failure diagnosis:** {run.failure_diagnosis}")
        if run.lesson_learned:
            lines.append(f"\n**Lesson:** {run.lesson_learned}")
        return "\n".join(lines)

    # ── get_output ───────────────────────────────────────────────

    def _get_output(self, run_id: str, tail: int) -> str:
        if not run_id:
            return "Error: 'id' is required. Provide a W-xxx run ID."

        raw_path = REPORTS_DIR / f"{run_id}.raw.log"
        if not raw_path.exists():
            return (
                f"No raw output found for {run_id}. "
                "This run may predate the log capture feature, or the file was cleaned up."
            )

        try:
            content = raw_path.read_text(encoding="utf-8")
        except OSError as e:
            return f"Error reading output for {run_id}: {e}"

        lines = content.splitlines()
        total = len(lines)
        if total <= tail:
            return f"## Raw Output: {run_id} ({total} lines)\n\n```\n{content}\n```"

        # Return tail portion
        truncated = lines[-tail:]
        return (
            f"## Raw Output: {run_id} (showing last {tail} of {total} lines)\n\n"
            f"*[{total - tail} lines truncated — use tail={total} to see all]*\n\n"
            f"```\n" + "\n".join(truncated) + "\n```"
        )

    # ── list_active ──────────────────────────────────────────────

    def _list_active(self) -> str:
        active_statuses = ("todo", "doing", "pending_approval", "blocked")
        tasks = []
        for status in active_statuses:
            tasks.extend(self._ledger.list_tasks(status=status))

        if not tasks:
            return "No active tasks. All tasks are done or cancelled."

        lines = [f"## Active Tasks ({len(tasks)})"]
        for t in tasks:
            icon = {
                "todo": "QUEUE", "doing": "RUN",
                "pending_approval": "WAIT", "blocked": "BLOCK",
            }.get(t.status, t.status)
            runs_info = ""
            if t.worker_runs:
                last_run = self._ledger.get_run(t.worker_runs[-1])
                if last_run:
                    runs_info = f" | last run: {last_run.run_id} [{last_run.status}]"
            lines.append(
                f"- **{t.task_id}** [{icon}] {t.risk_level} — {t.goal[:80]}"
                f" ({len(t.worker_runs)} runs){runs_info}"
            )
            if t.status == "blocked" and t.blocked_by:
                lines.append(f"  blocked by: {t.blocked_by}")

        return "\n".join(lines)

    # ── ralph_status ─────────────────────────────────────────────

    def _ralph_status(self, tail: int) -> str:
        lines = ["## Ralph Loop Status"]

        # 1. Heartbeat
        heartbeat_path = self._ledger.base_dir / ".ralph_heartbeat"
        if heartbeat_path.exists():
            try:
                hb = json.loads(heartbeat_path.read_text(encoding="utf-8"))
                pid = hb.get("pid", "?")
                task = hb.get("current_task") or "idle"
                run = hb.get("current_run") or "-"
                last_beat = hb.get("last_beat", "?")
                # Calculate age
                age_str = ""
                try:
                    beat_time = datetime.fromisoformat(last_beat)
                    age_s = (datetime.now(timezone.utc) - beat_time).total_seconds()
                    if age_s < 60:
                        age_str = f" ({age_s:.0f}s ago)"
                    else:
                        age_str = f" ({age_s / 60:.1f}min ago)"
                except (ValueError, TypeError):
                    pass
                lines += [
                    "",
                    f"**PID:** {pid}",
                    f"**Current task:** {task}",
                    f"**Current run:** {run}",
                    f"**Last heartbeat:** {last_beat}{age_str}",
                ]
            except (json.JSONDecodeError, OSError):
                lines.append("\n*Heartbeat file exists but unreadable.*")
        else:
            lines.append("\n**Ralph Loop is NOT running** (no heartbeat file).")

        # 2. Pending notifications
        notify_path = DATA_DIR / ".ralph_notifications.jsonl"
        if notify_path.exists():
            try:
                notif_lines = [l for l in notify_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                if notif_lines:
                    lines.append(f"\n**Unread notifications:** {len(notif_lines)}")
                    for nl in notif_lines[-5:]:
                        try:
                            n = json.loads(nl)
                            lines.append(f"  - [{n.get('task_id')}/{n.get('run_id')}] {n.get('status')}: {n.get('summary', '')[:80]}")
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass

        # 3. Recent ralph_loop.log entries
        log_path = LOGS_DIR / "ralph_loop.log"
        if log_path.exists():
            try:
                log_content = log_path.read_text(encoding="utf-8")
                log_lines = log_content.splitlines()
                recent = log_lines[-tail:] if len(log_lines) > tail else log_lines
                lines += [
                    f"\n## Recent Ralph Log (last {len(recent)} of {len(log_lines)} lines)",
                    "",
                    "```",
                    *recent,
                    "```",
                ]
            except OSError:
                lines.append("\n*Could not read ralph_loop.log*")
        else:
            lines.append("\n*No ralph_loop.log found.*")

        return "\n".join(lines)
