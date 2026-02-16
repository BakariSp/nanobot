"""Ralph Loop — Independent task execution process.

Polls task_ledger/ for todo tasks, executes via `claude -p`,
writes W-xxx worker run records + Reports, notifies 零号.

Run as a standalone process:
    python -m nanobot.ralph_loop
"""

import asyncio
import json
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from nanobot.agent.tools.task_ledger import (
    DualLedger, TaskDefinition, WorkerRunRecord,
)

# ── Configuration ─────────────────────────────────────────────────

POLL_INTERVAL = 5          # seconds between queue checks
WORKER_TIMEOUT = 3600      # max worker execution time (1h)
HANG_TIMEOUT = 120         # kill worker if no output for this long
MAX_RETRIES = 2            # auto-retry failed tasks up to N times
HEARTBEAT_STALE_S = 30     # treat heartbeat as stale if older than this
DEFAULT_CWD = "d:/Insight-AI"

# ── Auth failure detection (reused from worker.py) ────────────────

_AUTH_KEYWORDS = re.compile(
    r"(auth\w*\s+(required|expired|failed|error)|"
    r"log\s*in\s+(required|to continue)|"
    r"unauthorized|not\s+authenticated|session\s+expired|re-?auth)",
    re.IGNORECASE,
)


def _detect_auth_failure(output: str) -> bool:
    return bool(_AUTH_KEYWORDS.search(output))


def _parse_worker_report(output: str) -> dict:
    """Extract JSON report block from worker stdout."""
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', output)
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
    return {
        "status": "unknown",
        "summary": "Could not parse worker report",
        "files_changed": [],
        "checks": {},
    }


# ── Heartbeat ─────────────────────────────────────────────────────

class Heartbeat:
    """PID lock + heartbeat file to prevent double-run and enable crash recovery."""

    def __init__(self, ledger_dir: Path):
        self.path = ledger_dir / ".ralph_heartbeat"

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns False if another instance is running."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                other_pid = data.get("pid")
                last_beat = data.get("last_beat")

                # Check heartbeat staleness first — covers Windows PID reuse
                if last_beat:
                    try:
                        beat_time = datetime.fromisoformat(last_beat)
                        age = (datetime.now(timezone.utc) - beat_time).total_seconds()
                        if age > HEARTBEAT_STALE_S:
                            logger.info(
                                f"Stale heartbeat (PID {other_pid}, last beat {age:.0f}s ago). Taking over."
                            )
                            self._write(current_task=None, current_run=None)
                            return True
                    except (ValueError, TypeError):
                        pass  # unparseable timestamp, fall through to PID check

                if other_pid and self._pid_alive(other_pid):
                    logger.warning(f"Another Ralph Loop (PID {other_pid}) is running. Exiting.")
                    return False
                else:
                    logger.info(f"Stale heartbeat (PID {other_pid} dead). Taking over.")
            except (json.JSONDecodeError, OSError):
                pass
        self._write(current_task=None, current_run=None)
        return True

    def beat(self, task_id: str | None, run_id: str | None) -> None:
        self._write(task_id, run_id)

    def release(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def _write(self, current_task: str | None, current_run: str | None) -> None:
        data = {
            "pid": os.getpid(),
            "current_task": current_task,
            "current_run": current_run,
            "last_beat": datetime.now(timezone.utc).isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


# ── Worker Execution ──────────────────────────────────────────────

def _build_worker_prompt(task: TaskDefinition) -> str:
    """Build the prompt for `claude -p`."""
    scope_allow = ", ".join(task.scope.get("allow", [])) or "any"
    scope_deny = ", ".join(task.scope.get("deny", [])) or "none"

    acceptance_cmds = []
    for a in task.acceptance:
        if a.get("type") == "command":
            acceptance_cmds.append(a["value"])

    acceptance_section = ""
    if acceptance_cmds:
        cmds = "\n".join(f"  - {c}" for c in acceptance_cmds)
        acceptance_section = f"""
## Acceptance Checks (run these after completing the task)
{cmds}
"""

    # Inject project context files if they exist
    context_files = []
    for ctx_file in ["CLAUDE.md", "PROGRESS.md"]:
        p = Path(DEFAULT_CWD) / ctx_file
        if p.exists():
            context_files.append(ctx_file)

    context_section = ""
    if context_files:
        context_section = f"""
## Project Context
Read these files first for project context: {', '.join(context_files)}
"""

    return f"""You are a code worker. Complete the following task precisely.

## Task
{task.goal}

## Scope
- ALLOWED paths: {scope_allow}
- DENIED paths: {scope_deny}
{context_section}{acceptance_section}
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
  "checks": {{
    "lint": "pass|fail|skipped",
    "typecheck": "pass|fail|skipped",
    "build": "pass|fail|skipped",
    "test": "pass|fail|skipped"
  }}
}}
```"""


async def execute_worker(
    task: TaskDefinition,
    run: WorkerRunRecord,
    ledger: DualLedger,
    heartbeat: Heartbeat,
) -> WorkerRunRecord:
    """Execute a single worker run via `claude -p`."""
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
        run.status = "failed"
        run.summary = "Claude CLI not found"
        run.failure_diagnosis = "worker_error"
        return run

    prompt = _build_worker_prompt(task)
    run.prompt_sent = prompt
    run.prompt_context = ["CLAUDE.md", "PROGRESS.md"]

    cmd = [
        cli, "-p", prompt,
        "--output-format", "stream-json",
        "--permission-mode", "bypassPermissions",
        "--allowedTools", "Bash Edit Read Write Glob Grep",
        "--model", run.model,
        "--verbose",
    ]

    start_time = time.monotonic()
    run.status = "running"
    ledger.save_run(run)
    heartbeat.beat(task.task_id, run.run_id)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(run.cwd).resolve()),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=WORKER_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            run.status = "failed"
            run.duration_s = round(time.monotonic() - start_time, 1)
            run.summary = f"Worker timed out after {WORKER_TIMEOUT}s"
            run.failure_diagnosis = "worker_hang"
            return run

        run.duration_s = round(time.monotonic() - start_time, 1)
        run.exit_code = proc.returncode or 0
        raw = stdout.decode("utf-8", errors="replace")
        if stderr:
            raw += "\n" + stderr.decode("utf-8", errors="replace")

        # Auth failure detection
        if proc.returncode != 0 and _detect_auth_failure(raw):
            run.status = "failed"
            run.summary = "Claude auth expired. Run `claude` to re-authorize."
            run.failure_diagnosis = "auth_expired"
            return run

        # Parse worker output
        report = _parse_worker_report(raw)
        worker_status = report.get("status", "unknown")
        run.status = "success" if worker_status == "done" else "failed"
        run.summary = report.get("summary", "")
        run.files_changed = report.get("files_changed", [])
        run.checks = report.get("checks", {})

        if run.status == "failed" and not run.failure_diagnosis:
            run.failure_diagnosis = "worker_error"

        return run

    except Exception as e:
        run.duration_s = round(time.monotonic() - start_time, 1)
        run.status = "failed"
        run.summary = f"Execution error: {str(e)}"
        run.failure_diagnosis = "worker_error"
        return run


# ── Crash Recovery ────────────────────────────────────────────────

def recover_orphaned_tasks(ledger: DualLedger) -> int:
    """Find tasks stuck at 'doing' and mark them blocked for retry."""
    doing_tasks = ledger.list_tasks(status="doing")
    recovered = 0
    for task in doing_tasks:
        logger.warning(f"Orphaned task {task.task_id} (was doing). Marking blocked for retry.")
        ledger.update_task_status(task.task_id, "blocked")
        recovered += 1
    return recovered


# ── Main Loop ─────────────────────────────────────────────────────

async def ralph_loop() -> None:
    """Main execution loop — poll, pick, execute, report."""
    ledger = DualLedger()
    heartbeat = Heartbeat(ledger.base_dir)

    if not heartbeat.acquire():
        logger.error("Another Ralph Loop is running. Exiting.")
        sys.exit(1)

    # Crash recovery on startup
    recovered = recover_orphaned_tasks(ledger)
    if recovered:
        logger.info(f"Recovered {recovered} orphaned tasks")

    # Re-queue blocked tasks that can be retried
    for task in ledger.list_tasks(status="blocked"):
        retry_count = len(task.worker_runs)
        if retry_count < MAX_RETRIES:
            ledger.update_task_status(task.task_id, "todo")
            logger.info(f"Re-queued {task.task_id} for retry ({retry_count}/{MAX_RETRIES})")

    logger.info("Ralph Loop started. Polling for tasks...")
    _running = True

    def _shutdown(sig, frame):
        nonlocal _running
        logger.info(f"Received signal {sig}, shutting down...")
        _running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while _running:
            heartbeat.beat(None, None)

            # Pick next todo task (oldest first)
            todo_tasks = ledger.list_tasks(status="todo")
            if not todo_tasks:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            task = todo_tasks[0]
            logger.info(f"Picking up {task.task_id}: {task.goal[:80]}")

            # Mark as doing
            ledger.update_task_status(task.task_id, "doing")

            # Create worker run
            run_id = ledger.next_run_id()
            run = WorkerRunRecord(
                run_id=run_id,
                task_id=task.task_id,
                model="opus",
                cwd=DEFAULT_CWD,
            )
            ledger.add_worker_run(task.task_id, run_id)

            # Execute
            run = await execute_worker(task, run, ledger, heartbeat)

            # Save results
            ledger.save_run(run)

            # Update task status
            if run.status == "success":
                ledger.update_task_status(task.task_id, "done")
            else:
                retry_count = len(task.worker_runs)
                if retry_count < MAX_RETRIES:
                    ledger.update_task_status(task.task_id, "todo")
                    logger.info(f"{task.task_id} failed, re-queuing ({retry_count}/{MAX_RETRIES})")
                else:
                    ledger.update_task_status(task.task_id, "blocked")
                    logger.warning(f"{task.task_id} failed {retry_count} times, marking blocked")

            # Notify 零号
            ledger.append_notification(
                task_id=task.task_id,
                run_id=run.run_id,
                status=run.status,
                summary=run.summary[:200],
            )

            # Write lesson learned on failure
            if run.status == "failed" and run.failure_diagnosis:
                progress_path = Path(DEFAULT_CWD) / "PROGRESS.md"
                lesson = f"- [{run.run_id}] {task.task_id}: {run.failure_diagnosis} — {run.summary[:100]}\n"
                run.lesson_learned = lesson
                ledger.save_run(run)
                try:
                    with open(progress_path, "a", encoding="utf-8") as f:
                        f.write(lesson)
                except OSError as e:
                    logger.warning(f"Could not write to PROGRESS.md: {e}")

            logger.info(f"{task.task_id}/{run.run_id}: {run.status} ({run.duration_s}s)")

    finally:
        heartbeat.release()
        logger.info("Ralph Loop stopped.")


def main():
    """Entry point for `python -m nanobot.ralph_loop`."""
    logger.add(
        Path.home() / ".nanobot" / "logs" / "ralph_loop.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )
    asyncio.run(ralph_loop())


if __name__ == "__main__":
    main()
