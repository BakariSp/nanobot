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


def _parse_worker_tail(output: str) -> dict:
    """Extract the small machine-readable JSON tail from worker output.

    Best-effort: returns empty dict fields on parse failure — the caller
    uses exit code as the primary success/failure signal.
    """
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
    return {"status": "", "files_changed": []}


def _extract_summary(raw: str, max_chars: int = 300) -> str:
    """Extract a human-readable summary from raw worker output.

    Strategy: find the last markdown heading and take content after it,
    or fall back to the last N characters.
    """
    lines = raw.rstrip().split("\n")
    # Find last heading
    last_heading_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("## ") or lines[i].startswith("# "):
            last_heading_idx = i
            break
    if last_heading_idx >= 0:
        section = "\n".join(lines[last_heading_idx:]).strip()
        # Remove the JSON tail if present
        json_start = section.rfind('{"status"')
        if json_start > 0:
            section = section[:json_start].strip()
        if section:
            return section[:max_chars]
    # Fallback: last N chars, stripping JSON tail
    tail = raw[-max_chars * 2:] if len(raw) > max_chars * 2 else raw
    json_start = tail.rfind('{"status"')
    if json_start > 0:
        tail = tail[:json_start]
    return tail.strip()[-max_chars:]


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


# ── Raw Output Capture ────────────────────────────────────────────

RAW_OUTPUT_DIR = Path.home() / ".nanobot" / "data" / "reports"


def _save_raw_output(run_id: str, raw: str) -> None:
    """Save raw worker stdout/stderr to W-XXX.raw.log for inspection."""
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_OUTPUT_DIR / f"{run_id}.raw.log"
    try:
        path.write_text(raw, encoding="utf-8")
        logger.info(f"Raw output saved → {path} ({len(raw)} chars)")
    except OSError as e:
        logger.warning(f"Failed to save raw output for {run_id}: {e}")


def get_raw_output(run_id: str) -> str | None:
    """Read raw worker output. Returns None if not found."""
    path = RAW_OUTPUT_DIR / f"{run_id}.raw.log"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


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

## Output Guidelines
- Write your findings naturally: what you discovered, which files you changed, why you made those decisions, and anything that needs attention.
- Use headings, bullet points, code snippets — whatever makes the report clear.
- Include file paths for every change.
- This natural-language report is the primary deliverable — make it thorough.

## Machine-Readable Tail (REQUIRED)
As your VERY LAST output, append exactly this small JSON block:
```json
{{"status": "done|failed|blocked", "files_changed": ["path1", "path2"]}}
```
Keep it minimal. Your natural-language report above is what matters."""


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
        "--output-format", "text",
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
            # Capture partial output before timeout
            _save_raw_output(run.run_id, f"[TIMEOUT after {WORKER_TIMEOUT}s]\n(partial output not available after kill)")
            return run

        run.duration_s = round(time.monotonic() - start_time, 1)
        run.exit_code = proc.returncode or 0
        raw = stdout.decode("utf-8", errors="replace")
        if stderr:
            raw += "\n--- STDERR ---\n" + stderr.decode("utf-8", errors="replace")

        # Save raw output for later inspection
        _save_raw_output(run.run_id, raw)

        # Auth failure detection
        if proc.returncode != 0 and _detect_auth_failure(raw):
            run.status = "failed"
            run.summary = "Claude auth expired. Run `claude` to re-authorize."
            run.failure_diagnosis = "auth_expired"
            return run

        # Primary: exit code determines success/failure
        if proc.returncode == 0:
            run.status = "success"
        else:
            run.status = "failed"

        # Secondary: parse tail JSON for files_changed (best-effort)
        tail = _parse_worker_tail(raw)
        run.files_changed = tail.get("files_changed", [])

        # Override: if exit 0 but worker explicitly says failed/blocked, trust it
        tail_status = tail.get("status", "")
        if run.status == "success" and tail_status in ("failed", "blocked"):
            run.status = "failed"

        # Extract summary from natural-language output
        run.summary = _extract_summary(raw)

        if run.status == "failed" and not run.failure_diagnosis:
            run.failure_diagnosis = "worker_error"

        return run

    except Exception as e:
        run.duration_s = round(time.monotonic() - start_time, 1)
        run.status = "failed"
        run.summary = f"Execution error: {str(e)}"
        run.failure_diagnosis = "worker_error"
        _save_raw_output(run.run_id, f"[EXCEPTION] {e}")
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


# ── Notion sync ──────────────────────────────────────────────────


def _load_notion_config() -> tuple[str, str]:
    """Load Notion API token and database ID from nanobot config.

    Returns:
        (api_token, database_id) — both empty strings if not configured.
    """
    config_path = Path.home() / ".nanobot" / "config.json"
    if not config_path.exists():
        return "", ""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        notion = data.get("tools", {}).get("notion", {})
        return notion.get("api_token", notion.get("apiToken", "")), notion.get(
            "database_id", notion.get("databaseId", "")
        )
    except (json.JSONDecodeError, OSError):
        return "", ""


# Cache config at module level (loaded once on first call)
_notion_api_token: str | None = None
_notion_database_id: str | None = None


def _clean_raw_for_notion(raw: str, max_chars: int = 8000) -> str:
    """Clean raw worker output for Notion: strip verbose tool-use noise, keep findings."""
    lines = raw.split("\n")
    kept: list[str] = []
    total = 0
    for line in lines:
        # Skip verbose Claude CLI internal lines
        if any(line.startswith(p) for p in ("[tool_use]", "[tool_result]", "[thinking]", "⏺ ")):
            continue
        kept.append(line)
        total += len(line) + 1
        if total > max_chars:
            kept.append("... (truncated, see raw log for full output)")
            break
    return "\n".join(kept)


async def _sync_run_to_notion(
    run: WorkerRunRecord,
    task: TaskDefinition,
    ledger: DualLedger,
) -> str | None:
    """Save/update a worker run to Notion with the full report body.

    - First run for a task → creates a new Notion page.
    - Subsequent retries → appends a new section to the existing page.

    Stores notion_page_id/url back on the WorkerRunRecord.

    Returns:
        Page URL on success, None on failure or if Notion is not configured.
    """
    global _notion_api_token, _notion_database_id
    if _notion_api_token is None:
        _notion_api_token, _notion_database_id = _load_notion_config()

    if not _notion_api_token or not _notion_database_id:
        return None

    try:
        from nanobot.agent.tools.notion_save import (
            append_notion_blocks,
            create_notion_page,
        )

        # Read raw log for report body
        raw = get_raw_output(run.run_id) or "(raw log not available)"
        cleaned_body = _clean_raw_for_notion(raw)

        # Build report content
        meta_lines = [
            f"## Run: {run.run_id}",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| **Status** | {run.status} |",
            f"| **Model** | {run.model} |",
            f"| **Duration** | {run.duration_s}s |",
            f"| **Exit Code** | {run.exit_code} |",
            "",
            "## Goal",
            task.goal[:500],
            "",
            "## Report",
            cleaned_body,
        ]
        if run.files_changed:
            meta_lines += ["", "## Files Changed"]
            for f in run.files_changed:
                meta_lines.append(f"- `{f}`")
        raw_log_path = str(RAW_OUTPUT_DIR / f"{run.run_id}.raw.log")
        meta_lines += ["", f"**Raw log:** `{raw_log_path}`"]

        content = "\n".join(meta_lines)

        # Check if task already has a Notion page from a previous run
        previous_page_id = None
        for prev_run_id in task.worker_runs:
            if prev_run_id == run.run_id:
                continue
            prev_run = ledger.get_run(prev_run_id)
            if prev_run and prev_run.notion_page_id:
                previous_page_id = prev_run.notion_page_id
                break

        if previous_page_id:
            # Append to existing page
            separator = f"\n---\n\n# Retry: {run.run_id}\n\n"
            ok = await append_notion_blocks(
                api_token=_notion_api_token,
                page_id=previous_page_id,
                content=separator + content,
            )
            if ok:
                run.notion_page_id = previous_page_id
                # Reuse the URL from the previous run
                for prev_run_id in task.worker_runs:
                    prev_run = ledger.get_run(prev_run_id)
                    if prev_run and prev_run.notion_page_url:
                        run.notion_page_url = prev_run.notion_page_url
                        break
                ledger.save_run(run)
                logger.info(f"Notion appended {run.run_id} to existing page {previous_page_id}")
                return run.notion_page_url
            return None
        else:
            # Create new page with full report
            title_content = f"# Worker Report: {task.task_id}\n\n" + content
            url = await create_notion_page(
                api_token=_notion_api_token,
                database_id=_notion_database_id,
                title=f"Worker Report {task.task_id}",
                content=title_content,
                tags=["worker-report", run.status],
                content_type="Worker Report",
            )
            if url:
                # Extract page_id from URL (notion.so pages have id in URL)
                # URL format: https://www.notion.so/Title-<page_id_without_dashes>
                import re as _re
                page_id_match = _re.search(r'([0-9a-f]{32})$', url.replace('-', ''))
                if page_id_match:
                    raw_id = page_id_match.group(1)
                    run.notion_page_id = f"{raw_id[:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:]}"
                run.notion_page_url = url
                ledger.save_run(run)
                logger.info(f"Notion sync OK for {run.run_id}: {url}")
            return url
    except Exception as e:
        logger.warning(f"Notion sync failed for {run.run_id}: {e}")
        return None


async def _archive_stale_done_tasks(ledger: DualLedger) -> int:
    """Archive tasks that have been 'done' for more than 24 hours.

    Updates Notion page status to Archived and marks task as archived in ledger.
    Returns the number of tasks archived.
    """
    global _notion_api_token, _notion_database_id
    if _notion_api_token is None:
        _notion_api_token, _notion_database_id = _load_notion_config()

    stale = ledger.list_tasks(status="done", older_than_hours=24)
    archived = 0
    for task in stale:
        # Try to archive in Notion
        if _notion_api_token:
            # Find notion page_id from any run
            for run_id in task.worker_runs:
                run = ledger.get_run(run_id)
                if run and run.notion_page_id:
                    try:
                        from nanobot.agent.tools.notion_save import update_notion_page_property
                        await update_notion_page_property(
                            api_token=_notion_api_token,
                            page_id=run.notion_page_id,
                            properties={
                                "Tags": {"multi_select": [{"name": "archived"}]},
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to archive Notion page for {task.task_id}: {e}")
                    break
        ledger.update_task_status(task.task_id, "archived")
        archived += 1
        logger.info(f"Archived stale done task {task.task_id}")
    return archived


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

            # Sync to Notion (pass task + ledger for retry-update logic)
            await _sync_run_to_notion(run, task, ledger)

            # Notify 零号 with enriched context
            raw_log_path = str(RAW_OUTPUT_DIR / f"{run.run_id}.raw.log")
            ledger.append_notification(
                task_id=task.task_id,
                run_id=run.run_id,
                status=run.status,
                summary=run.summary[:200],
                goal=task.goal[:200],
                raw_log_path=raw_log_path,
                duration_s=run.duration_s,
                failure_diagnosis=run.failure_diagnosis or "",
                notion_page_url=getattr(run, "notion_page_url", "") or "",
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

            # Periodically archive stale done tasks
            try:
                archived = await _archive_stale_done_tasks(ledger)
                if archived:
                    logger.info(f"Archived {archived} stale done task(s)")
            except Exception as e:
                logger.warning(f"Archive sweep failed: {e}")

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
