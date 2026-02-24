"""Task ledger for worker dispatch tracking.

Provides:
- Legacy: TaskRecord + TaskLedger (JSONL-based W-xxx only)
- New:    TaskDefinition (T-xxx) + WorkerRunRecord (W-xxx) + DualLedger
          + classify_risk() for L1/L2/L3 risk classification
"""

import fnmatch
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

DATA_DIR = Path.home() / ".nanobot" / "data"
LEDGER_PATH = DATA_DIR / "task_ledger.jsonl"
REPORTS_DIR = DATA_DIR / "reports"


# ═══════════════════════════════════════════════════════════════════
# Risk Classification
# ═══════════════════════════════════════════════════════════════════

L3_PATH_PATTERNS = [
    "**/migration/**",
    "**/db/migration/**",
    "**/auth/**",
    "**/security/**",
    "**/jwt/**",
    "**/.env*",
    "**/config/prod*",
    "**/application-prod*",
    "**/docker-compose.prod*",
    "**/Dockerfile",
    "**/routes.ts",
    "**/openapi*",
]

L3_KEYWORDS = [
    "delete branch", "drop table", "remove migration",
    "deploy", "production", "schema change",
    "api contract", "breaking change",
    "rollback", "force push",
]


def classify_risk(task_type: str, goal: str, scope_allow: list[str]) -> str:
    """Classify task risk level: L1 (read-only), L2 (safe code change), L3 (needs approval)."""
    # Rule 1: type override
    if task_type in ("recon", "plan", "verify"):
        return "L1"

    # Rule 2: path escalation
    for pattern in scope_allow:
        for l3 in L3_PATH_PATTERNS:
            if fnmatch.fnmatch(pattern, l3) or fnmatch.fnmatch(l3, pattern):
                return "L3"

    # Rule 3: keyword escalation
    goal_lower = goal.lower()
    for kw in L3_KEYWORDS:
        if kw in goal_lower:
            return "L3"

    # Rule 4: default
    return "L2"


# ═══════════════════════════════════════════════════════════════════
# New Dual-ID Schemas (T-xxx / W-xxx)
# ═══════════════════════════════════════════════════════════════════

TASK_STATUSES = ("draft", "todo", "pending_approval", "doing", "waiting_for_input", "done", "blocked", "paused", "cancelled", "archived")
RUN_STATUSES = ("created", "running", "waiting_for_input", "success", "failed")
TASK_TYPES = ("recon", "fix", "feature", "refactor", "plan", "verify")


@dataclass
class TaskDefinition:
    """T-xxx: Task definition (what to do)."""
    task_id: str
    goal: str
    scope: dict = field(default_factory=lambda: {"allow": [], "deny": []})
    risk_level: str = "L2"
    task_type: str = "fix"
    acceptance: list[dict] = field(default_factory=list)
    status: str = "draft"
    plan_id: str | None = None
    worker_runs: list[str] = field(default_factory=list)
    blocked_by: str | None = None
    created_by: str = "zero"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class WorkerRunRecord:
    """W-xxx: One execution attempt of a task."""
    run_id: str
    task_id: str
    prompt_sent: str = ""
    prompt_context: list[str] = field(default_factory=list)
    model: str = "opus"
    cwd: str = ""
    status: str = "created"
    exit_code: int = -1
    duration_s: float = 0.0
    files_changed: list[str] = field(default_factory=list)
    summary: str = ""
    checks: dict[str, str] = field(default_factory=dict)
    failure_diagnosis: str | None = None
    lesson_learned: str | None = None
    report_path: str = ""
    notion_page_id: str | None = None
    notion_page_url: str | None = None
    created_at: str = ""


class DualLedger:
    """Manages T-xxx/W-xxx task definitions and worker runs in a directory."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or (DATA_DIR / "task_ledger")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = DATA_DIR / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._seq = self._load_seq()

    # ── ID generation ─────────────────────────────────────────────

    def _load_seq(self) -> dict:
        seq_file = self.base_dir / ".seq.json"
        if seq_file.exists():
            try:
                return json.loads(seq_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"task_seq": 0, "run_seq": 0}

    def _save_seq(self) -> None:
        seq_file = self.base_dir / ".seq.json"
        seq_file.write_text(json.dumps(self._seq), encoding="utf-8")

    def next_task_id(self) -> str:
        self._seq["task_seq"] += 1
        self._save_seq()
        return f"T-{self._seq['task_seq']:03d}"

    def next_run_id(self) -> str:
        self._seq["run_seq"] += 1
        self._save_seq()
        return f"W-{self._seq['run_seq']:03d}"

    # ── Persistence ───────────────────────────────────────────────

    def save_task(self, task: TaskDefinition) -> None:
        if not task.created_at:
            task.created_at = datetime.now(timezone.utc).isoformat()
        task.updated_at = datetime.now(timezone.utc).isoformat()
        path = self.base_dir / f"{task.task_id}.json"
        path.write_text(json.dumps(asdict(task), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Task {task.task_id} saved ({task.status})")

    def save_run(self, run: WorkerRunRecord) -> None:
        if not run.created_at:
            run.created_at = datetime.now(timezone.utc).isoformat()
        path = self.base_dir / f"{run.run_id}.json"
        path.write_text(json.dumps(asdict(run), indent=2, ensure_ascii=False), encoding="utf-8")
        # Markdown report
        report_path = self.reports_dir / f"{run.run_id}.md"
        report_path.write_text(self._render_run_report(run), encoding="utf-8")
        logger.info(f"Run {run.run_id} saved ({run.status}) → {report_path}")

    # ── Query ─────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> TaskDefinition | None:
        path = self.base_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TaskDefinition(**{k: v for k, v in data.items() if k in TaskDefinition.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load {task_id}: {e}")
            return None

    def get_run(self, run_id: str) -> WorkerRunRecord | None:
        path = self.base_dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return WorkerRunRecord(**{k: v for k, v in data.items() if k in WorkerRunRecord.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load {run_id}: {e}")
            return None

    def list_tasks(self, status: str | None = None, older_than_hours: float = 0) -> list[TaskDefinition]:
        tasks = []
        cutoff = None
        if older_than_hours > 0:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        for f in sorted(self.base_dir.glob("T-*.json")):
            task = self.get_task(f.stem)
            if not task:
                continue
            if status is not None and task.status != status:
                continue
            if cutoff and task.updated_at:
                try:
                    updated = datetime.fromisoformat(task.updated_at)
                    if updated > cutoff:
                        continue
                except ValueError:
                    pass
            tasks.append(task)
        return tasks

    def update_task_status(self, task_id: str, new_status: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        task.status = new_status
        self.save_task(task)
        return True

    def add_worker_run(self, task_id: str, run_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        if run_id not in task.worker_runs:
            task.worker_runs.append(run_id)
        self.save_task(task)
        return True

    # ── Notifications (Ralph Loop → 零号) ─────────────────────────

    @property
    def _notify_path(self) -> Path:
        return DATA_DIR / ".ralph_notifications.jsonl"

    def append_notification(
        self,
        task_id: str,
        run_id: str,
        status: str,
        summary: str,
        goal: str = "",
        raw_log_path: str = "",
        duration_s: float = 0.0,
        failure_diagnosis: str = "",
        notion_page_url: str = "",
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "run_id": run_id,
            "status": status,
            "summary": summary,
            "goal": goal,
            "raw_log_path": raw_log_path,
            "duration_s": duration_s,
            "failure_diagnosis": failure_diagnosis,
            "notion_page_url": notion_page_url,
        }
        with open(self._notify_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_and_clear_notifications(self) -> list[dict]:
        if not self._notify_path.exists():
            return []
        entries = []
        for line in self._notify_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self._notify_path.write_text("", encoding="utf-8")
        return entries

    # ── Status summary (for /status command) ──────────────────────

    def status_summary(self) -> str:
        active_statuses = ("todo", "doing", "pending_approval", "blocked")
        tasks = [t for t in self.list_tasks() if t.status in active_statuses]
        if not tasks:
            return "No active tasks."
        lines = []
        for t in tasks:
            runs = f" ({len(t.worker_runs)} runs)" if t.worker_runs else ""
            lines.append(f"  {t.task_id}: {t.goal[:60]} [{t.status}, {t.risk_level}]{runs}")
        return f"{len(tasks)} active tasks:\n" + "\n".join(lines)

    # ── Markdown rendering ────────────────────────────────────────

    @staticmethod
    def _render_run_report(r: WorkerRunRecord) -> str:
        lines = [
            f"# Worker Report: {r.run_id}",
            "",
            f"**Task:** {r.task_id}",
            f"**Status:** {r.status}",
            f"**Model:** {r.model}",
            f"**Duration:** {r.duration_s}s",
            f"**Started:** {r.created_at}",
            "",
            "## Summary",
            "",
            r.summary or "(no summary)",
            "",
        ]
        if r.files_changed:
            lines += ["## Files Changed", ""]
            for f in r.files_changed:
                lines.append(f"- `{f}`")
            lines.append("")
        if r.checks:
            lines += ["## Checks", "", "| Check | Result |", "|---|---|"]
            for k, v in r.checks.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
        if r.failure_diagnosis:
            lines += [f"## Failure Diagnosis: {r.failure_diagnosis}", ""]
        if r.lesson_learned:
            lines += ["## Lesson Learned", "", r.lesson_learned, ""]
        return "\n".join(lines)


@dataclass
class TaskRecord:
    """One worker dispatch task."""

    task_id: str  # "W-001"
    ts_start: str  # ISO 8601 UTC
    ts_end: str = ""
    duration_s: float = 0.0
    model: str = "opus"
    task: str = ""
    task_preview: str = ""
    cwd: str = ""
    status: str = "in_progress"
    exit_code: int = -1
    summary: str = ""
    files_changed: list[str] = field(default_factory=list)
    risk_signals: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)
    notion_url: str | None = None
    report_path: str = ""


class TaskLedger:
    """Persistent task tracker backed by JSONL + per-task Markdown reports."""

    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._next_seq = self._read_last_seq() + 1

    # ── ID generation ─────────────────────────────────────────────

    def _read_last_seq(self) -> int:
        if not LEDGER_PATH.exists():
            return 0
        last = 0
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tid = json.loads(line).get("task_id", "")
                    if tid.startswith("W-"):
                        last = max(last, int(tid[2:]))
                except (json.JSONDecodeError, ValueError):
                    continue
        return last

    def next_id(self) -> str:
        tid = f"W-{self._next_seq:03d}"
        self._next_seq += 1
        return tid

    # ── Persistence ───────────────────────────────────────────────

    def save(self, record: TaskRecord) -> None:
        """Append record to JSONL and write Markdown report."""
        report_path = REPORTS_DIR / f"{record.task_id}.md"
        record.report_path = str(report_path)

        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        md = self.render_markdown(record)
        report_path.write_text(md, encoding="utf-8")
        logger.info(f"Task {record.task_id} saved → {report_path}")

    # ── Query ─────────────────────────────────────────────────────

    def get(self, task_id: str) -> TaskRecord | None:
        if not LEDGER_PATH.exists():
            return None
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("task_id") == task_id:
                        return TaskRecord(**{
                            k: v for k, v in data.items()
                            if k in TaskRecord.__dataclass_fields__
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
        return None

    def list_recent(self, n: int = 10) -> list[TaskRecord]:
        if not LEDGER_PATH.exists():
            return []
        records: list[TaskRecord] = []
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(TaskRecord(**{
                        k: v for k, v in data.items()
                        if k in TaskRecord.__dataclass_fields__
                    }))
                except (json.JSONDecodeError, TypeError):
                    continue
        return records[-n:]

    # ── Markdown rendering ────────────────────────────────────────

    @staticmethod
    def render_markdown(r: TaskRecord) -> str:
        lines = [
            f"# Worker Report: {r.task_id}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| **Task ID** | {r.task_id} |",
            f"| **Status** | {r.status} |",
            f"| **Model** | {r.model} |",
            f"| **Started** | {r.ts_start} |",
            f"| **Duration** | {r.duration_s}s |",
            f"| **CWD** | {r.cwd} |",
            "",
            "## Task",
            "",
            r.task,
            "",
            "## Summary",
            "",
            r.summary or "(no summary)",
            "",
        ]

        if r.files_changed:
            lines += ["## Files Changed", ""]
            for f in r.files_changed:
                lines.append(f"- `{f}`")
            lines.append("")

        if r.checks:
            lines += [
                "## Quality Checks",
                "",
                "| Check | Result |",
                "|---|---|",
            ]
            for check, result in r.checks.items():
                lines.append(f"| {check} | {result} |")
            lines.append("")

        if r.risk_signals:
            lines += ["## Risk Signals", ""]
            for sig in r.risk_signals:
                lines.append(f"- {sig}")
            lines.append("")

        if r.notion_url:
            lines += [f"## Notion", "", f"[View in Notion]({r.notion_url})", ""]

        return "\n".join(lines)
