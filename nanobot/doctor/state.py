"""Doctor state persistence — tracks last_good_commit, crash counts, run history.

State file: ~/.nanobot/data/doctor_state.json
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

STATE_PATH = Path.home() / ".nanobot" / "data" / "doctor_state.json"
STABLE_THRESHOLD_S = 120  # seconds of stable run before recording last_good_commit


def load_state() -> dict[str, Any]:
    """Load doctor state from disk."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load doctor state: {e}")
    return _default_state()


def save_state(state: dict[str, Any]) -> None:
    """Persist doctor state to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _default_state() -> dict[str, Any]:
    return {
        "last_good_commit": None,
        "last_good_branch": None,
        "last_good_ts": None,
        "current_run_id": None,
        "current_run_start": None,
        "total_crashes": 0,
        "last_crash_ts": None,
        "consecutive_crash_loops": 0,
        "rollback_history": [],
    }


def new_run_id() -> str:
    """Generate a new run ID based on current timestamp."""
    return f"dr-{int(time.time())}"


def record_crash(state: dict, event: dict) -> dict:
    """Update state after a crash."""
    state["total_crashes"] = state.get("total_crashes", 0) + 1
    state["last_crash_ts"] = event.get("ts", datetime.now(timezone.utc).isoformat())
    save_state(state)
    return state


def record_crash_loop(state: dict) -> dict:
    """Increment consecutive crash loop counter."""
    state["consecutive_crash_loops"] = state.get("consecutive_crash_loops", 0) + 1
    save_state(state)
    return state


def record_stable_run(state: dict, commit: str, branch: str) -> dict:
    """Record current commit as last known good after stable period."""
    state["last_good_commit"] = commit
    state["last_good_branch"] = branch
    state["last_good_ts"] = datetime.now(timezone.utc).isoformat()
    state["consecutive_crash_loops"] = 0
    save_state(state)
    return state


def record_rollback(
    state: dict, from_commit: str, to_commit: str, success: bool
) -> dict:
    """Record a rollback attempt."""
    history = state.get("rollback_history", [])
    history.append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "from_commit": from_commit,
            "to_commit": to_commit,
            "success": success,
        }
    )
    state["rollback_history"] = history[-20:]  # keep last 20
    save_state(state)
    return state


def should_attempt_rollback(state: dict) -> bool:
    """Determine if automatic rollback should be attempted."""
    if not state.get("last_good_commit"):
        return False
    # Don't rollback if we already rolled back twice and still crashing
    if state.get("consecutive_crash_loops", 0) >= 2:
        return False
    return True
