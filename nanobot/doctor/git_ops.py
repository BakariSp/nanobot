"""Safe git operations for doctor rollback — atomic stash + checkout + verify."""

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from loguru import logger


@dataclass
class GitSnapshot:
    """Snapshot of git state before an operation."""

    branch: str
    commit: str
    is_dirty: bool
    stash_name: str | None = None


class GitError(Exception):
    """Raised when a git operation fails."""


def _run_git(args: list[str], cwd: str) -> str:
    """Run a git command and return stdout. Raises GitError on failure."""
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise GitError(f"git {' '.join(args)} timed out")
    except FileNotFoundError:
        raise GitError("git not found in PATH")


def snapshot(cwd: str) -> GitSnapshot:
    """Capture current branch, commit, and dirty state."""
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    commit = _run_git(["rev-parse", "HEAD"], cwd)
    status = _run_git(["status", "--porcelain"], cwd)
    return GitSnapshot(
        branch=branch,
        commit=commit,
        is_dirty=bool(status.strip()),
    )


def safe_rollback(
    cwd: str,
    target_ref: str,
    health_check_fn: Optional[Callable[[], bool]] = None,
) -> tuple[bool, str]:
    """Atomic rollback: snapshot -> stash if dirty -> checkout -> verify -> rollback-on-failure.

    Returns (success: bool, message: str).
    """
    # Step 1: Snapshot current state
    try:
        snap = snapshot(cwd)
    except GitError as e:
        return False, f"Cannot snapshot: {e}"

    logger.info(
        f"Git snapshot: branch={snap.branch}, "
        f"commit={snap.commit[:8]}, dirty={snap.is_dirty}"
    )

    # Step 2: Stash if dirty
    if snap.is_dirty:
        ts = int(time.time())
        stash_msg = f"nanobot-auto-stash-{ts}"
        try:
            _run_git(["stash", "push", "-u", "-m", stash_msg], cwd)
            snap.stash_name = stash_msg
            logger.info(f"Stashed dirty work: {stash_msg}")
        except GitError as e:
            return False, f"Cannot stash: {e}"

    # Step 3: Checkout target ref
    try:
        _run_git(["checkout", target_ref], cwd)
        logger.info(f"Checked out: {target_ref}")
    except GitError as e:
        _try_restore(snap, cwd)
        return False, f"Checkout failed: {e}"

    # Step 4: Health check (if provided)
    if health_check_fn:
        try:
            healthy = health_check_fn()
        except Exception as e:
            logger.warning(f"Health check raised: {e}")
            healthy = False

        if not healthy:
            # Step 5: Rollback on health check failure
            logger.warning("Health check failed after rollback, reverting...")
            try:
                _run_git(["checkout", snap.commit], cwd)
                _try_restore(snap, cwd)
                return (
                    False,
                    f"Rolled back to {target_ref} but health check failed; "
                    f"reverted to {snap.commit[:8]}",
                )
            except GitError as e2:
                return False, f"CRITICAL: revert also failed: {e2}"

    # Step 6: Try to pop stash (if we stashed)
    if snap.stash_name:
        try:
            _run_git(["stash", "pop"], cwd)
            logger.info("Stash popped successfully")
        except GitError:
            logger.warning(
                f"Stash pop conflict. Stash preserved as '{snap.stash_name}'. "
                f"Manual resolution needed."
            )
            return (
                True,
                f"Rolled back to {target_ref} but stash pop had conflicts. "
                f"Stash preserved.",
            )

    return True, f"Successfully rolled back to {target_ref}"


def _try_restore(snap: GitSnapshot, cwd: str) -> None:
    """Best-effort restore of original state after a failed operation."""
    try:
        _run_git(["checkout", snap.branch], cwd)
    except GitError:
        try:
            _run_git(["checkout", snap.commit], cwd)
        except GitError:
            pass
    if snap.stash_name:
        try:
            _run_git(["stash", "pop"], cwd)
        except GitError:
            logger.warning(f"Could not pop stash '{snap.stash_name}' during restore")
