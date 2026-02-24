"""Plugin edit tool — sandboxed self-modification with snapshot, validation, and rollback.

Allows the agent to safely modify its own plugin-ring files (providers, peripheral tools)
while protecting core-ring files (loop, context, channels, config, etc.) from modification.
"""

import hashlib
import importlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


# ── Ring definitions ────────────────────────────────────────────────

# Resolve the nanobot package root once
_NANOBOT_PKG_ROOT = Path(__file__).resolve().parent.parent.parent  # nanobot/nanobot/

# Plugin ring — bot MAY self-modify these (glob-style relative to _NANOBOT_PKG_ROOT)
PLUGIN_RING_GLOBS = [
    "providers/transcription.py",
    "providers/tts.py",
    "providers/oss.py",
    "providers/volcengine_tts.py",
    "agent/tools/send_photo.py",
    "agent/tools/voice_reply.py",
    "agent/tools/web.py",
    "agent/tools/drive_save.py",
    "agent/tools/telegram_search.py",
    "agent/tools/notion_save.py",
    "agent/tools/notion_tasks.py",
]

# Precompute absolute paths for fast lookup
PLUGIN_RING_PATHS: set[Path] = {
    (_NANOBOT_PKG_ROOT / g).resolve() for g in PLUGIN_RING_GLOBS
}

# Storage paths
_PLUGINS_DIR = Path.home() / ".nanobot" / "plugins"
_SNAPSHOTS_DIR = _PLUGINS_DIR / "snapshots"
_CHANGELOG_PATH = _PLUGINS_DIR / "changelog.jsonl"


def is_plugin_file(path: str | Path) -> bool:
    """Check if a file path is in the plugin ring."""
    resolved = Path(path).resolve()
    return resolved in PLUGIN_RING_PATHS


def is_nanobot_file(path: str | Path) -> bool:
    """Check if a file path is inside the nanobot package."""
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(_NANOBOT_PKG_ROOT)
        return True
    except ValueError:
        return False


def _short_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _snapshot_file(file_path: Path) -> Path:
    """Create a snapshot of a file. Returns the snapshot path."""
    _SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    snapshot_name = f"{file_path.stem}.{ts}{file_path.suffix}.bak"
    snapshot_path = _SNAPSHOTS_DIR / snapshot_name
    shutil.copy2(file_path, snapshot_path)
    logger.info(f"Plugin snapshot: {file_path.name} -> {snapshot_path}")
    return snapshot_path


def _restore_snapshot(snapshot_path: Path, target_path: Path) -> bool:
    """Restore a file from snapshot. Returns True on success."""
    try:
        shutil.copy2(snapshot_path, target_path)
        logger.info(f"Plugin restored: {snapshot_path} -> {target_path}")
        return True
    except Exception as e:
        logger.error(f"Plugin restore failed: {e}")
        return False


def _validate_module(file_path: Path) -> tuple[bool, str]:
    """Validate a modified plugin by attempting to compile and import it.

    Returns (success, message).
    """
    # Step 1: Syntax check via compile()
    try:
        source = file_path.read_text(encoding="utf-8")
        compile(source, str(file_path), "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Step 2: Try to reload the module if it's already imported
    # Convert file path to module name: nanobot/providers/tts.py -> nanobot.providers.tts
    try:
        rel = file_path.resolve().relative_to(_NANOBOT_PKG_ROOT.parent)
        module_name = str(rel).replace("/", ".").replace("\\", ".").removesuffix(".py")
    except ValueError:
        # Not inside the package tree — skip import validation
        return True, "OK (syntax-only check, outside package tree)"

    if module_name in sys.modules:
        try:
            importlib.reload(sys.modules[module_name])
            logger.info(f"Plugin reload OK: {module_name}")
        except Exception as e:
            return False, f"Import/reload error: {type(e).__name__}: {e}"
    else:
        try:
            importlib.import_module(module_name)
            logger.info(f"Plugin import OK: {module_name}")
        except Exception as e:
            return False, f"Import error: {type(e).__name__}: {e}"

    # Step 3: Call health_check() if the module defines one
    mod = sys.modules.get(module_name)
    if mod and hasattr(mod, "health_check"):
        try:
            result = mod.health_check()
            if result is False:
                return False, "health_check() returned False"
            logger.info(f"Plugin health_check OK: {module_name}")
        except Exception as e:
            return False, f"health_check() raised: {type(e).__name__}: {e}"

    return True, "OK"


def _append_changelog(entry: dict) -> None:
    """Append an entry to the plugin changelog."""
    _PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CHANGELOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _update_doctor_state(file_path: Path, snapshot_path: Path, reason: str) -> None:
    """Record the plugin edit in doctor state for crash correlation."""
    try:
        from nanobot.doctor.state import load_state, save_state
        state = load_state()
        state["last_plugin_edit"] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "file": str(file_path),
            "snapshot_path": str(snapshot_path),
            "reason": reason,
        }
        save_state(state)
    except Exception as e:
        logger.warning(f"Failed to update doctor state with plugin edit: {e}")


def read_changelog(limit: int = 5) -> list[dict]:
    """Read the last N changelog entries. Used by context builder for boot recovery."""
    if not _CHANGELOG_PATH.exists():
        return []
    try:
        lines = _CHANGELOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
    except Exception:
        return []


class PluginEditTool(Tool):
    """Edit a nanobot plugin file with automatic snapshot, validation, and rollback.

    Safer alternative to edit_file for self-modifying the agent's own code.
    Only files in the plugin ring (providers, peripheral tools) can be edited.
    Core files (loop, context, channels, config) are protected.
    """

    @property
    def name(self) -> str:
        return "plugin_edit"

    @property
    def description(self) -> str:
        return (
            "Edit a nanobot plugin file (providers, peripheral tools) with automatic "
            "snapshot and rollback. Use this instead of edit_file when modifying "
            "nanobot's own code. Core files (loop.py, context.py, channels/, config/) "
            "are protected and cannot be edited.\n\n"
            "Editable files (plugin ring):\n"
            "- providers/transcription.py, tts.py, oss.py, volcengine_tts.py\n"
            "- agent/tools/send_photo.py, voice_reply.py, web.py, drive_save.py, "
            "telegram_search.py, notion_save.py, notion_tasks.py\n\n"
            "After editing, the tool automatically:\n"
            "1. Snapshots the original file\n"
            "2. Applies the edit\n"
            "3. Validates (syntax check + import + health_check)\n"
            "4. Rolls back if validation fails"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the plugin file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this edit is needed (logged for audit trail)",
                },
            },
            "required": ["path", "old_text", "new_text", "reason"],
        }

    async def execute(
        self, path: str, old_text: str, new_text: str, reason: str, **kwargs: Any
    ) -> str:
        file_path = Path(path).resolve()

        # ── Gate: plugin ring only ──
        if not is_plugin_file(file_path):
            if is_nanobot_file(file_path):
                return (
                    f"Error: {file_path.name} is in nanobot's core ring and cannot be "
                    "self-modified. Core changes require manual intervention by the user."
                )
            return (
                f"Error: {path} is not a recognized nanobot plugin file. "
                "plugin_edit only works on files in the plugin ring. "
                "Use edit_file for non-nanobot files."
            )

        if not file_path.exists():
            return f"Error: File not found: {path}"

        # ── Read current content ──
        content = file_path.read_text(encoding="utf-8")
        if old_text not in content:
            return "Error: old_text not found in file. Make sure it matches exactly."

        count = content.count(old_text)
        if count > 1:
            return (
                f"Warning: old_text appears {count} times. "
                "Provide more context to make it unique."
            )

        # ── Snapshot ──
        snapshot_path = _snapshot_file(file_path)

        # ── Apply edit ──
        new_content = content.replace(old_text, new_text, 1)
        file_path.write_text(new_content, encoding="utf-8")

        # ── Validate ──
        valid, msg = _validate_module(file_path)

        if not valid:
            # Rollback
            _restore_snapshot(snapshot_path, file_path)
            _append_changelog({
                "ts": datetime.now(timezone.utc).isoformat(),
                "file": str(file_path),
                "reason": reason,
                "old_text_hash": _short_hash(old_text),
                "new_text_hash": _short_hash(new_text),
                "status": "rolled_back",
                "validation_error": msg,
            })
            return (
                f"Edit rolled back — validation failed: {msg}\n"
                f"Original file restored from snapshot."
            )

        # ── Success ──
        _append_changelog({
            "ts": datetime.now(timezone.utc).isoformat(),
            "file": str(file_path),
            "reason": reason,
            "old_text_hash": _short_hash(old_text),
            "new_text_hash": _short_hash(new_text),
            "status": "applied",
        })
        _update_doctor_state(file_path, snapshot_path, reason)

        return (
            f"Plugin edit applied successfully: {file_path.name}\n"
            f"Reason: {reason}\n"
            f"Validation: {msg}\n"
            f"Snapshot: {snapshot_path.name}"
        )
