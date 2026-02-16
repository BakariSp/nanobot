"""Doctor crash notifications — out-of-process Telegram alerting + event log.

Two layers:
  1. Primary: direct Telegram Bot API call (works when gateway is down).
  2. Compensation: append to doctor_events.jsonl for post-recovery replay.
"""

import asyncio
import json
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# ── Event log ────────────────────────────────────────────────────

EVENTS_PATH = Path.home() / ".nanobot" / "data" / "doctor_events.jsonl"


def make_event(
    component: str,
    exit_code: int | None,
    stderr_tail: str = "",
    crash_count: int = 0,
    window_s: int = 60,
    run_id: str = "",
    event_type: str = "crash",
) -> dict[str, Any]:
    """Create a unified doctor event dict."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "component": component,
        "exit_code": exit_code,
        "stderr_tail": stderr_tail[-500:] if stderr_tail else "",
        "crash_count": crash_count,
        "window_s": window_s,
        "run_id": run_id,
    }


def append_event(event: dict) -> None:
    """Append event to doctor_events.jsonl (append-only)."""
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_unplayed_events() -> list[dict]:
    """Read all events. Gateway can call this after recovery to replay via MessageBus."""
    if not EVENTS_PATH.exists():
        return []
    events = []
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def clear_events() -> None:
    """Clear event log after replay."""
    if EVENTS_PATH.exists():
        EVENTS_PATH.write_text("", encoding="utf-8")


# ── Telegram direct send ─────────────────────────────────────────

_last_notify_ts: dict[str, float] = {}  # event_type -> last send monotonic time
DEDUP_WINDOW_S = 60


async def _send_telegram_async(
    token: str,
    chat_ids: list[str],
    event: dict,
    proxy: str | None = None,
) -> None:
    """Internal async implementation of Telegram alert."""
    event_type = event.get("event_type", "unknown")
    now = _time.monotonic()

    # Rate limit: one notification per event_type per DEDUP_WINDOW_S
    if event_type in _last_notify_ts:
        if now - _last_notify_ts[event_type] < DEDUP_WINDOW_S:
            logger.debug(f"Suppressed duplicate {event_type} notification (rate limited)")
            return

    if not token or not chat_ids:
        logger.debug("Telegram notification skipped (no token or chat_ids)")
        return

    text = _format_alert(event)

    try:
        from telegram import Bot
        from telegram.request import HTTPXRequest

        kwargs: dict[str, Any] = {}
        if proxy:
            kwargs["request"] = HTTPXRequest(proxy=proxy)

        bot = Bot(token=token, **kwargs)
        for chat_id in chat_ids:
            try:
                await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
            except Exception as e:
                logger.warning(f"Failed to send alert to {chat_id}: {e}")

        _last_notify_ts[event_type] = now
        logger.info(f"Crash alert sent to {len(chat_ids)} chat(s)")

    except ImportError:
        logger.warning("python-telegram-bot not installed; skipping notification")
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")


def send_telegram_alert(
    token: str,
    chat_ids: list[str],
    event: dict,
    proxy: str | None = None,
) -> None:
    """Send crash alert via Telegram Bot API (sync wrapper for use in doctor loop).

    Rate-limited: at most one message per event_type per 60s.
    """
    try:
        asyncio.run(_send_telegram_async(token, chat_ids, event, proxy))
    except Exception as e:
        logger.error(f"send_telegram_alert failed: {e}")


def _format_alert(event: dict) -> str:
    """Format event as a human-readable Telegram HTML message."""
    event_type = event.get("event_type", "unknown")
    component = event.get("component", "unknown")
    exit_code = event.get("exit_code")
    crash_count = event.get("crash_count", 0)
    stderr_tail = event.get("stderr_tail", "")
    ts = event.get("ts", "")
    run_id = event.get("run_id", "")

    icon = {
        "crash": "\u274c",
        "crash_loop": "\u203c\ufe0f",
        "rollback": "\u23ea",
        "recovery": "\u2705",
        "startup": "\U0001f680",
    }.get(event_type, "\U0001f514")

    lines = [
        f"{icon} <b>nanobot doctor: {event_type}</b>",
        f"Component: <code>{component}</code>",
        f"Exit code: <code>{exit_code}</code>",
    ]
    if crash_count > 0:
        lines.append(f"Crashes in window: {crash_count}")
    if run_id:
        lines.append(f"Run: <code>{run_id}</code>")
    if stderr_tail:
        safe = stderr_tail[:300].replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f"<pre>{safe}</pre>")
    lines.append(f"<i>{ts}</i>")
    return "\n".join(lines)
