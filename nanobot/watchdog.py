"""Watchdog — Lightweight supervisor for `nanobot doctor`.

Doctor is the single control plane that manages gateway + ralph_loop
internally. Watchdog only ensures doctor stays alive.

Run:
    python -m nanobot.watchdog
"""

import asyncio
import sys
import time
from pathlib import Path

from loguru import logger

RESTART_DELAY = 5   # seconds to wait before restarting
MAX_CRASHES = 5     # crash-loop threshold
CRASH_WINDOW = 120  # seconds (wider than doctor's own 60s window)


async def supervise(name: str, cmd: list[str]) -> None:
    """Run a subprocess forever, restarting on crash with crash-loop detection."""
    crash_times: list[float] = []

    while True:
        logger.info(f"[{name}] Starting: {' '.join(cmd)}")
        start = time.monotonic()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=None,   # inherit parent stdout
            stderr=None,   # inherit parent stderr
        )

        await proc.wait()
        duration = round(time.monotonic() - start, 1)
        exit_code = proc.returncode

        if exit_code == 0:
            logger.info(f"[{name}] Exited cleanly after {duration}s")
        else:
            logger.warning(f"[{name}] Crashed (exit {exit_code}) after {duration}s")
            now = time.monotonic()
            crash_times.append(now)
            crash_times[:] = [t for t in crash_times if now - t < CRASH_WINDOW]

            if len(crash_times) >= MAX_CRASHES:
                logger.error(
                    f"[{name}] {MAX_CRASHES} crashes in {CRASH_WINDOW}s — "
                    f"crash loop detected. Stopping watchdog."
                )
                return

        logger.info(f"[{name}] Restarting in {RESTART_DELAY}s...")
        await asyncio.sleep(RESTART_DELAY)


async def main() -> None:
    """Supervise the doctor process (which manages gateway + ralph_loop)."""
    python = sys.executable
    await supervise("doctor", [python, "-m", "nanobot", "doctor"])


if __name__ == "__main__":
    logger.add(
        Path.home() / ".nanobot" / "logs" / "watchdog.log",
        rotation="5 MB",
        retention="7 days",
        level="INFO",
    )
    asyncio.run(main())
