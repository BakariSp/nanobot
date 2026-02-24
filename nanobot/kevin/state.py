"""Kevin's state management — portfolio, trade log, status."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class KevinState:
    """Manages Kevin's persistent state files.

    Files (all under workspace/kevin/):
      portfolio.json  — current balance, initial balance, P&L
      trades.jsonl    — every real trade (append-only, auto-logged by pm_trade)
      decisions.jsonl — structured decision log (auto-logged by end_turn tool)
      memo.md         — note to next wakeup (written by end_turn, injected into system prompt)
    """

    def __init__(self, workspace: Path):
        self._dir = workspace / "kevin"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._portfolio_path = self._dir / "portfolio.json"
        self._trades_path = self._dir / "trades.jsonl"

    # ── Portfolio ─────────────────────────────────────────────

    def get_portfolio(self) -> dict[str, Any]:
        """Read current portfolio state."""
        if self._portfolio_path.exists():
            try:
                return json.loads(self._portfolio_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"balance": 0.0, "initial_balance": 0.0, "updated_at": ""}

    def update_portfolio(self, balance: float, initial_balance: float | None = None) -> None:
        """Update portfolio snapshot."""
        current = self.get_portfolio()
        current["balance"] = balance
        if initial_balance is not None:
            current["initial_balance"] = initial_balance
        current["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Calculate P&L
        init = current.get("initial_balance", 0)
        if init > 0:
            current["pnl"] = round(balance - init, 4)
            current["pnl_pct"] = round((balance - init) / init * 100, 2)
        else:
            current["pnl"] = 0.0
            current["pnl_pct"] = 0.0

        self._portfolio_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Trade log ─────────────────────────────────────────────

    def log_trade(self, trade: dict[str, Any]) -> None:
        """Append a trade record to trades.jsonl."""
        trade["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self._trades_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"Kevin trade log write failed: {e}")

    def get_recent_trades(self, n: int = 10) -> list[dict[str, Any]]:
        """Read the last N trades."""
        if not self._trades_path.exists():
            return []
        trades = []
        try:
            with open(self._trades_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            return []
        return trades[-n:]

    # ── Status summary ────────────────────────────────────────

    def get_status_summary(self) -> str:
        """Format a human-readable status summary."""
        p = self.get_portfolio()
        balance = p.get("balance", 0)
        initial = p.get("initial_balance", 0)
        pnl = p.get("pnl", 0)
        pnl_pct = p.get("pnl_pct", 0)
        updated = p.get("updated_at", "unknown")

        total_rent = p.get("total_rent_paid", 0)

        lines = [
            f"Balance: {balance:.2f} USDC",
            f"Initial: {initial:.2f} USDC",
            f"P&L: {pnl:+.2f} USDC ({pnl_pct:+.1f}%)",
            f"Rent paid: {total_rent:.2f} USDC (daily burn)",
            f"Last update: {updated}",
        ]

        trades = self.get_recent_trades(5)
        if trades:
            lines.append("")
            lines.append("Recent trades:")
            for t in trades:
                side = t.get("side", "?")
                amount = t.get("amount", 0)
                market = t.get("market_question", t.get("token_id", "?"))[:60]
                ts = t.get("ts", "")[:16]
                lines.append(f"  [{ts}] {side} {amount} — {market}")
        else:
            lines.append("No trades yet.")

        return "\n".join(lines)

    # ── Trade statistics (for review cycle) ────────────────────

    def get_trade_stats(self) -> dict[str, Any]:
        """Compute aggregate trade statistics for Kevin's self-review.

        Returns counts, volume, time span — raw material for Kevin to
        evaluate his own performance.
        """
        all_trades = self.get_recent_trades(10000)  # effectively all
        if not all_trades:
            return {"total_trades": 0}

        buys = [t for t in all_trades if t.get("side") == "BUY"]
        sells = [t for t in all_trades if t.get("side") == "SELL"]
        errors = [t for t in all_trades if "error" in str(t.get("result", ""))]

        total_buy_vol = sum(t.get("amount", 0) for t in buys)
        total_sell_vol = sum(t.get("amount", 0) for t in sells)

        # Time span
        first_ts = all_trades[0].get("ts", "")[:10] if all_trades else ""
        last_ts = all_trades[-1].get("ts", "")[:10] if all_trades else ""

        # Unique markets traded
        markets = {t.get("market_question", t.get("token_id", ""))
                   for t in all_trades if t.get("market_question") or t.get("token_id")}

        return {
            "total_trades": len(all_trades),
            "buys": len(buys),
            "sells": len(sells),
            "errors": len(errors),
            "buy_volume": round(total_buy_vol, 2),
            "sell_volume": round(total_sell_vol, 2),
            "markets_traded": len(markets),
            "first_trade": first_ts,
            "last_trade": last_ts,
        }

    def get_recent_decisions(self, n: int = 10) -> list[str]:
        """Read the last N entries from decisions.jsonl.

        decisions.jsonl is free-form (Kevin writes it himself via write_file),
        so each line is returned as-is — could be JSON or plain text.
        """
        decisions_path = self._dir / "decisions.jsonl"
        if not decisions_path.exists():
            return []
        lines: list[str] = []
        try:
            with open(decisions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
        except OSError:
            return []
        return lines[-n:]

    def get_strategy(self) -> str:
        """Read Kevin's current strategy from strategies.md."""
        path = self._dir / "strategies.md"
        try:
            content = path.read_text(encoding="utf-8").strip()
            return content
        except (FileNotFoundError, OSError):
            return ""

    def format_review_context(self) -> str:
        """Build a review context block for the system prompt.

        Includes: trade stats, recent decisions, current strategy.
        Kevin sees this every wakeup so he naturally reflects.
        """
        sections = []

        # Trade stats
        stats = self.get_trade_stats()
        if stats["total_trades"] > 0:
            sections.append(
                f"Trade stats: {stats['total_trades']} trades "
                f"({stats['buys']}B/{stats['sells']}S, {stats['errors']} errors), "
                f"buy vol {stats['buy_volume']} / sell vol {stats['sell_volume']} USDC, "
                f"{stats['markets_traded']} markets, "
                f"from {stats['first_trade']} to {stats['last_trade']}"
            )

        # Recent decisions
        decisions = self.get_recent_decisions(5)
        if decisions:
            sections.append("Recent decisions:")
            for d in decisions:
                # Truncate long entries for prompt space
                sections.append(f"  {d[:200]}")

        # Current strategy
        strategy = self.get_strategy()
        if strategy:
            sections.append(f"\nCurrent strategy:\n{strategy}")

        return "\n".join(sections) if sections else ""

    # ── Daily rent (survival pressure) ─────────────────────────

    def deduct_daily_rent(self, daily_cost: float) -> float:
        """Deduct rent for elapsed days since last settlement. Returns new balance.

        Rent accrues by calendar day, whether Kevin is awake or not.
        On each wakeup we settle: days_elapsed * daily_cost.
        This makes sleeping NOT a cost-saving strategy — time is the enemy.
        """
        if daily_cost <= 0:
            return self.get_portfolio().get("balance", 0)

        p = self.get_portfolio()
        old_balance = p.get("balance", 0)
        last_rent_date = p.get("last_rent_date", "")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if last_rent_date == today:
            # Already settled today
            return old_balance

        # Calculate days elapsed
        if last_rent_date:
            try:
                last_dt = datetime.strptime(last_rent_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                now_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                days = max(1, (now_dt - last_dt).days)
            except ValueError:
                days = 1
        else:
            days = 1  # First settlement

        rent = round(daily_cost * days, 4)
        new_balance = max(0, old_balance - rent)

        self.update_portfolio(new_balance)

        # Log the deduction
        self.log_trade({
            "type": "daily_rent",
            "days": days,
            "daily_rate": daily_cost,
            "amount": -rent,
            "balance_before": round(old_balance, 4),
            "balance_after": round(new_balance, 4),
        })

        # Update rent tracking in portfolio
        p = self.get_portfolio()
        p["last_rent_date"] = today
        p["total_rent_paid"] = round(p.get("total_rent_paid", 0) + rent, 4)
        self._portfolio_path.write_text(
            json.dumps(p, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(f"Kevin daily rent: {days}d × {daily_cost} = -{rent} USDT → balance {new_balance:.2f}")
        return new_balance

    # ── Memo (cross-session notes to self) ────────────────────

    def get_memo(self) -> str:
        """Read memo.md — last note Kevin left for his next wakeup."""
        path = self._dir / "memo.md"
        try:
            content = path.read_text(encoding="utf-8").strip()
            return content
        except (FileNotFoundError, OSError):
            return ""

    def save_memo(self, content: str) -> None:
        """Overwrite memo.md with new content."""
        path = self._dir / "memo.md"
        try:
            path.write_text(content.strip(), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Kevin memo write failed: {e}")

    # ── Decision log (system-managed) ──────────────────────────

    def log_decision(self, record: dict[str, Any]) -> None:
        """Append a structured decision record to decisions.jsonl.

        Called by the end_turn tool — Kevin no longer needs to call
        write_file manually for decision logging.
        """
        record["ts"] = datetime.now(timezone.utc).isoformat()
        decisions_path = self._dir / "decisions.jsonl"
        try:
            with open(decisions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"Kevin decision log write failed: {e}")

    # ── Turn tracking & simulation ────────────────────────────

    SIMULATION_TURNS = 10  # Paper trade for first N turns

    def get_turn_count(self) -> int:
        """Read current turn count from portfolio."""
        return self.get_portfolio().get("turn_count", 0)

    def increment_turn(self) -> int:
        """Increment turn counter. Called by end_turn tool. Returns new count."""
        p = self.get_portfolio()
        count = p.get("turn_count", 0) + 1
        p["turn_count"] = count
        self._portfolio_path.write_text(
            json.dumps(p, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return count

    def is_simulation(self) -> bool:
        """True if Kevin is still in paper-trading phase (first N turns)."""
        return self.get_turn_count() < self.SIMULATION_TURNS

    # ── Life check ────────────────────────────────────────────

    def is_dead(self) -> bool:
        """Kevin is dead if balance ≤ 0.01 USDC."""
        p = self.get_portfolio()
        return p.get("balance", 0) <= 0.01
