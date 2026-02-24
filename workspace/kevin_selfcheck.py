#!/usr/bin/env python3
"""Kevin self-check script — prints startup diagnostics.

Run at Kevin startup to quickly diagnose state:
- Current turn count
- Simulation mode status (yes/no)
- Next wakeup time (if scheduled)
- Leverage setting (from config)
- Current holdings (BTC, ETH, etc.)

Usage:
    python d:/Insight-AI/nanobot/workspace/kevin_selfcheck.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".nanobot" / "config.json"


def get_data_dir() -> Path:
    """Get the nanobot data directory."""
    return Path.home() / ".nanobot" / "workspace"


def load_portfolio(workspace: Path) -> dict:
    """Load portfolio.json."""
    portfolio_path = workspace / "kevin" / "portfolio.json"
    if portfolio_path.exists():
        try:
            return json.loads(portfolio_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def load_config() -> dict:
    """Load config.json."""
    config_path = get_config_path()
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def get_next_wakeup() -> str | None:
    """Try to read next wakeup time from cron state.
    
    Note: This is a simplified check. For accurate cron info,
    query the running nanobot instance.
    """
    # Cron jobs are stored in memory, not persisted to disk by default.
    # We can only estimate based on last decision's next_wakeup_minutes.
    decisions_path = get_data_dir() / "kevin" / "decisions.jsonl"
    if not decisions_path.exists():
        return None
    
    try:
        with open(decisions_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return None
        last_line = lines[-1].strip()
        if not last_line:
            return None
        last_decision = json.loads(last_line)
        minutes = last_decision.get("next_wakeup_minutes", 15)
        ts = last_decision.get("ts", "")
        if ts:
            # Parse timestamp and add minutes
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            next_dt = dt.timestamp() + (minutes * 60)
            next_str = datetime.fromtimestamp(next_dt, tz=timezone.utc).isoformat()
            return f"{next_str} (in ~{minutes} min from last turn)"
    except Exception:
        pass
    return "Unknown (check cron service)"


def get_holdings(workspace: Path) -> list[dict]:
    """Read current holdings from trades.jsonl (simulated positions)."""
    trades_path = workspace / "kevin" / "trades.jsonl"
    if not trades_path.exists():
        return []
    
    holdings = {}
    try:
        with open(trades_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trade = json.loads(line)
                    if trade.get("type") == "daily_rent":
                        continue
                    if trade.get("simulated"):
                        # Paper trade — track position
                        symbol = trade.get("symbol", "")
                        side = trade.get("side", "")
                        qty = trade.get("qty", 0)
                        if symbol not in holdings:
                            holdings[symbol] = {"qty": 0, "buys": 0, "sells": 0}
                        if side == "Buy":
                            holdings[symbol]["qty"] += qty
                            holdings[symbol]["buys"] += qty
                        elif side == "Sell":
                            holdings[symbol]["qty"] -= qty
                            holdings[symbol]["sells"] += qty
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    
    return [
        {"symbol": k, "net_qty": v["qty"], "total_buys": v["buys"], "total_sells": v["sells"]}
        for k, v in holdings.items()
        if v["qty"] != 0
    ]


def print_selfcheck():
    """Print Kevin self-check report."""
    print("=" * 60)
    print("KEVIN SELF-CHECK REPORT")
    print("=" * 60)
    
    # Load data
    config = load_config()
    workspace = get_data_dir()
    portfolio = load_portfolio(workspace)
    
    # Kevin config
    kevin_cfg = config.get("kevin", {})
    kevin_enabled = kevin_cfg.get("enabled", False)
    bybit_testnet = kevin_cfg.get("bybitTestnet", False)
    daily_cost = kevin_cfg.get("dailyCost", 0.5)
    
    # Portfolio state
    turn_count = portfolio.get("turn_count", 0)
    balance = portfolio.get("balance", 0)
    initial_balance = portfolio.get("initial_balance", 0)
    pnl = portfolio.get("pnl", 0)
    pnl_pct = portfolio.get("pnl_pct", 0)
    total_rent = portfolio.get("total_rent_paid", 0)
    
    # Simulation status
    SIMULATION_TURNS = 10
    is_simulation = turn_count < SIMULATION_TURNS
    remaining_sim = SIMULATION_TURNS - turn_count if is_simulation else 0
    
    # Holdings
    holdings = get_holdings(workspace)
    
    # Next wakeup
    next_wakeup = get_next_wakeup()
    
    # Print report
    print(f"\n📊 KEVIN STATUS")
    print(f"   Enabled:          {'✅ Yes' if kevin_enabled else '❌ No'}")
    print(f"   Testnet:          {'✅ Yes' if bybit_testnet else '❌ No (REAL trading)'}")
    print(f"   Current Turn:     {turn_count}")
    print(f"   Simulation Mode:  {'✅ YES (paper trading)' if is_simulation else '❌ NO (LIVE trading)'}")
    if is_simulation:
        print(f"   Remaining Sim:    {remaining_sim} turns until LIVE")
    
    print(f"\n💰 PORTFOLIO")
    print(f"   Balance:          {balance:.4f} USDT")
    print(f"   Initial:          {initial_balance:.4f} USDT")
    print(f"   P&L:              {pnl:+.4f} USDT ({pnl_pct:+.2f}%)")
    print(f"   Total Rent Paid:  {total_rent:.4f} USDT (daily: {daily_cost} USDT)")
    
    print(f"\n⏰ NEXT WAKEUP")
    print(f"   Scheduled:        {next_wakeup or 'Not scheduled (call end_turn to schedule)'}")
    
    print(f"\n🪙 CURRENT HOLDINGS (Paper Positions)")
    if holdings:
        for h in holdings:
            print(f"   {h['symbol']}: {h['net_qty']:+.6f} (B:{h['total_buys']:.6f} S:{h['total_sells']:.6f})")
    else:
        print("   (No open positions)")
    
    print(f"\n⚠️  LEVERAGE NOTE")
    print(f"   Kevin uses spot trading by default (1x leverage).")
    print(f"   To use leverage, set category='linear' and leverage=N in trade calls.")
    print(f"   Current config does not specify default leverage — per-trade setting applies.")
    
    print("\n" + "=" * 60)
    print("END OF SELF-CHECK")
    print("=" * 60)


if __name__ == "__main__":
    print_selfcheck()
