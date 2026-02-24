"""Kevin's Bybit crypto trading tools + sentiment + calculate + end_turn + status query."""

from __future__ import annotations

import math
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.kevin.state import KevinState


# ── Technical analysis helpers (pure math, no deps) ──────────


def _sma(values: list[float], period: int) -> list[float | None]:
    """Simple Moving Average."""
    result: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result


def _ema(values: list[float], period: int) -> list[float | None]:
    """Exponential Moving Average."""
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    k = 2 / (period + 1)
    # Seed with SMA
    result[period - 1] = sum(values[:period]) / period
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)  # type: ignore[operator]
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Relative Strength Index."""
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return result
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def _macd(
    closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, list[float | None]]:
    """MACD (line, signal, histogram)."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]
    # Signal line = EMA of MACD line (only non-None values)
    valid = [(i, v) for i, v in enumerate(macd_line) if v is not None]
    signal_line: list[float | None] = [None] * len(closes)
    if len(valid) >= signal:
        vals = [v for _, v in valid]
        sig = _ema(vals, signal)
        for j, (orig_i, _) in enumerate(valid):
            signal_line[orig_i] = sig[j]
    histogram: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def _bollinger(closes: list[float], period: int = 20, std_dev: float = 2.0) -> dict[str, list[float | None]]:
    """Bollinger Bands (upper, middle, lower)."""
    middle = _sma(closes, period)
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        mean = middle[i]
        std = math.sqrt(sum((x - mean) ** 2 for x in window) / period)  # type: ignore[operator]
        upper[i] = mean + std_dev * std  # type: ignore[operator]
        lower[i] = mean - std_dev * std  # type: ignore[operator]
    return {"upper": upper, "middle": middle, "lower": lower}


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float | None]:
    """Average True Range."""
    result: list[float | None] = [None] * len(closes)
    if len(closes) < 2:
        return result
    trs = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return result
    result[period - 1] = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        result[i] = (result[i - 1] * (period - 1) + trs[i]) / period  # type: ignore[operator]
    return result


# ── Calculate tool ───────────────────────────────────────────


class CalculateTool(Tool):
    """Run technical analysis on K-line data from Bybit."""

    def __init__(self, client):
        self._client = client

    @property
    def name(self) -> str:
        return "calculate"

    @property
    def description(self) -> str:
        return (
            "Compute technical indicators on a symbol's K-line data: "
            "RSI, EMA, SMA, MACD, Bollinger Bands, ATR. "
            "Fetches K-lines automatically — just specify symbol, interval, and which indicators."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair, e.g. BTCUSDT",
                },
                "interval": {
                    "type": "string",
                    "description": "K-line interval: 1,5,15,30,60,240,D (default 60 = 1h)",
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Which indicators to compute. Options: "
                        "rsi, ema_short (9), ema_long (21), sma_20, "
                        "macd, bollinger, atr. "
                        "Omit for all."
                    ),
                },
            },
            "required": ["symbol"],
        }

    async def execute(
        self,
        symbol: str,
        interval: str = "60",
        indicators: list[str] | None = None,
        **kwargs,
    ) -> str:
        try:
            klines = self._client.get_kline(symbol, interval=interval, limit=100)
            if not klines:
                return f"No K-line data for {symbol} interval={interval}"

            # Bybit returns newest first — reverse to chronological order
            klines.reverse()

            closes = [k["close"] for k in klines]
            highs = [k["high"] for k in klines]
            lows = [k["low"] for k in klines]

            all_indicators = {"rsi", "ema_short", "ema_long", "sma_20", "macd", "bollinger", "atr"}
            requested = set(indicators) if indicators else all_indicators

            lines = [f"{symbol} Technical Analysis (interval={interval}, {len(klines)} candles)"]
            lines.append(f"Latest: O={klines[-1]['open']:.2f} H={klines[-1]['high']:.2f} L={klines[-1]['low']:.2f} C={klines[-1]['close']:.2f}")
            lines.append("")

            if "rsi" in requested:
                rsi_vals = _rsi(closes, 14)
                latest = next((v for v in reversed(rsi_vals) if v is not None), None)
                if latest is not None:
                    zone = "oversold" if latest < 30 else "overbought" if latest > 70 else "neutral"
                    lines.append(f"RSI(14): {latest:.1f} ({zone})")

            if "ema_short" in requested:
                ema9 = _ema(closes, 9)
                latest = next((v for v in reversed(ema9) if v is not None), None)
                if latest is not None:
                    diff_pct = (closes[-1] - latest) / latest * 100
                    lines.append(f"EMA(9): {latest:.2f} (price {diff_pct:+.2f}% from EMA)")

            if "ema_long" in requested:
                ema21 = _ema(closes, 21)
                latest = next((v for v in reversed(ema21) if v is not None), None)
                if latest is not None:
                    diff_pct = (closes[-1] - latest) / latest * 100
                    lines.append(f"EMA(21): {latest:.2f} (price {diff_pct:+.2f}% from EMA)")

            # EMA cross signal
            if "ema_short" in requested and "ema_long" in requested:
                ema9 = _ema(closes, 9)
                ema21 = _ema(closes, 21)
                if ema9[-1] is not None and ema21[-1] is not None and ema9[-2] is not None and ema21[-2] is not None:
                    if ema9[-2] < ema21[-2] and ema9[-1] > ema21[-1]:
                        lines.append("  → EMA9/21 GOLDEN CROSS (bullish)")
                    elif ema9[-2] > ema21[-2] and ema9[-1] < ema21[-1]:
                        lines.append("  → EMA9/21 DEATH CROSS (bearish)")

            if "sma_20" in requested:
                sma20 = _sma(closes, 20)
                latest = next((v for v in reversed(sma20) if v is not None), None)
                if latest is not None:
                    lines.append(f"SMA(20): {latest:.2f}")

            if "macd" in requested:
                m = _macd(closes)
                ml = next((v for v in reversed(m["macd"]) if v is not None), None)
                sl = next((v for v in reversed(m["signal"]) if v is not None), None)
                hl = next((v for v in reversed(m["histogram"]) if v is not None), None)
                if ml is not None:
                    trend = "bullish" if (hl or 0) > 0 else "bearish"
                    lines.append(f"MACD: line={ml:.2f} signal={sl:.2f if sl else '?'} hist={hl:.2f if hl else '?'} ({trend})")

            if "bollinger" in requested:
                bb = _bollinger(closes)
                u = next((v for v in reversed(bb["upper"]) if v is not None), None)
                mid = next((v for v in reversed(bb["middle"]) if v is not None), None)
                lo = next((v for v in reversed(bb["lower"]) if v is not None), None)
                if u is not None and lo is not None:
                    width = (u - lo) / mid * 100 if mid else 0
                    pos = (closes[-1] - lo) / (u - lo) * 100 if (u - lo) > 0 else 50
                    lines.append(f"Bollinger(20,2): upper={u:.2f} mid={mid:.2f} lower={lo:.2f}")
                    lines.append(f"  width={width:.1f}% | price at {pos:.0f}% (0=lower, 100=upper)")

            if "atr" in requested:
                atr_vals = _atr(highs, lows, closes, 14)
                latest = next((v for v in reversed(atr_vals) if v is not None), None)
                if latest is not None:
                    atr_pct = latest / closes[-1] * 100
                    lines.append(f"ATR(14): {latest:.2f} ({atr_pct:.2f}% of price)")

            return "\n".join(lines)
        except Exception as e:
            return f"Calculate error: {e}"


# ── Trading tools ────────────────────────────────────────────


class BybitTradeTool(Tool):
    """Place a spot or futures order on Bybit with optional SL/TP.

    During simulation phase (first 10 turns), orders are paper-traded:
    logged with current market price but NOT submitted to Bybit.
    """

    def __init__(self, client, state: KevinState):
        self._client = client
        self._state = state

    @property
    def name(self) -> str:
        return "trade"

    @property
    def description(self) -> str:
        return (
            "Place a trade on Bybit (spot or linear perpetual futures). "
            "For spot Market Buy, qty is in USDT. For linear, qty is in base currency (e.g. 0.001 BTC). "
            "Supports stop_loss and take_profit (Bybit-managed, not mental). "
            "For linear orders, set leverage first via the leverage parameter. "
            "During simulation phase (first 10 turns), trades are paper-traded — "
            "logged at market price but NOT submitted to the exchange."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair, e.g. BTCUSDT, ETHUSDT",
                },
                "side": {
                    "type": "string",
                    "enum": ["Buy", "Sell"],
                    "description": "Buy or Sell",
                },
                "qty": {
                    "type": "number",
                    "description": "Order quantity (USDT for spot Market Buy, base currency for linear)",
                    "minimum": 0.001,
                },
                "price": {
                    "type": "number",
                    "description": "Limit price. Omit for market order.",
                },
                "order_type": {
                    "type": "string",
                    "enum": ["Market", "Limit"],
                    "description": "Order type (default Market)",
                },
                "category": {
                    "type": "string",
                    "enum": ["spot", "linear"],
                    "description": "spot = spot trading, linear = USDT perpetual futures (default spot)",
                },
                "leverage": {
                    "type": "integer",
                    "description": "Leverage multiplier for linear orders (1-5). Auto-sets before order.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "stop_loss": {
                    "type": "number",
                    "description": "Stop-loss trigger price. Bybit will auto-close at this price.",
                },
                "take_profit": {
                    "type": "number",
                    "description": "Take-profit trigger price. Bybit will auto-close at this price.",
                },
            },
            "required": ["symbol", "side", "qty"],
        }

    async def execute(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        order_type: str = "Market",
        category: str = "spot",
        leverage: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        **kwargs,
    ) -> str:
        if self._state.is_dead():
            return "Cannot trade: balance is zero. Kevin is shut down."

        # ── Simulation mode (first 10 turns → paper trade) ────
        if self._state.is_simulation():
            return await self._simulate_trade(
                symbol, side, qty, price, order_type, category,
                leverage, stop_loss, take_profit,
            )

        # ── Live trading ──────────────────────────────────────
        try:
            # Set leverage for linear orders
            if category == "linear" and leverage:
                try:
                    self._client.set_leverage(symbol, leverage)
                except Exception as e:
                    return f"Failed to set leverage: {e}"

            result = self._client.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                order_type=order_type,
                category=category,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            # Auto-log trade
            self._state.log_trade({
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "order_type": order_type,
                "category": category,
                "leverage": leverage or 1,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_id": result.get("order_id", ""),
                "result": result.get("status", "unknown"),
            })

            # Update balance after trade
            try:
                new_balance = self._client.get_balance("USDT")
                self._state.update_portfolio(new_balance)
            except Exception:
                pass

            parts = [f"Order {side} {qty} {symbol} ({category}/{order_type})"]
            if leverage and category == "linear":
                parts.append(f"{leverage}x leverage")
            if stop_loss:
                parts.append(f"SL={stop_loss}")
            if take_profit:
                parts.append(f"TP={take_profit}")
            parts.append(f"order_id={result.get('order_id', '?')}")
            return " | ".join(parts)

        except Exception as e:
            self._state.log_trade({
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "category": category,
                "result": f"error: {e}",
            })
            return f"Trade failed: {e}"

    async def _simulate_trade(
        self,
        symbol: str, side: str, qty: float,
        price: float | None, order_type: str, category: str,
        leverage: int | None, stop_loss: float | None, take_profit: float | None,
    ) -> str:
        """Paper trade: log at current market price, don't hit exchange."""
        turn = self._state.get_turn_count()
        remaining = KevinState.SIMULATION_TURNS - turn

        try:
            ticker = self._client.get_ticker(symbol)
            sim_price = price if (price and order_type == "Limit") else ticker.get("last_price", 0)
        except Exception:
            sim_price = price or 0

        self._state.log_trade({
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": sim_price,
            "order_type": order_type,
            "category": category,
            "leverage": leverage or 1,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "order_id": f"SIM-{turn}",
            "result": "simulated",
            "simulated": True,
        })

        parts = [f"[SIMULATED] {side} {qty} {symbol} @ {sim_price:.2f} ({category}/{order_type})"]
        if leverage and category == "linear":
            parts.append(f"{leverage}x leverage")
        if stop_loss:
            parts.append(f"SL={stop_loss}")
        if take_profit:
            parts.append(f"TP={take_profit}")
        parts.append(f"Turn {turn + 1}/{KevinState.SIMULATION_TURNS} — {remaining} sim turns left")
        return " | ".join(parts)


class BybitBalanceTool(Tool):
    """Check account balance and holdings on Bybit."""

    def __init__(self, client, state: KevinState):
        self._client = client
        self._state = state

    @property
    def name(self) -> str:
        return "balance"

    @property
    def description(self) -> str:
        return "Check current Bybit account balance, coin holdings, and portfolio P&L."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "coin": {
                    "type": "string",
                    "description": "Specific coin to check (e.g. USDT, BTC). Omit for all.",
                },
            },
        }

    async def execute(self, coin: str | None = None, **kwargs) -> str:
        try:
            if coin:
                bal = self._client.get_balance(coin)
                self._state.update_portfolio(self._client.get_balance("USDT"))
                return f"{coin}: {bal}\n\n{self._state.get_status_summary()}"

            holdings = self._client.get_all_balances()
            self._state.update_portfolio(self._client.get_balance("USDT"))
            lines = ["Holdings:"]
            for h in holdings:
                lines.append(
                    f"  {h['coin']}: {h['balance']:.6f} "
                    f"(avail: {h['available']:.6f}, ~${h['usd_value']:.2f})"
                )
            if not holdings:
                lines.append("  (empty)")
            lines.append("")
            lines.append(self._state.get_status_summary())
            return "\n".join(lines)
        except Exception as e:
            return f"Error checking balance: {e}"


# ── Position tool ─────────────────────────────────────────────


class BybitPositionTool(Tool):
    """Check open futures positions on Bybit."""

    def __init__(self, client):
        self._client = client

    @property
    def name(self) -> str:
        return "positions"

    @property
    def description(self) -> str:
        return (
            "Check open linear perpetual (futures) positions on Bybit: "
            "direction, size, entry price, mark price, unrealized P&L, "
            "liquidation price, leverage, and active SL/TP."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Specific symbol to check (e.g. BTCUSDT). Omit for all.",
                },
            },
        }

    async def execute(self, symbol: str | None = None, **kwargs) -> str:
        try:
            positions = self._client.get_positions(symbol)
            if not positions:
                return "No open futures positions."

            lines = ["Open Futures Positions:"]
            for p in positions:
                pnl = p["unrealized_pnl"]
                pnl_sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"\n  {p['symbol']} {p['side']} {p['size']} @ {p['avg_price']:,.2f}"
                )
                lines.append(
                    f"    Mark: {p['mark_price']:,.2f} | "
                    f"PnL: {pnl_sign}{pnl:.2f} USDT | "
                    f"Lev: {p['leverage']}x"
                )
                lines.append(
                    f"    Liq: {p['liq_price']:,.2f} | "
                    f"Value: {p['position_value']:,.2f} USDT"
                )
                if p.get("stop_loss"):
                    lines.append(f"    SL: {p['stop_loss']}")
                if p.get("take_profit"):
                    lines.append(f"    TP: {p['take_profit']}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching positions: {e}"


# ── Market data tool ──────────────────────────────────────────


class BybitTickerTool(Tool):
    """Get price, K-line, or orderbook data from Bybit."""

    def __init__(self, client):
        self._client = client

    @property
    def name(self) -> str:
        return "ticker"

    @property
    def description(self) -> str:
        return (
            "Get market data for a crypto pair: current price, 24h stats, "
            "and optionally K-line (OHLCV) or orderbook data."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair, e.g. BTCUSDT, ETHUSDT, SOLUSDT",
                },
                "kline": {
                    "type": "boolean",
                    "description": "Include K-line (OHLCV) data (default false)",
                },
                "interval": {
                    "type": "string",
                    "description": "K-line interval: 1,5,15,30,60,240,D,W (default 60 = 1h)",
                },
                "orderbook": {
                    "type": "boolean",
                    "description": "Include top orderbook entries (default false)",
                },
            },
            "required": ["symbol"],
        }

    async def execute(
        self,
        symbol: str,
        kline: bool = False,
        interval: str = "60",
        orderbook: bool = False,
        **kwargs,
    ) -> str:
        try:
            # Always get ticker
            t = self._client.get_ticker(symbol)
            if "error" in t:
                return t["error"]

            lines = [
                f"{t['symbol']}: ${t['last_price']:,.2f}",
                f"  24h: {t['price_24h_pct']:+.2f}% | H: ${t['high_24h']:,.2f} L: ${t['low_24h']:,.2f}",
                f"  Vol: {t['volume_24h']:,.2f} | Bid: ${t['bid']:,.2f} Ask: ${t['ask']:,.2f}",
            ]

            if kline:
                klines = self._client.get_kline(symbol, interval=interval, limit=20)
                lines.append(f"\nK-line ({interval}min, last {len(klines)}):")
                for k in klines[:20]:
                    lines.append(
                        f"  {k['ts']} | O:{k['open']:.2f} H:{k['high']:.2f} "
                        f"L:{k['low']:.2f} C:{k['close']:.2f} V:{k['volume']:.2f}"
                    )

            if orderbook:
                book = self._client.get_orderbook(symbol, limit=5)
                lines.append("\nOrderbook:")
                lines.append("  Asks (sell):")
                for a in reversed(book.get("asks", [])[:5]):
                    lines.append(f"    ${a['price']:,.2f} × {a['qty']}")
                lines.append("  Bids (buy):")
                for b in book.get("bids", [])[:5]:
                    lines.append(f"    ${b['price']:,.2f} × {b['qty']}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching ticker: {e}"


# ── Sentiment tool ────────────────────────────────────────────


class CryptoSentimentTool(Tool):
    """Aggregate crypto market sentiment signals."""

    def __init__(self, signals):
        self._signals = signals

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def description(self) -> str:
        return (
            "Get aggregated crypto market sentiment: "
            "Fear & Greed Index, BTC/ETH funding rates, "
            "and top Polymarket macro event odds (Fed, geopolitics)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        try:
            data = self._signals.get_all()
            lines = []

            # Fear & Greed
            fg = data.get("fear_greed", {})
            if fg.get("value", -1) >= 0:
                lines.append(f"Fear & Greed Index: {fg['value']}/100 ({fg['sentiment']})")
            else:
                lines.append(f"Fear & Greed Index: unavailable ({fg.get('error', '?')})")

            # Funding rates
            rates = data.get("funding_rates", [])
            if rates:
                lines.append("\nFunding Rates:")
                for r in rates:
                    rate_pct = r.get("funding_rate", 0) * 100
                    label = "longs pay" if rate_pct > 0 else "shorts pay"
                    lines.append(f"  {r['symbol']}: {rate_pct:+.4f}% ({label})")

            # Polymarket macro
            macro = data.get("polymarket_macro", [])
            if macro:
                lines.append("\nMacro Signals (Polymarket):")
                for e in macro[:5]:
                    lines.append(f"  {e['title']} (vol: ${e['volume']:,.0f})")
                    for m in e.get("markets", [])[:2]:
                        lines.append(f"    {m['question']}: Yes={m['yes_pct']}% | ends {m['end_date']}")

            return "\n".join(lines) if lines else "No sentiment data available."
        except Exception as e:
            return f"Error fetching sentiment: {e}"


# ── End-of-turn tool (mandatory cognitive loop) ──────────────


class KevinEndTurnTool(Tool):
    """Mandatory end-of-turn checkpoint.

    Kevin MUST call this exactly once at the end of every wakeup.
    It handles: decision logging, memo persistence, and cron scheduling.
    """

    def __init__(self, state: KevinState, cron_service: CronService):
        self._state = state
        self._cron = cron_service

    @property
    def name(self) -> str:
        return "end_turn"

    @property
    def description(self) -> str:
        return (
            "MANDATORY — call exactly once at the end of every wakeup. "
            "Records your cognitive loop (observation → analysis → decision → reflection), "
            "increments your turn counter, and schedules your next wakeup. "
            "Without this call, you won't wake up again."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": "What you saw: market data, news, sentiment, position changes.",
                },
                "analysis": {
                    "type": "string",
                    "description": "How you interpret it: signal verification, logic chain, confluence score.",
                },
                "decision": {
                    "type": "string",
                    "description": "What you did or chose not to do, and why.",
                },
                "reflection": {
                    "type": "string",
                    "description": "What went well/badly, what to repeat, what to improve.",
                },
                "next_wakeup_minutes": {
                    "type": "integer",
                    "description": (
                        "Minutes until next wakeup (default 15 min). "
                        "Short-term trading: 12-30 min (2-5 times/hour). "
                        "Low-volatility or sleeping: up to 60 min."
                    ),
                    "minimum": 5,
                },
                "memo": {
                    "type": "string",
                    "description": "Note to your future self: what to focus on, what to watch, next steps.",
                },
            },
            "required": ["observation", "analysis", "decision", "reflection"],
        }

    async def execute(
        self,
        observation: str,
        analysis: str,
        decision: str,
        reflection: str,
        next_wakeup_minutes: int | None = None,
        memo: str = "",
        **kwargs,
    ) -> str:
        # 1. Increment turn counter
        turn = self._state.increment_turn()
        is_sim = self._state.is_simulation()

        # 2. Log decision to decisions.jsonl
        self._state.log_decision({
            "turn": turn,
            "simulation": is_sim,
            "observation": observation,
            "analysis": analysis,
            "decision": decision,
            "reflection": reflection,
            "memo": memo,
            "next_wakeup_minutes": next_wakeup_minutes,
        })

        # 3. Save memo for next wakeup
        self._state.save_memo(memo if memo else "(no memo)")

        # 4. Schedule next wakeup — default 15 min for short-term trading
        interval = next_wakeup_minutes or 15
        self._replace_kevin_cron(interval, memo)

        phase = f"SIMULATION turn {turn}/{KevinState.SIMULATION_TURNS}" if is_sim else f"LIVE turn {turn}"
        return (
            f"[{phase}] Logged decision + memo. Next wakeup in {interval} min. "
            f"Turn complete — output after this is log only."
        )

    def _replace_kevin_cron(self, interval_minutes: int, memo: str) -> None:
        """Remove all existing kevin cron jobs and create a fresh one."""
        # Remove old kevin jobs
        existing = self._cron.list_jobs(include_disabled=True)
        for job in existing:
            if job.payload.to == "kevin":
                self._cron.remove_job(job.id)
                logger.debug(f"Kevin end_turn: removed old cron job {job.id}")

        # Create new wakeup job
        wakeup_msg = memo.strip() if memo.strip() else "Scheduled wakeup"
        schedule = CronSchedule(kind="every", every_ms=interval_minutes * 60 * 1000)
        job = self._cron.add_job(
            name=f"Kevin wakeup ({interval_minutes}m)",
            schedule=schedule,
            message=f"[Wakeup] {wakeup_msg}",
            deliver=True,
            channel="system",
            to="kevin",
            delete_after_run=False,
        )
        logger.info(f"Kevin end_turn: cron set → {interval_minutes}min (job {job.id})")


# ── Query tool (for Zero / user) ─────────────────────────────


class KevinStatusTool(Tool):
    """Query Kevin's trading status — available to Zero and the user."""

    def __init__(self, state: KevinState | None = None):
        self._state = state

    @property
    def name(self) -> str:
        return "kevin_status"

    @property
    def description(self) -> str:
        return (
            "Check Kevin's (autonomous crypto trading agent) current status: "
            "balance, P&L, recent trades, and activity."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "trades": {
                    "type": "integer",
                    "description": "Number of recent trades to show (default 10)",
                },
            },
        }

    async def execute(self, trades: int = 10, **kwargs) -> str:
        if self._state is None:
            return "Kevin is not enabled. Configure kevin.enabled=true in config.json."

        if self._state.is_dead():
            summary = self._state.get_status_summary()
            return f"Kevin is SHUT DOWN (balance depleted).\n\n{summary}"

        return self._state.get_status_summary()
