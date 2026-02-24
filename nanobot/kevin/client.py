"""Bybit trading client for Kevin.

Wraps pybit V5 Unified Trading API into a clean synchronous interface
that nanobot tools can call.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Heavy deps are optional — only imported when Kevin is enabled.
try:
    from pybit.unified_trading import HTTP

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


class BybitClient:
    """Thin wrapper around Bybit V5 Unified Trading API (spot + linear perpetuals)."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        if not _HAS_DEPS:
            raise ImportError(
                "Kevin requires pybit. Install with: pip install 'nanobot-ai[kevin]'"
            )

        self._session = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
        self._testnet = testnet
        logger.info(f"Kevin Bybit client ready (testnet={testnet})")

    # ── Account ───────────────────────────────────────────────

    def get_balance(self, coin: str = "USDT") -> float:
        """Get available balance for a coin in the unified account."""
        resp = self._session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        self._check(resp)
        for acct in resp["result"]["list"]:
            for c in acct.get("coin", []):
                if c["coin"] == coin:
                    return float(c.get("walletBalance", 0))
        return 0.0

    def get_all_balances(self) -> list[dict[str, Any]]:
        """Get all non-zero coin balances."""
        resp = self._session.get_wallet_balance(accountType="UNIFIED")
        self._check(resp)
        holdings = []
        for acct in resp["result"]["list"]:
            for c in acct.get("coin", []):
                bal = float(c.get("walletBalance", 0) or 0)
                if bal > 0:
                    holdings.append({
                        "coin": c["coin"],
                        "balance": bal,
                        "available": float(c.get("availableToWithdraw", 0) or 0),
                        "usd_value": float(c.get("usdValue", 0) or 0),
                    })
        return holdings

    # ── Market data ───────────────────────────────────────────

    def get_ticker(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        """Get current ticker for a spot symbol."""
        resp = self._session.get_tickers(category="spot", symbol=symbol)
        self._check(resp)
        items = resp["result"]["list"]
        if not items:
            return {"error": f"No data for {symbol}"}
        t = items[0]
        return {
            "symbol": t["symbol"],
            "last_price": float(t.get("lastPrice", 0)),
            "price_24h_pct": float(t.get("price24hPcnt", 0)) * 100,
            "high_24h": float(t.get("highPrice24h", 0)),
            "low_24h": float(t.get("lowPrice24h", 0)),
            "volume_24h": float(t.get("volume24h", 0)),
            "quote_volume_24h": float(t.get("turnover24h", 0)),
            "bid": float(t.get("bid1Price", 0)),
            "ask": float(t.get("ask1Price", 0)),
        }

    def get_kline(
        self, symbol: str, interval: str = "60", limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get K-line (OHLCV) data.

        Intervals: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        """
        resp = self._session.get_kline(
            category="spot", symbol=symbol, interval=interval, limit=limit
        )
        self._check(resp)
        klines = []
        for k in resp["result"]["list"]:
            klines.append({
                "ts": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        return klines

    def get_orderbook(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        """Get order book (bids + asks)."""
        resp = self._session.get_orderbook(category="spot", symbol=symbol, limit=limit)
        self._check(resp)
        book = resp["result"]
        return {
            "symbol": book.get("s", symbol),
            "bids": [{"price": float(b[0]), "qty": float(b[1])} for b in book.get("b", [])],
            "asks": [{"price": float(a[0]), "qty": float(a[1])} for a in book.get("a", [])],
        }

    def get_funding_rate(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        """Get latest funding rate for a perpetual contract (sentiment indicator)."""
        resp = self._session.get_funding_rate_history(
            category="linear", symbol=symbol, limit=1
        )
        self._check(resp)
        items = resp["result"]["list"]
        if not items:
            return {"symbol": symbol, "funding_rate": 0.0, "ts": ""}
        r = items[0]
        return {
            "symbol": r.get("symbol", symbol),
            "funding_rate": float(r.get("fundingRate", 0)),
            "ts": r.get("fundingRateTimestamp", ""),
        }

    # ── Trading ───────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: str | float,
        price: str | float | None = None,
        order_type: str = "Market",
        category: str = "spot",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """Place a spot or linear perpetual order.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT")
            side: "Buy" or "Sell"
            qty: Order quantity (quote currency for spot Market Buy, base for others)
            price: Required for Limit orders
            order_type: "Market" or "Limit"
            category: "spot" or "linear" (perpetual futures)
            stop_loss: Stop-loss trigger price (Bybit-managed)
            take_profit: Take-profit trigger price (Bybit-managed)
        """
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
        }
        if price is not None and order_type == "Limit":
            params["price"] = str(price)
            params["timeInForce"] = "GTC"
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
            params["slOrderType"] = "Market"
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
            params["tpOrderType"] = "Market"

        resp = self._session.place_order(**params)
        self._check(resp)
        result = resp["result"]
        sl_tp = ""
        if stop_loss:
            sl_tp += f" SL={stop_loss}"
        if take_profit:
            sl_tp += f" TP={take_profit}"
        logger.info(
            f"Kevin order: {side} {qty} {symbol} ({category}/{order_type}){sl_tp}"
            f" → {result.get('orderId', '?')}"
        )
        return {
            "order_id": result.get("orderId", ""),
            "order_link_id": result.get("orderLinkId", ""),
            "status": "submitted",
        }

    def set_leverage(
        self, symbol: str, leverage: int, category: str = "linear"
    ) -> dict[str, Any]:
        """Set leverage for a linear perpetual symbol."""
        resp = self._session.set_leverage(
            category=category,
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
        )
        # Bybit returns retCode 110043 if leverage is already set — treat as success
        if resp.get("retCode", -1) not in (0, 110043):
            self._check(resp)
        logger.info(f"Kevin leverage: {symbol} → {leverage}x")
        return {"symbol": symbol, "leverage": leverage}

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open linear perpetual positions."""
        params: dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        resp = self._session.get_positions(**params)
        self._check(resp)
        positions = []
        for p in resp["result"]["list"]:
            size = float(p.get("size", 0) or 0)
            if size == 0:
                continue
            positions.append({
                "symbol": p.get("symbol", ""),
                "side": p.get("side", ""),
                "size": size,
                "avg_price": float(p.get("avgPrice", 0) or 0),
                "mark_price": float(p.get("markPrice", 0) or 0),
                "leverage": p.get("leverage", "1"),
                "unrealized_pnl": float(p.get("unrealisedPnl", 0) or 0),
                "liq_price": float(p.get("liqPrice", 0) or 0),
                "take_profit": p.get("takeProfit", ""),
                "stop_loss": p.get("stopLoss", ""),
                "position_value": float(p.get("positionValue", 0) or 0),
            })
        return positions

    def cancel_order(
        self, symbol: str, order_id: str, category: str = "spot"
    ) -> dict[str, Any]:
        """Cancel an open order."""
        resp = self._session.cancel_order(
            category=category, symbol=symbol, orderId=order_id
        )
        self._check(resp)
        return {"order_id": order_id, "status": "cancelled"}

    def get_open_orders(
        self, symbol: str | None = None, category: str = "spot"
    ) -> list[dict[str, Any]]:
        """Get open orders."""
        params: dict[str, Any] = {"category": category}
        if symbol:
            params["symbol"] = symbol
        resp = self._session.get_open_orders(**params)
        self._check(resp)
        orders = []
        for o in resp["result"]["list"]:
            orders.append({
                "order_id": o.get("orderId", ""),
                "symbol": o.get("symbol", ""),
                "side": o.get("side", ""),
                "price": o.get("price", ""),
                "qty": o.get("qty", ""),
                "order_type": o.get("orderType", ""),
                "status": o.get("orderStatus", ""),
                "created": o.get("createdTime", ""),
            })
        return orders

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _check(resp: dict) -> None:
        """Raise on Bybit API error."""
        code = resp.get("retCode", -1)
        if code != 0:
            msg = resp.get("retMsg", "unknown error")
            raise RuntimeError(f"Bybit API error {code}: {msg}")
