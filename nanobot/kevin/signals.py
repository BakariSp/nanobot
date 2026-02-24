"""Read-only market signal aggregation for Kevin.

Collects sentiment data from multiple sources:
- Crypto Fear & Greed Index (alternative.me)
- Polymarket macro event odds (Gamma API)
- Bybit funding rates (via BybitClient)
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger


class CryptoSignals:
    """Aggregates read-only sentiment data for Kevin's decision-making."""

    def __init__(self, bybit_client: Any | None = None):
        self._http = httpx.Client(timeout=15)
        self._bybit = bybit_client

    # ── Fear & Greed Index ────────────────────────────────────

    def get_fear_greed(self) -> dict[str, Any]:
        """Get Crypto Fear & Greed Index from alternative.me."""
        try:
            resp = self._http.get("https://api.alternative.me/fng/", params={"limit": 1})
            resp.raise_for_status()
            data = resp.json()["data"][0]
            return {
                "value": int(data["value"]),
                "sentiment": data["value_classification"],
                "timestamp": data["timestamp"],
            }
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return {"value": -1, "sentiment": "unavailable", "error": str(e)}

    # ── Polymarket macro signals ──────────────────────────────

    def get_polymarket_macro(self, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch top Polymarket events as macro sentiment indicators.

        These are read-only — Kevin doesn't trade on Polymarket,
        but uses odds (e.g. Fed rate decision, geopolitical events)
        as inputs for crypto trading decisions.
        """
        try:
            resp = self._http.get(
                "https://gamma-api.polymarket.com/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "order": "volume",
                    "ascending": "false",
                },
            )
            resp.raise_for_status()
            events = []
            for e in resp.json():
                title = e.get("title", "?")
                markets = e.get("markets", [])
                top_markets = []
                for m in markets[:3]:
                    import json as _json
                    prices = m.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        prices = _json.loads(prices)
                    yes = float(prices[0]) if prices else 0
                    top_markets.append({
                        "question": m.get("question", ""),
                        "yes_pct": round(yes * 100, 1),
                        "end_date": (m.get("endDate") or "")[:10],
                    })
                events.append({
                    "title": title,
                    "volume": float(e.get("volume", 0)),
                    "markets": top_markets,
                })
            return events
        except Exception as e:
            logger.warning(f"Polymarket macro fetch failed: {e}")
            return []

    # ── Funding rates ─────────────────────────────────────────

    def get_funding_rates(
        self, symbols: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Get funding rates for major perpetuals (sentiment indicator).

        Positive = longs pay shorts (bullish sentiment / overleveraged longs)
        Negative = shorts pay longs (bearish sentiment / overleveraged shorts)
        """
        if not self._bybit:
            return []
        symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        rates = []
        for sym in symbols:
            try:
                r = self._bybit.get_funding_rate(sym)
                rates.append(r)
            except Exception as e:
                logger.warning(f"Funding rate fetch failed for {sym}: {e}")
        return rates

    # ── Aggregate ─────────────────────────────────────────────

    def get_all(self) -> dict[str, Any]:
        """Get all signals in one call."""
        return {
            "fear_greed": self.get_fear_greed(),
            "polymarket_macro": self.get_polymarket_macro(limit=5),
            "funding_rates": self.get_funding_rates(),
        }
