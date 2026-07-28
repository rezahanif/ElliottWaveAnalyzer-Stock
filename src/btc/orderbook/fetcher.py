"""
fetcher.py
----------
Multi-exchange L2 order-book fetcher.

Pulls BTC/USDT (BTC/USD on Coinbase) order-book snapshots from Binance,
Coinbase, and OKX via the CCXT sync API. All three exchanges are queried
in parallel via threads to keep latency under ~3 seconds; each exchange
has independent tenacity-based retry so a single flaky endpoint does not
block the others.

Public API:
    fetch_multi_exchange_orderbook(config_path="config/orderbook.yaml")
        -> Dict[str, OrderBookSnapshot]

    fetch_single_exchange(name, symbol, depth, timeout)
        -> Optional[OrderBookSnapshot]

Pattern follows src/btc/ingestion/fetch_ohlcv.py:
    - ccxt sync API (not ccxt.pro async)
    - {'enableRateLimit': True, 'verify': False} construction
    - urllib3 InsecureRequestWarning suppression
"""

from __future__ import annotations

import os
import sys
import urllib3
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Optional imports — guarded
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    warnings.warn("ccxt not installed — order-book fetcher unavailable. Run: pip install ccxt")

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# Suppress unverified HTTPS warnings (matches fetch_ohlcv.py)
try:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# Project root on path — required for config loading from any CWD
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────

@dataclass
class OrderBookLevel:
    """A single price level in the order book."""
    price: float
    qty: float
    usd_value: float  # price * qty


@dataclass
class OrderBookSnapshot:
    """L2 order-book snapshot for one exchange."""
    exchange: str
    symbol: str
    timestamp_ms: int
    spot_price: float       # best bid/ask midpoint
    best_bid: float
    best_ask: float
    bids: List[OrderBookLevel] = field(default_factory=list)  # sorted descending by price
    asks: List[OrderBookLevel] = field(default_factory=list)  # sorted ascending by price
    fetch_error: Optional[str] = None  # set if fetch failed

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ─────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────

def _load_yaml(relative_path: str) -> Dict[str, Any]:
    """Walk up N parents from this file to project root, join relative_path."""
    # This file is at src/btc/orderbook/fetcher.py — 4 parents up = project root
    root = Path(__file__).resolve().parent.parent.parent.parent
    full_path = root / relative_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config not found: {full_path}")
    with open(full_path, "r") as f:
        return yaml.safe_load(f) or {}


def _build_exchange(name: str):
    """Instantiate a CCXT exchange object with consistent options."""
    name_lower = name.lower()
    if not CCXT_AVAILABLE:
        raise RuntimeError("ccxt not installed")
    if name_lower == "binance":
        return ccxt.binance({"enableRateLimit": True, "verify": False})
    elif name_lower == "coinbase":
        # Coinbase Exchange (formerly Pro) — CCXT unified symbol BTC/USDT
        # auto-maps to BTC/USD on this venue.
        return ccxt.coinbase({"enableRateLimit": True, "verify": False})
    elif name_lower == "okx":
        return ccxt.okx({"enableRateLimit": True, "verify": False})
    else:
        raise ValueError(f"Unsupported exchange: {name}")


# ─────────────────────────────────────────────────────────────
# Single-exchange fetch (with retry)
# ─────────────────────────────────────────────────────────────

def _fetch_one(name: str, symbol: str, depth: int, timeout: int) -> OrderBookSnapshot:
    """Fetch one exchange's order book. Raises on failure."""
    exchange = _build_exchange(name)
    exchange.timeout = timeout * 1000  # CCXT uses ms
    raw = exchange.fetch_order_book(symbol, limit=depth)

    bids_raw = raw.get("bids", []) or []
    asks_raw = raw.get("asks", []) or []
    ts_ms = raw.get("timestamp") or int(datetime.now(timezone.utc).timestamp() * 1000)

    # Some exchanges (notably OKX) return 4-element entries [price, qty, 0, 0];
    # we only need the first two. Use index access instead of tuple unpacking
    # to handle variable-length entries defensively.
    bids = []
    for entry in bids_raw:
        try:
            if len(entry) < 2:
                continue
            p, q = float(entry[0]), float(entry[1])
            if q > 0:
                bids.append(OrderBookLevel(price=p, qty=q, usd_value=p * q))
        except (TypeError, ValueError, IndexError):
            continue
    asks = []
    for entry in asks_raw:
        try:
            if len(entry) < 2:
                continue
            p, q = float(entry[0]), float(entry[1])
            if q > 0:
                asks.append(OrderBookLevel(price=p, qty=q, usd_value=p * q))
        except (TypeError, ValueError, IndexError):
            continue

    # Sort defensively (CCXT normally returns them sorted, but don't trust it)
    bids.sort(key=lambda lv: lv.price, reverse=True)
    asks.sort(key=lambda lv: lv.price)

    best_bid = bids[0].price if bids else 0.0
    best_ask = asks[0].price if asks else 0.0
    spot = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else (best_bid or best_ask)

    return OrderBookSnapshot(
        exchange=name,
        symbol=symbol,
        timestamp_ms=ts_ms,
        spot_price=spot,
        best_bid=best_bid,
        best_ask=best_ask,
        bids=bids,
        asks=asks,
    )


def fetch_single_exchange(
    name: str,
    symbol: str,
    depth: int = 100,
    timeout: int = 15,
    retry_attempts: int = 3,
    retry_min_wait: int = 2,
    retry_max_wait: int = 10,
) -> OrderBookSnapshot:
    """
    Fetch one exchange's order book with optional tenacity retry.

    On failure (after retries), returns an OrderBookSnapshot with
    fetch_error set — never raises. This guarantees the multi-exchange
    orchestrator can degrade gracefully when one venue is unreachable.
    """
    if not CCXT_AVAILABLE:
        return OrderBookSnapshot(
            exchange=name, symbol=symbol, timestamp_ms=0,
            spot_price=0.0, best_bid=0.0, best_ask=0.0,
            fetch_error="ccxt not installed",
        )

    def _do_fetch() -> OrderBookSnapshot:
        return _fetch_one(name, symbol, depth, timeout)

    try:
        if TENACITY_AVAILABLE and retry_attempts > 1:
            retried = retry(
                stop=stop_after_attempt(retry_attempts),
                wait=wait_exponential(min=retry_min_wait, max=retry_max_wait),
                reraise=True,
            )(_do_fetch)
            return retried()
        else:
            # Manual retry fallback if tenacity not available
            last_err = None
            for attempt in range(retry_attempts):
                try:
                    return _do_fetch()
                except Exception as e:
                    last_err = e
                    if attempt < retry_attempts - 1:
                        import time
                        time.sleep(min(retry_min_wait * (2 ** attempt), retry_max_wait))
            raise last_err
    except Exception as e:
        return OrderBookSnapshot(
            exchange=name, symbol=symbol,
            timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            spot_price=0.0, best_bid=0.0, best_ask=0.0,
            fetch_error=str(e),
        )


# ─────────────────────────────────────────────────────────────
# Multi-exchange fan-out
# ─────────────────────────────────────────────────────────────

def fetch_multi_exchange_orderbook(
    config_path: str = "config/orderbook.yaml",
) -> Dict[str, OrderBookSnapshot]:
    """
    Fetch order-book snapshots from all enabled exchanges in parallel.

    Returns a dict keyed by exchange name. Failed exchanges are still
    included in the dict, with `fetch_error` set on their snapshot —
    the caller (scorer) decides how to handle partial failures.
    """
    cfg = _load_yaml(config_path)
    net = cfg.get("network", {})
    timeout = int(net.get("timeout_seconds", 15))
    retry_attempts = int(net.get("retry_attempts", 3))
    retry_min = int(net.get("retry_min_wait_seconds", 2))
    retry_max = int(net.get("retry_max_wait_seconds", 10))

    exchanges_cfg = cfg.get("exchanges", {})
    enabled = [(name, ec) for name, ec in exchanges_cfg.items() if ec.get("enabled", True)]

    if not enabled:
        return {}

    results: Dict[str, OrderBookSnapshot] = {}
    with ThreadPoolExecutor(max_workers=len(enabled)) as pool:
        futures = {
            pool.submit(
                fetch_single_exchange,
                name=name,
                symbol=ec.get("symbol", "BTC/USDT"),
                depth=int(ec.get("depth", 100)),
                timeout=timeout,
                retry_attempts=retry_attempts,
                retry_min_wait=retry_min,
                retry_max_wait=retry_max,
            ): name
            for name, ec in enabled
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                # Defensive — fetch_single_exchange should already return
                # an error snapshot, but guard against thread-level failures.
                results[name] = OrderBookSnapshot(
                    exchange=name, symbol="",
                    timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
                    spot_price=0.0, best_bid=0.0, best_ask=0.0,
                    fetch_error=f"thread exception: {e}",
                )

    return results


# ─────────────────────────────────────────────────────────────
# CLI entrypoint — for manual testing
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching BTC order book from Binance / Coinbase / OKX ...\n")
    snaps = fetch_multi_exchange_orderbook()
    for name, snap in snaps.items():
        if snap.fetch_error:
            print(f"  [{name}] FAILED — {snap.fetch_error}")
            continue
        bid_total = sum(lv.usd_value for lv in snap.bids)
        ask_total = sum(lv.usd_value for lv in snap.asks)
        print(f"  [{name}] spot=${snap.spot_price:,.2f}  "
              f"bids={len(snap.bids)} levels (${bid_total:,.0f})  "
              f"asks={len(snap.asks)} levels (${ask_total:,.0f})")
