"""
Unit tests for src.btc.orderbook.scorer — order-book conviction computation.

Covers:
  - $1M wall threshold filter (smaller orders ignored)
  - Cross-exchange agreement requirement (spoofing defence)
  - Direction match: bid walls + bullish → boost; ask walls + bullish → penalty
  - Multiplier bounds [0.5, 1.10]
  - Fail-safe neutral when all exchanges fail
  - Snapshot dict round-trip via score_from_snapshot_dict
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.btc.orderbook.fetcher import OrderBookSnapshot, OrderBookLevel
from src.btc.orderbook.scorer import (
    OrderBookConvictionScorer,
    ConvictionReport,
    score_from_snapshot_dict,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_snap(
    name: str,
    spot: float = 60_000.0,
    bid_walls_usd: list[float] | None = None,
    ask_walls_usd: list[float] | None = None,
    fetch_error: str | None = None,
) -> OrderBookSnapshot:
    """Build a synthetic snapshot with the specified walls (USD notional)."""
    if fetch_error:
        return OrderBookSnapshot(
            exchange=name, symbol="BTC/USDT", timestamp_ms=0,
            spot_price=0, best_bid=0, best_ask=0, fetch_error=fetch_error,
        )

    bids, asks = [], []
    for usd in (bid_walls_usd or []):
        price = spot * 0.998  # 0.2% below spot
        qty = usd / price
        bids.append(OrderBookLevel(price=price, qty=qty, usd_value=usd))
    for usd in (ask_walls_usd or []):
        price = spot * 1.002
        qty = usd / price
        asks.append(OrderBookLevel(price=price, qty=qty, usd_value=usd))
    # Add small noise so imbalance has a non-zero denominator
    for i in range(1, 10):
        p = spot * (1 - 0.001 * i)
        bids.append(OrderBookLevel(price=p, qty=0.5, usd_value=p * 0.5))
        p = spot * (1 + 0.001 * i)
        asks.append(OrderBookLevel(price=p, qty=0.5, usd_value=p * 0.5))
    return OrderBookSnapshot(
        exchange=name, symbol="BTC/USDT",
        timestamp_ms=0, spot_price=spot,
        best_bid=bids[0].price if bids else spot,
        best_ask=asks[0].price if asks else spot,
        bids=bids, asks=asks,
    )


@pytest.fixture
def scorer() -> OrderBookConvictionScorer:
    return OrderBookConvictionScorer()


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestWallThreshold:
    """Verify the $1M wall threshold filter."""

    def test_walls_below_1m_are_ignored(self, scorer):
        """3 exchanges all with $900k bid walls — should be neutral (below $1M threshold)."""
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[900_000]),
            "coinbase": make_snap("coinbase", bid_walls_usd=[900_000]),
            "okx":      make_snap("okx",      bid_walls_usd=[900_000]),
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert r.multiplier == 1.0
        assert r.direction == "neutral"
        assert r.neutral_reason is not None
        assert "no walls" in r.neutral_reason.lower()
        assert len(r.bid_walls) == 0

    def test_walls_at_exactly_1m_count(self, scorer):
        """A wall at exactly $1,000,000 should count (>= threshold)."""
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[1_000_000]),
            "coinbase": make_snap("coinbase", bid_walls_usd=[1_000_000]),
            "okx":      make_snap("okx",      bid_walls_usd=[1_000_000]),
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert len(r.bid_walls) == 3
        assert r.direction == "boost"
        assert r.multiplier > 1.0


class TestSpoofingDefence:
    """Verify the cross-exchange agreement requirement."""

    def test_single_exchange_wall_is_neutral(self, scorer):
        """A wall on only 1/3 exchanges should NOT trigger a signal (agreement 0.33 < 0.5)."""
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase"),  # no walls
            "okx":      make_snap("okx"),        # no walls
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert r.multiplier == 1.0
        assert r.direction == "neutral"
        assert "agreement" in (r.neutral_reason or "").lower()

    def test_two_of_three_agree_triggers_signal(self, scorer):
        """2/3 exchanges with bid walls → agreement 0.67 ≥ 0.5 → signal fires."""
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", bid_walls_usd=[3_000_000]),
            "okx":      make_snap("okx"),  # dissenter
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert r.direction == "boost"
        assert r.multiplier > 1.0
        assert abs(r.agreement_score - 2/3) < 0.01


class TestDirectionMatch:
    """Verify boost/penalty logic against wave direction."""

    def test_bid_walls_bullish_boosts(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", bid_walls_usd=[3_000_000]),
            "okx":      make_snap("okx",      bid_walls_usd=[2_000_000]),
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert r.direction == "boost"
        assert r.multiplier > 1.0
        assert r.multiplier <= 1.10  # boost ceiling

    def test_ask_walls_bullish_penalises(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  ask_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", ask_walls_usd=[3_000_000]),
            "okx":      make_snap("okx",      ask_walls_usd=[2_000_000]),
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert r.direction == "penalty"
        assert r.multiplier < 1.0
        assert r.multiplier >= 0.50  # penalty floor

    def test_bid_walls_bearish_penalises(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", bid_walls_usd=[3_000_000]),
            "okx":      make_snap("okx",      bid_walls_usd=[2_000_000]),
        }
        r = scorer.compute(snaps, wave_direction="bearish")
        assert r.direction == "penalty"
        assert r.multiplier < 1.0

    def test_ask_walls_bearish_boosts(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  ask_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", ask_walls_usd=[3_000_000]),
            "okx":      make_snap("okx",      ask_walls_usd=[2_000_000]),
        }
        r = scorer.compute(snaps, wave_direction="bearish")
        assert r.direction == "boost"
        assert r.multiplier > 1.0


class TestMultiplierBounds:
    """Verify multiplier stays within [0.5, 1.10] even at extreme inputs."""

    def test_max_boost_ceiling(self, scorer):
        """Even with massive walls on all 3 exchanges, multiplier ≤ 1.10."""
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[1_000_000_000]),
            "coinbase": make_snap("coinbase", bid_walls_usd=[1_000_000_000]),
            "okx":      make_snap("okx",      bid_walls_usd=[1_000_000_000]),
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert r.multiplier == pytest.approx(1.10, abs=0.001)

    def test_max_penalty_floor(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  ask_walls_usd=[1_000_000_000]),
            "coinbase": make_snap("coinbase", ask_walls_usd=[1_000_000_000]),
            "okx":      make_snap("okx",      ask_walls_usd=[1_000_000_000]),
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert r.multiplier == pytest.approx(0.50, abs=0.001)


class TestFailSafe:
    """Verify fail-safe behaviour on network/exchange failures."""

    def test_all_exchanges_failed_returns_neutral(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  fetch_error="timeout"),
            "coinbase": make_snap("coinbase", fetch_error="timeout"),
            "okx":      make_snap("okx",      fetch_error="timeout"),
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        assert r.multiplier == 1.0
        assert r.direction == "neutral"
        assert r.exchanges_succeeded == 0
        assert r.exchanges_failed == 3
        assert set(r.failed_exchanges) == {"binance", "coinbase", "okx"}

    def test_one_exchange_failed_others_succeed(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", bid_walls_usd=[3_000_000]),
            "okx":      make_snap("okx",      fetch_error="timeout"),
        }
        r = scorer.compute(snaps, wave_direction="bullish")
        # 2/2 succeeded exchanges agree → agreement = 1.0
        assert r.exchanges_succeeded == 2
        assert r.exchanges_failed == 1
        assert "okx" in r.failed_exchanges
        assert r.direction == "boost"
        assert r.multiplier > 1.0


class TestSnapshotRoundTrip:
    """Verify score_from_snapshot_dict reconstructs snapshots correctly."""

    def test_round_trip_preserves_signal(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  bid_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", bid_walls_usd=[3_000_000]),
            "okx":      make_snap("okx",      bid_walls_usd=[2_000_000]),
        }
        # Score with original snapshots
        r1 = scorer.compute(snaps, wave_direction="bullish", confluence_zone=(59500, 60500))

        # Serialise → dict → re-score
        snap_dict = {
            "exchanges": {name: snap.to_dict() for name, snap in snaps.items()}
        }
        r2 = score_from_snapshot_dict(snap_dict, wave_direction="bullish", confluence_zone=(59500, 60500))

        assert r2.multiplier == r1.multiplier
        assert r2.direction == r1.direction
        assert r2.agreement_score == r1.agreement_score
        assert r2.weighted_imbalance == r1.weighted_imbalance
        assert len(r2.bid_walls) == len(r1.bid_walls)


class TestConfluenceZone:
    """Verify walls inside the confluence zone are tracked in the flag string."""

    def test_walls_in_zone_appear_in_flag(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  spot=60_000, bid_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", spot=60_000, bid_walls_usd=[3_000_000]),
            "okx":      make_snap("okx",      spot=60_000, bid_walls_usd=[2_000_000]),
        }
        # Bid walls are at 0.998 * 60000 = 59880 — inside [59500, 60500]
        r = scorer.compute(snaps, wave_direction="bullish", confluence_zone=(59500, 60500))
        assert "IN ZONE" in r.flag_string

    def test_walls_outside_zone_no_zone_note(self, scorer):
        snaps = {
            "binance":  make_snap("binance",  spot=60_000, bid_walls_usd=[5_000_000]),
            "coinbase": make_snap("coinbase", spot=60_000, bid_walls_usd=[3_000_000]),
            "okx":      make_snap("okx",      spot=60_000, bid_walls_usd=[2_000_000]),
        }
        # Walls at 59880, zone is [61000, 62000] — walls not in zone
        r = scorer.compute(snaps, wave_direction="bullish", confluence_zone=(61000, 62000))
        assert "IN ZONE" not in r.flag_string
