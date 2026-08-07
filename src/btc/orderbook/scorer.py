"""
scorer.py
---------
Order-book conviction scorer.

Computes a conviction multiplier in [penalty_floor, boost_ceiling] (default
[0.5, 1.10]) from a multi-exchange order-book snapshot. The multiplier is
applied to the calendar-adjusted confluence strength in the main BTC pipeline.

Design (per config/orderbook.yaml):
    1. Filter for WALLS — single price levels with USD notional value
       >= wall_threshold_usd (default $100,000). Both bid walls (buy
       side) and ask walls (sell side) count.
    2. Per-exchange imbalance — (sum_bid_usd - sum_ask_usd) within ±2%
       of spot. Range: [-1, +1].
    3. Wall direction — dominant side (bid-heavy vs ask-heavy) per
       exchange, considering only wall-sized levels.
    4. Cross-exchange agreement — fraction of exchanges whose wall
       direction matches the majority. Below min_agreement_fraction
       (default 0.5), no signal is emitted (multiplier = 1.0).
    5. Multiplier — scales linearly with agreement * weighted imbalance:
         - matches wave direction → boost (up to boost_ceiling)
         - contradicts wave direction → penalty (down to penalty_floor)
         - neutral → 1.0

Public API:
    OrderBookConvictionScorer(config_path="config/orderbook.yaml")
    scorer.compute(snapshots, wave_direction, confluence_zone=None)
        -> ConvictionReport
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.btc.orderbook.fetcher import OrderBookSnapshot, OrderBookLevel


# ─────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────

@dataclass
class WallReport:
    """One wall (>= wall_threshold_usd) detected on one exchange."""
    exchange: str
    side: str                 # 'bid' (buy) or 'ask' (sell)
    price: float
    qty: float
    usd_value: float
    distance_pct: float       # distance from spot in %


@dataclass
class ExchangeSignal:
    """Aggregated per-exchange signal."""
    exchange: str
    spot_price: float
    fetch_error: Optional[str]
    bid_usd_in_band: float     # sum of bid USD within ±imbalance_band_pct of spot
    ask_usd_in_band: float     # sum of ask USD within ±imbalance_band_pct of spot
    imbalance: float           # (bid - ask) / (bid + ask) in band, [-1, +1]
    bid_walls: List[WallReport] = field(default_factory=list)
    ask_walls: List[WallReport] = field(default_factory=list)
    dominant_wall_side: Optional[str] = None    # 'bid' / 'ask' / None
    dominant_wall_usd: float = 0.0


@dataclass
class ConvictionReport:
    """Final conviction report consumed by the BTC pipeline."""
    multiplier: float                       # value to multiply strength by
    direction: str                          # 'boost' / 'penalty' / 'neutral'
    wave_direction: str                     # 'bullish' / 'bearish' / 'neutral'
    agreement_score: float                  # 0.0 - 1.0
    weighted_imbalance: float               # weighted avg imbalance across exchanges
    dominant_side: Optional[str]            # majority wall side across exchanges
    exchanges_succeeded: int
    exchanges_failed: int
    failed_exchanges: List[str]
    bid_walls: List[WallReport]             # all walls >= threshold across exchanges
    ask_walls: List[WallReport]
    per_exchange: List[ExchangeSignal]
    flag_string: str                        # one-line human-readable summary
    neutral_reason: Optional[str] = None    # set when multiplier == 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ─────────────────────────────────────────────────────────────
# Config loader (matches fetcher.py pattern)
# ─────────────────────────────────────────────────────────────

def _load_yaml(relative_path: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent.parent.parent
    full = root / relative_path
    if not full.exists():
        raise FileNotFoundError(f"Config not found: {full}")
    with open(full, "r") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────

class OrderBookConvictionScorer:
    """
    Computes the order-book conviction multiplier.

    Stateless across calls — load config once, call compute() many times.
    """

    def __init__(self, config_path: str = "config/orderbook.yaml"):
        self.config_path = config_path
        cfg = _load_yaml(config_path)
        self.wall_threshold_usd = float(cfg.get("wall_threshold_usd", 1_000_000))
        self.imbalance_band_pct = float(cfg.get("imbalance_band_pct", 2.0))
        mp = cfg.get("multiplier", {})
        self.boost_ceiling = float(mp.get("boost_ceiling", 1.10))
        self.neutral_mult = float(mp.get("neutral", 1.00))
        self.penalty_floor = float(mp.get("penalty_floor", 0.50))
        self.min_agreement_fraction = float(mp.get("min_agreement_fraction", 0.5))
        self.min_exchanges_for_signal = int(cfg.get("min_exchanges_for_signal", 2))
        net = cfg.get("network", {})
        self.fail_safe_neutral = bool(net.get("fail_safe_neutral", True))

    # ── Public API ───────────────────────────────────────────

    def compute(
        self,
        snapshots: Dict[str, OrderBookSnapshot],
        wave_direction: str,
        confluence_zone: Optional[Tuple[float, float]] = None,
    ) -> ConvictionReport:
        """
        Compute the conviction multiplier.

        Args:
            snapshots: dict of {exchange_name: OrderBookSnapshot}, typically
                the return value of fetch_multi_exchange_orderbook().
            wave_direction: 'bullish' / 'bearish' / 'neutral' — the wave-
                projected bias. The multiplier boosts strength when OB
                supports this direction, penalises when it contradicts.
            confluence_zone: optional (lower, upper) price tuple for the
                Fibonacci cluster. Walls inside this zone get extra weight
                in the flag string (but do not change the multiplier math,
                which already considers all wall-sized levels).

        Returns: ConvictionReport
        """
        wave_direction = (wave_direction or "neutral").lower()
        per_exchange = [self._analyse_one(name, snap) for name, snap in snapshots.items()]

        succeeded = [s for s in per_exchange if not s.fetch_error]
        failed = [s for s in per_exchange if s.fetch_error]

        # Fail-safe: if NO exchange returned data, return neutral
        if not succeeded:
            return self._neutral_report(
                wave_direction, per_exchange,
                reason=f"all {len(failed)} exchanges failed",
            )

        # If fail_safe_neutral is set and we have fewer than min_exchanges_for_signal
        # successful exchanges, return neutral
        if self.fail_safe_neutral and len(succeeded) < self.min_exchanges_for_signal:
            return self._neutral_report(
                wave_direction, per_exchange,
                reason=f"only {len(succeeded)}/{len(snapshots)} exchanges succeeded (min={self.min_exchanges_for_signal})",
            )

        # ── Wall direction majority vote ──
        # Each exchange votes 'bid' or 'ask' based on which side has more wall USD
        votes = []
        for s in succeeded:
            if s.dominant_wall_side:
                votes.append(s.dominant_wall_side)
        if not votes:
            # No walls >= threshold on any exchange
            return self._neutral_report(
                wave_direction, per_exchange,
                reason=f"no walls >= ${self.wall_threshold_usd:,.0f} detected on any exchange",
            )

        bid_votes = sum(1 for v in votes if v == "bid")
        ask_votes = sum(1 for v in votes if v == "ask")
        if bid_votes >= ask_votes:
            dominant_side = "bid"
            dominant_count = bid_votes
        else:
            dominant_side = "ask"
            dominant_count = ask_votes

        # Agreement is computed over ALL successful exchanges, not just those
        # that voted. This ensures a wall on only 1/3 exchanges yields
        # agreement = 1/3 = 0.33 (correctly below the spoofing threshold)
        # rather than 1/1 = 1.0.
        agreement_score = dominant_count / len(succeeded) if succeeded else 0.0

        # Below minimum agreement fraction → no signal (spoofing defence)
        if agreement_score < self.min_agreement_fraction:
            return self._neutral_report(
                wave_direction, per_exchange,
                reason=f"cross-exchange agreement {agreement_score:.2f} < {self.min_agreement_fraction} (possible spoofing)",
                dominant_side=dominant_side,
            )

        # ── Weighted imbalance ──
        # Weight each exchange's imbalance by its wall USD (more wall liquidity = more weight)
        total_weight = 0.0
        weighted_sum = 0.0
        for s in succeeded:
            wall_usd = s.dominant_wall_usd or 1.0  # avoid div-by-zero
            w = wall_usd
            weighted_sum += s.imbalance * w
            total_weight += w
        weighted_imbalance = weighted_sum / total_weight if total_weight > 0 else 0.0

        # ── Direction match → boost or penalty ──
        # bid-side dominant + bullish wave → boost
        # ask-side dominant + bearish wave → boost
        # bid-side dominant + bearish wave → penalty
        # ask-side dominant + bullish wave → penalty
        wave_supports_bid = wave_direction == "bullish"
        wave_supports_ask = wave_direction == "bearish"
        directional_match = (
            (dominant_side == "bid" and wave_supports_bid) or
            (dominant_side == "ask" and wave_supports_ask)
        )

        magnitude = abs(weighted_imbalance) * agreement_score  # [0, 1]
        if wave_direction == "neutral":
            multiplier = self.neutral_mult
            direction_label = "neutral"
        elif directional_match:
            # Boost path: 1.0 + (boost_ceiling - 1.0) * magnitude
            multiplier = self.neutral_mult + (self.boost_ceiling - self.neutral_mult) * magnitude
            multiplier = min(multiplier, self.boost_ceiling)
            direction_label = "boost"
        else:
            # Penalty path: 1.0 - (1.0 - penalty_floor) * magnitude
            multiplier = self.neutral_mult - (self.neutral_mult - self.penalty_floor) * magnitude
            multiplier = max(multiplier, self.penalty_floor)
            direction_label = "penalty"

        # ── Aggregate all walls (for storage + alerting) ──
        all_bid_walls = [w for s in succeeded for w in s.bid_walls]
        all_ask_walls = [w for s in succeeded for w in s.ask_walls]
        all_bid_walls.sort(key=lambda w: w.usd_value, reverse=True)
        all_ask_walls.sort(key=lambda w: w.usd_value, reverse=True)

        flag_string = self._build_flag_string(
            direction_label, multiplier, dominant_side, agreement_score,
            len(all_bid_walls), len(all_ask_walls), wave_direction,
            all_bid_walls, all_ask_walls, confluence_zone,
        )

        return ConvictionReport(
            multiplier=round(multiplier, 4),
            direction=direction_label,
            wave_direction=wave_direction,
            agreement_score=round(agreement_score, 4),
            weighted_imbalance=round(weighted_imbalance, 4),
            dominant_side=dominant_side,
            exchanges_succeeded=len(succeeded),
            exchanges_failed=len(failed),
            failed_exchanges=[s.exchange for s in failed],
            bid_walls=all_bid_walls,
            ask_walls=all_ask_walls,
            per_exchange=per_exchange,
            flag_string=flag_string,
            neutral_reason=None,
        )

    # ── Private helpers ──────────────────────────────────────

    def _analyse_one(self, name: str, snap: OrderBookSnapshot) -> ExchangeSignal:
        """Per-exchange signal extraction."""
        if snap.fetch_error:
            return ExchangeSignal(
                exchange=name, spot_price=snap.spot_price,
                fetch_error=snap.fetch_error,
                bid_usd_in_band=0.0, ask_usd_in_band=0.0, imbalance=0.0,
            )
        if not snap.bids and not snap.asks:
            return ExchangeSignal(
                exchange=name, spot_price=snap.spot_price,
                fetch_error="empty order book",
                bid_usd_in_band=0.0, ask_usd_in_band=0.0, imbalance=0.0,
            )

        spot = snap.spot_price
        band_lo = spot * (1.0 - self.imbalance_band_pct / 100.0)
        band_hi = spot * (1.0 + self.imbalance_band_pct / 100.0)

        bid_usd_band = sum(lv.usd_value for lv in snap.bids if band_lo <= lv.price <= band_hi)
        ask_usd_band = sum(lv.usd_value for lv in snap.asks if band_lo <= lv.price <= band_hi)
        denom = bid_usd_band + ask_usd_band
        imbalance = (bid_usd_band - ask_usd_band) / denom if denom > 0 else 0.0

        # Walls — levels with USD notional >= threshold
        bid_walls = [
            WallReport(
                exchange=name, side="bid", price=lv.price, qty=lv.qty,
                usd_value=lv.usd_value,
                distance_pct=(spot - lv.price) / spot * 100.0 if spot > 0 else 0.0,
            )
            for lv in snap.bids if lv.usd_value >= self.wall_threshold_usd
        ]
        ask_walls = [
            WallReport(
                exchange=name, side="ask", price=lv.price, qty=lv.qty,
                usd_value=lv.usd_value,
                distance_pct=(lv.price - spot) / spot * 100.0 if spot > 0 else 0.0,
            )
            for lv in snap.asks if lv.usd_value >= self.wall_threshold_usd
        ]

        bid_wall_usd = sum(w.usd_value for w in bid_walls)
        ask_wall_usd = sum(w.usd_value for w in ask_walls)
        if bid_wall_usd > ask_wall_usd and bid_wall_usd > 0:
            dominant_side = "bid"
            dominant_usd = bid_wall_usd
        elif ask_wall_usd > bid_wall_usd and ask_wall_usd > 0:
            dominant_side = "ask"
            dominant_usd = ask_wall_usd
        else:
            dominant_side = None
            dominant_usd = max(bid_wall_usd, ask_wall_usd)

        return ExchangeSignal(
            exchange=name, spot_price=spot, fetch_error=None,
            bid_usd_in_band=bid_usd_band, ask_usd_in_band=ask_usd_band,
            imbalance=imbalance,
            bid_walls=bid_walls, ask_walls=ask_walls,
            dominant_wall_side=dominant_side,
            dominant_wall_usd=dominant_usd,
        )

    def _neutral_report(
        self,
        wave_direction: str,
        per_exchange: List[ExchangeSignal],
        reason: str,
        dominant_side: Optional[str] = None,
    ) -> ConvictionReport:
        succeeded = [s for s in per_exchange if not s.fetch_error]
        failed = [s.exchange for s in per_exchange if s.fetch_error]
        all_bid_walls = [w for s in succeeded for w in s.bid_walls]
        all_ask_walls = [w for s in succeeded for w in s.ask_walls]

        flag = f"Order book: NEUTRAL ({reason})"
        return ConvictionReport(
            multiplier=self.neutral_mult,
            direction="neutral",
            wave_direction=wave_direction,
            agreement_score=0.0,
            weighted_imbalance=0.0,
            dominant_side=dominant_side,
            exchanges_succeeded=len(succeeded),
            exchanges_failed=len(failed),
            failed_exchanges=failed,
            bid_walls=all_bid_walls,
            ask_walls=all_ask_walls,
            per_exchange=per_exchange,
            flag_string=flag,
            neutral_reason=reason,
        )

    def _build_flag_string(
        self,
        direction_label: str,
        multiplier: float,
        dominant_side: Optional[str],
        agreement_score: float,
        n_bid_walls: int,
        n_ask_walls: int,
        wave_direction: str,
        bid_walls: List[WallReport],
        ask_walls: List[WallReport],
        confluence_zone: Optional[Tuple[float, float]],
    ) -> str:
        """One-line summary for the Telegram alert."""
        if direction_label == "boost":
            emoji = "🟦" if dominant_side == "bid" else "🟥"
            verb = "supports"
        elif direction_label == "penalty":
            emoji = "🟥" if dominant_side == "bid" else "🟦"
            verb = "contradicts"
        else:
            emoji = "⬜"
            verb = "neutral"

        side_word = "BID" if dominant_side == "bid" else "ASK" if dominant_side == "ask" else "—"

        # Largest wall for context
        if dominant_side == "bid" and bid_walls:
            top = bid_walls[0]
            wall_desc = f"top bid ${top.usd_value/1e6:.2f}M @ ${top.price:,.0f}"
        elif dominant_side == "ask" and ask_walls:
            top = ask_walls[0]
            wall_desc = f"top ask ${top.usd_value/1e6:.2f}M @ ${top.price:,.0f}"
        else:
            wall_desc = f"no walls >= ${self.wall_threshold_usd:,.0f}"

        # Note walls inside confluence zone
        zone_note = ""
        if confluence_zone:
            lo, hi = confluence_zone
            zone_bids = [w for w in bid_walls if lo <= w.price <= hi]
            zone_asks = [w for w in ask_walls if lo <= w.price <= hi]
            if zone_bids or zone_asks:
                zone_note = f" | {len(zone_bids)} bid + {len(zone_asks)} ask walls IN ZONE"

        return (
            f"{emoji} Order book {verb} {wave_direction} wave "
            f"({side_word}, {n_bid_walls}B/{n_ask_walls}A walls, "
            f"agree={agreement_score:.2f}, ×{multiplier:.2f}) "
            f"[{wall_desc}{zone_note}]"
        )


# ─────────────────────────────────────────────────────────────
# Convenience: score from a stored snapshot dict
# ─────────────────────────────────────────────────────────────

def score_from_snapshot_dict(
    snapshot_dict: Dict[str, Any],
    wave_direction: str,
    confluence_zone: Optional[Tuple[float, float]] = None,
    config_path: str = "config/orderbook.yaml",
) -> ConvictionReport:
    """
    Reconstruct OrderBookSnapshot objects from a JSON-serialised snapshot
    (e.g. loaded via snapshot.load_latest_snapshot()) and score them.
    """
    snaps: Dict[str, OrderBookSnapshot] = {}
    for name, raw in (snapshot_dict.get("exchanges") or {}).items():
        if not isinstance(raw, dict):
            continue
        if raw.get("fetch_error"):
            snaps[name] = OrderBookSnapshot(
                exchange=name, symbol=raw.get("symbol", ""),
                timestamp_ms=raw.get("timestamp_ms", 0),
                spot_price=raw.get("spot_price", 0.0),
                best_bid=raw.get("best_bid", 0.0),
                best_ask=raw.get("best_ask", 0.0),
                fetch_error=raw["fetch_error"],
            )
            continue
        bids = [
            OrderBookLevel(price=float(b.get("price", 0)), qty=float(b.get("qty", 0)),
                           usd_value=float(b.get("usd_value", 0)))
            for b in (raw.get("bids") or [])
        ]
        asks = [
            OrderBookLevel(price=float(a.get("price", 0)), qty=float(a.get("qty", 0)),
                           usd_value=float(a.get("usd_value", 0)))
            for a in (raw.get("asks") or [])
        ]
        snaps[name] = OrderBookSnapshot(
            exchange=name, symbol=raw.get("symbol", ""),
            timestamp_ms=raw.get("timestamp_ms", 0),
            spot_price=float(raw.get("spot_price", 0.0)),
            best_bid=float(raw.get("best_bid", 0.0)),
            best_ask=float(raw.get("best_ask", 0.0)),
            bids=bids, asks=asks,
        )
    scorer = OrderBookConvictionScorer(config_path=config_path)
    return scorer.compute(snaps, wave_direction=wave_direction, confluence_zone=confluence_zone)
