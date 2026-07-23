"""
zigzag.py
---------
Dynamic Volatility ZigZag — converts the enriched OHLCV+threshold JSON
(produced by calculate_layers.py) into a confirmed list of PivotPoint objects.

Design decisions (v2.0 spec):
  - Two independent threshold layers:
      MACRO  (wall_street_threshold_pct)  → institutional swing pivots
      MICRO  (behavioral_threshold_pct)   → internal sub-wave pivots
  - Threshold is LOCKED at the moment a new swing direction begins.
    Rationale: floating threshold mid-swing can whipsaw confirmation.
    The threshold that starts a new direction is the one that must be
    beaten to confirm the prior extreme as a pivot.
  - Minimum bar distance enforced to suppress micro-noise pivots even
    when the percentage threshold is technically met.
  - Uses candle HIGH/LOW (not close) to track running extremes — a pivot
    high's price is the candle high, not the close. Confirmation trigger
    is still checked against close to avoid shadow-only false breaks.
  - HH/HL/LH/LL structure labels assigned here after each confirmation
    (requires only one prior pivot of same type, not a full sequence).
    Full sequence labeling (BOS, CHOCH etc.) is StructureTokenizer's job.
  - Both layers run in a single pass over the same DataFrame.
    Returns a ZigZagResult with two separate pivot lists.

Usage:
    from src.waveconf.pivots.zigzag import ZigZagDetector

    detector = ZigZagDetector(timeframe='4H', min_bars_between_pivots=3)
    result   = detector.run_from_file('data/pivots/BTC_4H_with_layers.json')

    macro_pivots = result.macro   # List[PivotPoint] — institutional swings
    micro_pivots = result.micro   # List[PivotPoint] — sub-wave pivots
    all_pivots   = result.all()   # merged, sorted by bar_index
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from src.waveconf.pivots.pivot_schema import (
    PivotLayer,
    PivotPoint,
    StructureLabel,
    SwingType,
    WaveDegree,
)


# ─────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────

@dataclass
class ZigZagResult:
    """
    Output of ZigZagDetector.run().
    Contains two independent pivot lists — one per threshold layer.
    """
    macro: List[PivotPoint] = field(default_factory=list)
    micro: List[PivotPoint] = field(default_factory=list)
    timeframe: str = ""
    asset: str = ""
    total_bars: int = 0

    def all(self) -> List[PivotPoint]:
        """Merged list sorted by bar_index. Includes duplicates if a pivot
        appears on both layers (same bar, different layer tag)."""
        combined = self.macro + self.micro
        return sorted(combined, key=lambda p: p.bar_index)

    def to_dict_list(self, layer: str = "both") -> list:
        """Serialize to list of dicts for JSON output."""
        if layer == "macro":
            return [p.to_dict() for p in self.macro]
        elif layer == "micro":
            return [p.to_dict() for p in self.micro]
        else:
            return [p.to_dict() for p in self.all()]

    def summary(self) -> str:
        return (
            f"ZigZagResult [{self.asset} {self.timeframe}]\n"
            f"  Total bars  : {self.total_bars}\n"
            f"  Macro pivots: {len(self.macro)} "
            f"({sum(1 for p in self.macro if p.is_high())} highs, "
            f"{sum(1 for p in self.macro if p.is_low())} lows)\n"
            f"  Micro pivots: {len(self.micro)} "
            f"({sum(1 for p in self.micro if p.is_high())} highs, "
            f"{sum(1 for p in self.micro if p.is_low())} lows)"
        )


# ─────────────────────────────────────────────────────────────
# Internal state machine
# ─────────────────────────────────────────────────────────────

class _ZigZagState:
    """
    Single-layer state machine for one threshold column.
    Tracks running extreme price, bar index, and direction state.
    Confirms pivots when a reversal exceeds the locked threshold.
    """

    SEEKING_HIGH = "SEEKING_HIGH"
    SEEKING_LOW  = "SEEKING_LOW"
    INIT         = None

    def __init__(
        self,
        layer: PivotLayer,
        min_bars: int,
        degree: WaveDegree,
    ):
        self.layer      = layer
        self.min_bars   = min_bars          # minimum bars between confirmed pivots
        self.degree     = degree

        self.state: Optional[str] = self.INIT

        # Running extreme tracking
        self.extreme_price: float = 0.0     # highest high (SEEKING_HIGH) or lowest low
        self.extreme_bar:   int   = 0       # bar_index of current extreme
        self.extreme_ts:    int   = 0       # timestamp_ms of current extreme
        self.extreme_high:  float = 0.0     # full candle high at extreme bar
        self.extreme_low:   float = 0.0     # full candle low at extreme bar
        self.extreme_close: float = 0.0     # close at extreme bar
        self.extreme_vol:   float = 0.0     # volume at extreme bar

        # Locked threshold — set when direction changes, held until next pivot
        self.locked_threshold: float = 0.0

        # Last confirmed pivot price (for magnitude calculation)
        self.last_pivot_price: float = 0.0
        self.last_pivot_bar:   int   = -999

        # Confirmed pivots — same type tracking for HH/HL/LH/LL
        self.last_confirmed_high: Optional[float] = None
        self.last_confirmed_low:  Optional[float] = None

        self.pivots: List[PivotPoint] = []

    def _assign_structure_label(self, swing_type: SwingType, price: float) -> StructureLabel:
        """
        Assign HH/HL/LH/LL by comparing this confirmed pivot against the
        last confirmed pivot of the same type.
        """
        if swing_type == SwingType.HIGH:
            prior = self.last_confirmed_high
            if prior is None:
                label = StructureLabel.UNKNOWN
            elif price > prior:
                label = StructureLabel.HH
            else:
                label = StructureLabel.LH
            self.last_confirmed_high = price
        else:
            prior = self.last_confirmed_low
            if prior is None:
                label = StructureLabel.UNKNOWN
            elif price < prior:
                label = StructureLabel.LL
            else:
                label = StructureLabel.HL
            self.last_confirmed_low = price
        return label

    def _confirm_pivot(
        self,
        swing_type:     SwingType,
        trigger_bar:    int,
        trigger_close:  float,
    ) -> PivotPoint:
        """Lock in the running extreme as a confirmed pivot."""
        magnitude = 0.0
        if self.last_pivot_price > 0:
            magnitude = abs(self.extreme_price - self.last_pivot_price) / self.last_pivot_price * 100

        struct_label = self._assign_structure_label(swing_type, self.extreme_price)

        pivot = PivotPoint(
            timestamp_ms        = self.extreme_ts,
            price               = self.extreme_price,
            swing_type          = swing_type,
            bar_index           = self.extreme_bar,
            layer               = self.layer,
            candle_high         = self.extreme_high,
            candle_low          = self.extreme_low,
            candle_close        = self.extreme_close,
            volume              = self.extreme_vol,
            threshold_used      = self.locked_threshold,
            structure_label     = struct_label,
            degree              = self.degree,
            swing_magnitude_pct = round(magnitude, 4),
            bars_from_prior     = self.extreme_bar - self.last_pivot_bar,
        )

        self.last_pivot_price = self.extreme_price
        self.last_pivot_bar   = self.extreme_bar
        return pivot

    def _min_bars_ok(self, current_bar: int) -> bool:
        """Enforce minimum bar distance from last confirmed pivot."""
        return (current_bar - self.last_pivot_bar) >= self.min_bars

    def process_bar(self, row: pd.Series, bar_index: int, threshold_col: str):
        """
        Process one bar. May append 0 or 1 pivot to self.pivots.

        Threshold is read from the row but LOCKED at direction-change time.
        Confirmation check uses close price; extreme tracking uses high/low.
        """
        ts         = int(row["timestamp_ms"])
        high       = float(row["high"])
        low        = float(row["low"])
        close      = float(row["close"])
        volume     = float(row.get("volume", 0.0))
        bar_threshold = float(row[threshold_col]) / 100.0  # convert pct → ratio

        # ── INIT: establish first direction from first meaningful move ──
        if self.state == self.INIT:
            if self.extreme_price == 0.0:
                # Bootstrap: set initial tracking point
                self.extreme_price = close
                self.extreme_bar   = bar_index
                self.extreme_ts    = ts
                self.extreme_high  = high
                self.extreme_low   = low
                self.extreme_close = close
                self.extreme_vol   = volume
                self.last_pivot_price = close
                return

            pct_change = (close - self.extreme_price) / self.extreme_price

            if pct_change >= bar_threshold:
                # First meaningful up-move → look for the high
                self.state            = self.SEEKING_HIGH
                self.locked_threshold = bar_threshold
                self.extreme_price    = high
                self.extreme_bar      = bar_index
                self.extreme_ts       = ts
                self.extreme_high     = high
                self.extreme_low      = low
                self.extreme_close    = close
                self.extreme_vol      = volume

            elif pct_change <= -bar_threshold:
                # First meaningful down-move → look for the low
                self.state            = self.SEEKING_LOW
                self.locked_threshold = bar_threshold
                self.extreme_price    = low
                self.extreme_bar      = bar_index
                self.extreme_ts       = ts
                self.extreme_high     = high
                self.extreme_low      = low
                self.extreme_close    = close
                self.extreme_vol      = volume
            return

        # ── SEEKING_HIGH: track the running peak ──
        if self.state == self.SEEKING_HIGH:

            if high > self.extreme_price:
                # New peak — update running extreme
                self.extreme_price = high
                self.extreme_bar   = bar_index
                self.extreme_ts    = ts
                self.extreme_high  = high
                self.extreme_low   = low
                self.extreme_close = close
                self.extreme_vol   = volume

            # Check reversal: has close dropped by more than the LOCKED threshold?
            pct_from_extreme = (close - self.extreme_price) / self.extreme_price

            if pct_from_extreme <= -self.locked_threshold and self._min_bars_ok(bar_index):
                # Confirm the HIGH
                pivot = self._confirm_pivot(SwingType.HIGH, bar_index, close)
                self.pivots.append(pivot)

                # Switch to SEEKING_LOW
                self.state            = self.SEEKING_LOW
                self.locked_threshold = bar_threshold   # lock new threshold
                self.extreme_price    = low
                self.extreme_bar      = bar_index
                self.extreme_ts       = ts
                self.extreme_high     = high
                self.extreme_low      = low
                self.extreme_close    = close
                self.extreme_vol      = volume

        # ── SEEKING_LOW: track the running trough ──
        elif self.state == self.SEEKING_LOW:

            if low < self.extreme_price:
                # New trough — update running extreme
                self.extreme_price = low
                self.extreme_bar   = bar_index
                self.extreme_ts    = ts
                self.extreme_high  = high
                self.extreme_low   = low
                self.extreme_close = close
                self.extreme_vol   = volume

            # Check reversal: has close bounced by more than the LOCKED threshold?
            pct_from_extreme = (close - self.extreme_price) / self.extreme_price

            if pct_from_extreme >= self.locked_threshold and self._min_bars_ok(bar_index):
                # Confirm the LOW
                pivot = self._confirm_pivot(SwingType.LOW, bar_index, close)
                self.pivots.append(pivot)

                # Switch to SEEKING_HIGH
                self.state            = self.SEEKING_HIGH
                self.locked_threshold = bar_threshold   # lock new threshold
                self.extreme_price    = high
                self.extreme_bar      = bar_index
                self.extreme_ts       = ts
                self.extreme_high     = high
                self.extreme_low      = low
                self.extreme_close    = close
                self.extreme_vol      = volume


# ─────────────────────────────────────────────────────────────
# Public detector class
# ─────────────────────────────────────────────────────────────

class ZigZagDetector:
    """
    Public API for the Dynamic Volatility ZigZag pivot detector.

    Parameters
    ----------
    timeframe : str
        Source timeframe string, e.g. '1D', '4H', '1W'.
        Used for logging/output metadata only — thresholds are
        already calendar-adjusted in the enriched JSON.
    min_bars_between_pivots : int
        Minimum number of bars required between two confirmed pivots.
        Prevents micro-noise pivots during low-volatility patches.
        Recommended values:
            1D  →  3   (3 daily bars minimum)
            4H  →  6   (6 × 4H bars = 1 calendar day minimum)
            1W  →  2   (2 weekly bars minimum)
    """

    # Timeframe → (min_bars, WaveDegree for macro, WaveDegree for micro)
    TIMEFRAME_DEFAULTS = {
        "1W":  (2,  WaveDegree.PRIMARY,      WaveDegree.INTERMEDIATE),
        "1D":  (3,  WaveDegree.INTERMEDIATE, WaveDegree.MINOR),
        "4H":  (6,  WaveDegree.MINOR,        WaveDegree.MINUTE),
        "1H":  (12, WaveDegree.MINUTE,       WaveDegree.UNKNOWN),
    }

    def __init__(
        self,
        timeframe: str = "1D",
        min_bars_between_pivots: Optional[int] = None,
    ):
        self.timeframe = timeframe.upper()
        defaults       = self.TIMEFRAME_DEFAULTS.get(self.timeframe, (3, WaveDegree.UNKNOWN, WaveDegree.UNKNOWN))

        self.min_bars     = min_bars_between_pivots if min_bars_between_pivots is not None else defaults[0]
        self.macro_degree = defaults[1]
        self.micro_degree = defaults[2]

    def run(self, df: pd.DataFrame, asset: str = "BTCUSD") -> ZigZagResult:
        """
        Run both state machines (macro + micro) over the enriched DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns:
                timestamp_ms, open, high, low, close, volume,
                wall_street_threshold_pct, behavioral_threshold_pct
            Rows must be sorted ascending by timestamp_ms.

        asset : str
            Asset identifier for metadata (default 'BTCUSD').

        Returns
        -------
        ZigZagResult with .macro and .micro pivot lists.
        """
        required_cols = {
            "timestamp_ms", "high", "low", "close",
            "wall_street_threshold_pct", "behavioral_threshold_pct",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Enriched DataFrame missing required columns: {missing}\n"
                f"Run calculate_layers.py first to generate the enriched JSON."
            )

        # Sort ascending — state machine requires chronological order
        df = df.sort_values("timestamp_ms").reset_index(drop=True)

        macro_sm = _ZigZagState(PivotLayer.MACRO, self.min_bars, self.macro_degree)
        micro_sm = _ZigZagState(PivotLayer.MICRO, self.min_bars, self.micro_degree)

        for bar_index, row in df.iterrows():
            macro_sm.process_bar(row, bar_index, "wall_street_threshold_pct")
            micro_sm.process_bar(row, bar_index, "behavioral_threshold_pct")

        return ZigZagResult(
            macro      = macro_sm.pivots,
            micro      = micro_sm.pivots,
            timeframe  = self.timeframe,
            asset      = asset,
            total_bars = len(df),
        )

    def run_from_file(self, path: str) -> ZigZagResult:
        """
        Load enriched JSON produced by calculate_layers.py and run detection.

        Parameters
        ----------
        path : str
            Path to enriched JSON, e.g. 'data/pivots/BTC_4H_with_layers.json'

        Returns
        -------
        ZigZagResult
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Enriched layer file not found: {path}\n"
                f"Run calculate_layers.py to generate it first."
            )

        with open(path, "r") as f:
            raw = json.load(f)

        df = pd.DataFrame(raw["data"], columns=raw["columns"])
        asset     = raw.get("asset", "BTCUSD")
        timeframe = raw.get("timeframe", self.timeframe)

        if timeframe != self.timeframe:
            # Auto-adjust min_bars if timeframe in file differs from constructor arg
            defaults      = self.TIMEFRAME_DEFAULTS.get(timeframe.upper(), (3, WaveDegree.UNKNOWN, WaveDegree.UNKNOWN))
            self.timeframe    = timeframe.upper()
            self.min_bars     = defaults[0]
            self.macro_degree = defaults[1]
            self.micro_degree = defaults[2]

        return self.run(df, asset=asset)

    def save_result(self, result: ZigZagResult, output_dir: str = "data/pivots") -> Tuple[str, str]:
        """
        Save macro and micro pivot lists as JSON files.

        Returns
        -------
        Tuple of (macro_path, micro_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        macro_path = os.path.join(output_dir, f"{result.asset}_{result.timeframe}_pivots_macro.json")
        micro_path = os.path.join(output_dir, f"{result.asset}_{result.timeframe}_pivots_micro.json")

        def _write(path, pivots, layer_name):
            out = {
                "asset":       result.asset,
                "timeframe":   result.timeframe,
                "layer":       layer_name,
                "total_bars":  result.total_bars,
                "pivot_count": len(pivots),
                "pivots":      [p.to_dict() for p in pivots],
            }
            with open(path, "w") as f:
                json.dump(out, f, indent=2)

        _write(macro_path, result.macro, "macro")
        _write(micro_path, result.micro, "micro")

        return macro_path, micro_path


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/pivots/BTC_1D_with_layers.json"

    detector = ZigZagDetector()
    result   = detector.run_from_file(path)

    print(result.summary())
    print()

    print("── Last 5 MACRO pivots ──")
    for p in result.macro[-5:]:
        print(" ", repr(p))

    print()
    print("── Last 5 MICRO pivots ──")
    for p in result.micro[-5:]:
        print(" ", repr(p))