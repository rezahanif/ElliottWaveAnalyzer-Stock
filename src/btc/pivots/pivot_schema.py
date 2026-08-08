"""
pivot_schema.py
---------------
Defines the PivotPoint dataclass — the single output unit of the ZigZag
pivot detector and the single input unit for every downstream module:

    StructureTokenizer  →  reads PivotPoint.swing_type + metadata
    FibonacciEngine     →  reads PivotPoint.price + timestamp
    CorrectionClassifier→  reads PivotPoint.swing_type + degree
    TFT feature builder →  reads PivotPoint fields as numeric features

This file contains NO logic. It is a data contract only.
Import it wherever a list of pivots is consumed or produced.

Usage:
    from src.btc.pivots.pivot_schema import (
        PivotPoint, SwingType, WaveDegree, PivotLayer, LiveSwingState,
    )
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Enums — constrained vocabularies
# ─────────────────────────────────────────────

class SwingType(str, Enum):
    """
    Whether this pivot is a confirmed swing high or swing low.
    Determined by the ZigZag state machine on reversal confirmation.
    """
    HIGH = "High"
    LOW  = "Low"


class StructureLabel(str, Enum):
    """
    Higher-order structural label assigned AFTER a sequence of pivots
    is evaluated. Represents the relationship between this pivot and
    the prior pivot of the same type.

    Assigned by the ZigZag detector after each confirmed reversal.

    HH  Higher High  — impulse continuation (uptrend)
    HL  Higher Low   — corrective floor (uptrend)
    LH  Lower High   — corrective ceiling (downtrend)
    LL  Lower Low    — impulse continuation (downtrend)
    """
    HH      = "HH"   # Higher High
    HL      = "HL"   # Higher Low
    LH      = "LH"   # Lower High
    LL      = "LL"   # Lower Low
    UNKNOWN = "UNKNOWN"  # not yet evaluated


class WaveDegree(str, Enum):
    """
    Elliott Wave degree hierarchy.
    Assigned during wave labeling, not during pivot detection.

    Timeframe mapping (from v2.0 spec Section 9):
        1W  →  Primary / Cycle degree
        1D  →  Intermediate / Minor degree
        4H  →  Minute degree (timing/invalidation only)
    """
    SUPERCYCLE   = "supercycle"
    CYCLE        = "cycle"
    PRIMARY      = "primary"
    INTERMEDIATE = "intermediate"
    MINOR        = "minor"
    MINUTE       = "minute"
    UNKNOWN      = "unknown"


class PivotLayer(str, Enum):
    """
    Which volatility threshold layer was used to confirm this pivot.
    Determines the structural significance of the pivot.

    MACRO  →  confirmed by wall_street_threshold_pct (20-day ATR × 3)
              represents institutional / major swing pivots
    MICRO  →  confirmed by behavioral_threshold_pct (14-day ATR × 1.5)
              represents sub-wave / internal structure pivots
    """
    MACRO = "macro"   # wall_street_threshold_pct layer
    MICRO = "micro"   # behavioral_threshold_pct layer


# ─────────────────────────────────────────────
# Core dataclass
# ─────────────────────────────────────────────

@dataclass
class PivotPoint:
    """
    A single confirmed swing high or swing low.

    Produced by:   zigzag.py  (ZigZag state machine)
    Consumed by:   StructureTokenizer, FibonacciEngine,
                   CorrectionClassifier, TFT feature builder

    Fields
    ------
    Required (set by ZigZag on confirmation):
        timestamp_ms    int         Unix timestamp in milliseconds (candle open time)
        price           float       Confirmed extreme price (high for High, low for Low)
        swing_type      SwingType   HIGH or LOW
        bar_index       int         Monotonic integer index in the source DataFrame
        layer           PivotLayer  MACRO or MICRO — which threshold confirmed this pivot

    Contextual (set by ZigZag from the confirming candle):
        candle_high     float       Full candle high at the pivot bar
        candle_low      float       Full candle low at the pivot bar
        candle_close    float       Close price at the pivot bar
        volume          float       Volume at the pivot bar (0.0 if unavailable)
        threshold_used  float       Exact threshold % that triggered confirmation

    Structural (set by StructureTokenizer, after ZigZag):
        structure_label StructureLabel  HH / HL / LH / LL / UNKNOWN
        degree          WaveDegree      Wave degree assigned during labeling
        wave_label      str             e.g. "A", "B", "C", "1", "2", "e", "(C)"
                                        Empty string if not yet labeled.

    Diagnostic (optional, for debugging and TFT features):
        swing_magnitude_pct  float   % move from prior pivot to this pivot
        bars_from_prior      int     bar count since prior confirmed pivot
        rsi_at_pivot         float   RSI value at pivot bar (NaN if not computed)
        macd_hist_at_pivot   float   MACD histogram at pivot bar (NaN if not computed)
        divergence_flag      bool    True if price vs RSI/MACD divergence detected
        volume_surge         bool    True if volume > 1.5× 20-period average
        fib_context          float   Nearest key Fib ratio at this pivot (NaN if none)
    """

    # ── Required ──────────────────────────────
    timestamp_ms:   int
    price:          float
    swing_type:     SwingType
    bar_index:      int
    layer:          PivotLayer

    # ── Contextual ────────────────────────────
    candle_high:    float = 0.0
    candle_low:     float = 0.0
    candle_close:   float = 0.0
    volume:         float = 0.0
    threshold_used: float = 0.0   # the exact pct threshold that triggered confirmation

    # ── Structural (filled by StructureTokenizer) ──
    structure_label: StructureLabel = StructureLabel.UNKNOWN
    degree:          WaveDegree     = WaveDegree.UNKNOWN
    wave_label:      str            = ""

    # ── Diagnostic ────────────────────────────
    swing_magnitude_pct: float = 0.0   # % move from prior pivot price to this price
    bars_from_prior:     int   = 0     # bar count since prior confirmed pivot
    rsi_at_pivot:        float = float("nan")
    macd_hist_at_pivot:  float = float("nan")
    divergence_flag:     bool  = False
    volume_surge:        bool  = False
    fib_context:         float = float("nan")  # nearest key Fib ratio (0.382/0.618/etc)

    # ─────────────────────────────────────────
    # Convenience methods
    # ─────────────────────────────────────────

    def is_high(self) -> bool:
        return self.swing_type == SwingType.HIGH

    def is_low(self) -> bool:
        return self.swing_type == SwingType.LOW

    def is_macro(self) -> bool:
        return self.layer == PivotLayer.MACRO

    def is_micro(self) -> bool:
        return self.layer == PivotLayer.MICRO

    def has_divergence(self) -> bool:
        """True if this pivot shows RSI or MACD divergence — key signal for structural tops/bottoms."""
        return self.divergence_flag

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON output or DataFrame construction."""
        return {
            "timestamp_ms":         self.timestamp_ms,
            "price":                self.price,
            "swing_type":           self.swing_type.value,
            "bar_index":            self.bar_index,
            "layer":                self.layer.value,
            "candle_high":          self.candle_high,
            "candle_low":           self.candle_low,
            "candle_close":         self.candle_close,
            "volume":               self.volume,
            "threshold_used":       self.threshold_used,
            "structure_label":      self.structure_label.value,
            "degree":               self.degree.value,
            "wave_label":           self.wave_label,
            "swing_magnitude_pct":  self.swing_magnitude_pct,
            "bars_from_prior":      self.bars_from_prior,
            "rsi_at_pivot":         self.rsi_at_pivot,
            "macd_hist_at_pivot":   self.macd_hist_at_pivot,
            "divergence_flag":      self.divergence_flag,
            "volume_surge":         self.volume_surge,
            "fib_context":          self.fib_context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PivotPoint:
        """Deserialize from a plain dict (e.g. loaded from JSON fixture)."""
        return cls(
            timestamp_ms        = d["timestamp_ms"],
            price               = d["price"],
            swing_type          = SwingType(d["swing_type"]),
            bar_index           = d["bar_index"],
            layer               = PivotLayer(d["layer"]),
            candle_high         = d.get("candle_high", 0.0),
            candle_low          = d.get("candle_low", 0.0),
            candle_close        = d.get("candle_close", 0.0),
            volume              = d.get("volume", 0.0),
            threshold_used      = d.get("threshold_used", 0.0),
            structure_label     = StructureLabel(d.get("structure_label", "UNKNOWN")),
            degree              = WaveDegree(d.get("degree", "unknown")),
            wave_label          = d.get("wave_label", ""),
            swing_magnitude_pct = d.get("swing_magnitude_pct", 0.0),
            bars_from_prior     = d.get("bars_from_prior", 0),
            rsi_at_pivot        = d.get("rsi_at_pivot", float("nan")),
            macd_hist_at_pivot  = d.get("macd_hist_at_pivot", float("nan")),
            divergence_flag     = d.get("divergence_flag", False),
            volume_surge        = d.get("volume_surge", False),
            fib_context         = d.get("fib_context", float("nan")),
        )

    def __repr__(self) -> str:
        label = f" [{self.wave_label}]" if self.wave_label else ""
        struct = f" {self.structure_label.value}" if self.structure_label != StructureLabel.UNKNOWN else ""
        div = " DIV" if self.divergence_flag else ""
        return (
            f"PivotPoint({self.swing_type.value}{label}{struct}{div} "
            f"@ {self.price:,.2f} | bar={self.bar_index} | "
            f"layer={self.layer.value} | Δ={self.swing_magnitude_pct:.2f}%)"
        )


# ─────────────────────────────────────────────
# Live (unconfirmed) swing state
# ─────────────────────────────────────────────

@dataclass
class LiveSwingState:
    """
    Snapshot of a ZigZag state machine's in-progress swing, as of the
    last bar processed — i.e. what hasn't been confirmed as a PivotPoint
    yet, but is being tracked toward confirmation.

    Produced by:   _ZigZagState.live_state()  (zigzag.py)
    Consumed by:   dashboard signal-tension endpoint

    Unlike PivotPoint, which is always settled history, this is
    forward-looking: "how close is the next pivot to confirming, and
    what price would confirm it right now." Read-only introspection over
    state the detector already tracks every bar — computing this never
    changes which pivots get confirmed.

    Fields
    ------
    layer               PivotLayer  MACRO or MICRO
    state               str         "SEEKING_HIGH" or "SEEKING_LOW"
    direction           str         "short" if SEEKING_HIGH (confirming a
                                     top), "long" if SEEKING_LOW (confirming
                                     a bottom) — derived, not stored twice
    extreme_price       float       running high (SEEKING_HIGH) or low
                                     (SEEKING_LOW) since the last pivot
    locked_threshold    float       ratio (e.g. 0.018 for 1.8%), locked at
                                     the last direction change
    trigger_price       float       exact price that would confirm the
                                     pivot right now, given extreme_price
                                     and locked_threshold
    price_progress_pct  float       0-100, how far the current close has
                                     moved from extreme_price toward
                                     trigger_price
    bars_elapsed        int         bars since the last confirmed pivot
                                     on this layer
    bars_required       int         minimum bars required before the next
                                     pivot can confirm (timeframe-dependent)
    bars_ready          bool        bars_elapsed >= bars_required
    last_pivot_*        —           the last confirmed pivot on this layer,
                                     for context (None if no pivot yet)
    """
    layer:               PivotLayer
    state:               str
    direction:           str
    extreme_price:       float
    locked_threshold:    float
    trigger_price:        float
    price_progress_pct:  float
    bars_elapsed:         int
    bars_required:         int
    bars_ready:             bool
    last_pivot_structure_label: Optional[str]   = None
    last_pivot_magnitude_pct:   Optional[float] = None
    last_pivot_timestamp_ms:    Optional[int]   = None

    def to_dict(self) -> dict:
        """Serialize to plain dict, snake_case — matches PivotPoint.to_dict()'s
        convention. Presentation-layer renaming (e.g. camelCase for the API)
        happens at the API boundary, not here."""
        return {
            "layer":              self.layer.value,
            "state":              self.state,
            "direction":          self.direction,
            "extreme_price":      self.extreme_price,
            "locked_threshold":   self.locked_threshold,
            "trigger_price":      self.trigger_price,
            "price_progress_pct": self.price_progress_pct,
            "bars_elapsed":       self.bars_elapsed,
            "bars_required":      self.bars_required,
            "bars_ready":         self.bars_ready,
            "last_pivot": None if self.last_pivot_structure_label is None else {
                "structure_label": self.last_pivot_structure_label,
                "magnitude_pct":   self.last_pivot_magnitude_pct,
                "timestamp_ms":    self.last_pivot_timestamp_ms,
            },
        }


# ─────────────────────────────────────────────
# JSON-serialization helpers
# ─────────────────────────────────────────────

def sanitize_pivot_dict(d: dict) -> dict:
    """
    JSON-safe copy of a PivotPoint.to_dict(): NaN/Inf floats → None.

    Node's JSON.parse rejects bare NaN/Infinity literals, so raw to_dict()
    output (rsi_at_pivot, fib_context, ... default to NaN) would break the
    dashboard /api/pivots route. Use this wherever pivots are serialized
    for the web layer (e.g. run_daily_analysis.dump_pivots).
    """
    return {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            for k, v in d.items()}
