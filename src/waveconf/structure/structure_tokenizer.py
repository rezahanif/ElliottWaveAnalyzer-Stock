"""
structure_tokenizer.py
-----------------------
Converts a chronological list of PivotPoint objects (output of zigzag.py)
into a sequence of StructureToken records — the extended 15-token
vocabulary defined in the v2.0 spec, Section 5.

Pipeline position: step [2], between PivotDetector (zigzag.py) and the
ML/classification stages (CorrectionClassifier, ImpulseClassifier, TFT).

Scope decision — what this module DOES and DOES NOT compute
=============================================================
Per the v2.0 pipeline, StructureTokenizer is purely rule-based and only
has access to the pivot sequence itself (+ optional indicator series).
It does NOT have wave-position labels (1-5, A-B-C) yet — those are
assigned downstream by ImpulseClassifier / CorrectionClassifier.

Tokens computed HERE (directly derivable from pivot sequence alone):
    HH, HL, LH, LL   — pass-through from PivotPoint.structure_label
    BOS              — trend-confirming break (continuation)
    CHOCH            — first opposing break (trend-state flip)
    FIB_T            — current pivot's retracement of the prior leg
                        lands near a key Fibonacci ratio
    SWEEP            — fast break of prior same-type extreme,
                        immediately reversed by the next pivot
    DIV_H / DIV_L    — ONLY if rsi_series / macd_hist_series are passed in;
                        otherwise silently skipped (indicators.py is still
                        an empty stub in this repo, so this is currently
                        inactive — wire it up once indicators.py exists)

Tokens explicitly DEFERRED (NOT emitted by this module):
    W3_EXT, W4_REJ   — require explicit 1-2-3-4-5 wave labels
                        (ImpulseClassifier's job)
    DIAG             — requires confirmed leading/ending diagonal wave
                        labels + trendline convergence check
                        (ImpulseClassifier + TrendlineBuilder's job)
    TRUNC            — requires a confirmed wave-5 label compared to
                        wave-3 (ImpulseClassifier's job)
    ABW_T            — requires drawn ABW trendlines
                        (TrendlineBuilder + PatternDetector's job)

Emitting these here would mean guessing wave position from raw pivots,
which contradicts the pipeline's separation of concerns (Section 2).
They are listed in STRUCTURE_TOKENS for vocabulary completeness, but
this module will never emit them. Downstream classifiers append them
to the same token stream once wave labels exist.

Usage:
    from src.waveconf.structure.structure_tokenizer import StructureTokenizer

    tokenizer = StructureTokenizer()
    macro_tokens = tokenizer.run(zigzag_result.macro)
    micro_tokens = tokenizer.run(zigzag_result.micro)

    # both layers independently, mirroring the dual-layer ZigZag design
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from src.waveconf.pivots.pivot_schema import (
    PivotLayer,
    PivotPoint,
    StructureLabel,
    SwingType,
    WaveDegree,
)


# ─────────────────────────────────────────────────────────────
# Token vocabulary (Section 5 of the v2.0 spec)
# ─────────────────────────────────────────────────────────────

STRUCTURE_TOKENS = {
    # Directional structure — standard (pass-through from PivotPoint)
    "HH":     0,
    "HL":     1,
    "LH":     2,
    "LL":     3,

    # Structural events — computed here
    "BOS":    4,
    "CHOCH":  5,

    # Elliott Wave specific extensions — DEFERRED, see module docstring
    "W3_EXT": 6,
    "W4_REJ": 7,
    "DIV_H":  8,   # computed here, but only if indicator series supplied
    "DIV_L":  9,   # computed here, but only if indicator series supplied
    "FIB_T":  10,  # computed here
    "ABW_T":  11,  # DEFERRED
    "SWEEP":  12,  # computed here
    "DIAG":   13,  # DEFERRED
    "TRUNC":  14,  # DEFERRED
}

# Tokens this module is actually capable of emitting today.
EMITTABLE_TOKENS = {"HH", "HL", "LH", "LL", "BOS", "CHOCH", "FIB_T", "SWEEP", "DIV_H", "DIV_L"}

# Key Fibonacci ratios checked for FIB_T tagging (retracement AND extension).
KEY_FIB_RATIOS = [0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.382, 1.618, 2.618]


# ─────────────────────────────────────────────────────────────
# Output record
# ─────────────────────────────────────────────────────────────

@dataclass
class StructureToken:
    """
    One emitted token, anchored to a specific pivot.
    Mirrors the `token_record` dict shown in spec Section 5.
    """
    token:         str
    token_id:      int
    price:         float
    timestamp_ms:  int
    bar_index:     int
    layer:         PivotLayer
    degree:        WaveDegree
    fib_context:   Optional[float] = None   # nearest matched Fib ratio, if any
    volume_surge:  bool = False
    rsi_value:     Optional[float] = None
    macd_hist:     Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "token":        self.token,
            "token_id":     self.token_id,
            "price":        self.price,
            "timestamp_ms": self.timestamp_ms,
            "bar_index":    self.bar_index,
            "layer":        self.layer.value,
            "degree":       self.degree.value,
            "fib_context":  self.fib_context,
            "volume_surge": self.volume_surge,
            "rsi_value":    self.rsi_value,
            "macd_hist":    self.macd_hist,
        }

    def __repr__(self) -> str:
        fib = f" fib={self.fib_context:.3f}" if self.fib_context is not None else ""
        return f"<{self.token} @ {self.price:,.2f} bar={self.bar_index} layer={self.layer.value}{fib}>"


# ─────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────

class StructureTokenizer:
    """
    Parameters
    ----------
    fib_tolerance : float
        Relative tolerance for matching a retracement ratio to a key
        Fibonacci ratio. E.g. 0.03 means ±3% of the ratio value.
    sweep_max_bars : int
        A break of the prior same-type extreme confirmed within this many
        bars of the prior pivot is considered "fast" — a candidate for SWEEP.
    sweep_min_reversal_pct : float
        Minimum % retracement (relative to the swept leg) the following
        pivot must achieve for the break to be tagged SWEEP rather than a
        genuine continuation (HH/LL standing on its own).
    """

    def __init__(
        self,
        fib_tolerance: float = 0.03,
        sweep_max_bars: int = 3,
        sweep_min_reversal_pct: float = 0.5,
    ):
        self.fib_tolerance = fib_tolerance
        self.sweep_max_bars = sweep_max_bars
        self.sweep_min_reversal_pct = sweep_min_reversal_pct

    # ── public API ────────────────────────────────────────────

    def run(
        self,
        pivots: Sequence[PivotPoint],
        rsi_series: Optional[dict] = None,       # {bar_index: rsi_value}
        macd_hist_series: Optional[dict] = None, # {bar_index: macd_hist_value}
    ) -> List[StructureToken]:
        """
        Tokenize ONE layer (macro or micro) of pivots independently.
        Call once per layer — this module does not merge layers.
        """
        if not pivots:
            return []

        tokens: List[StructureToken] = []
        trend_state: Optional[str] = None  # 'up' / 'down' / None

        for i, pivot in enumerate(pivots):
            # 1. Pass-through directional token (HH/HL/LH/LL/UNKNOWN)
            if pivot.structure_label != StructureLabel.UNKNOWN:
                tokens.append(self._make_token(pivot.structure_label.value, pivot))

            # 2. BOS / CHOCH — trend-state machine over the label stream
            bos_choch, trend_state = self._classify_bos_choch(pivot, trend_state)
            if bos_choch:
                tokens.append(self._make_token(bos_choch, pivot))

            # 3. FIB_T — retracement of the prior leg
            if i >= 2:
                fib_token = self._check_fib_tag(pivots[i - 2], pivots[i - 1], pivot)
                if fib_token:
                    tokens.append(fib_token)

            # 4. SWEEP — fast break + immediate reversal
            if i >= 2 and i + 1 < len(pivots):
                sweep_token = self._check_sweep(pivots, i)
                if sweep_token:
                    tokens.append(sweep_token)

            # 5. DIV_H / DIV_L — only if indicator series supplied
            if rsi_series is not None or macd_hist_series is not None:
                div_token = self._check_divergence(pivots, i, rsi_series, macd_hist_series)
                if div_token:
                    tokens.append(div_token)

        return sorted(tokens, key=lambda t: t.bar_index)

    # ── internals ─────────────────────────────────────────────

    def _make_token(self, token_name: str, pivot: PivotPoint, **overrides) -> StructureToken:
        return StructureToken(
            token        = token_name,
            token_id     = STRUCTURE_TOKENS[token_name],
            price        = overrides.get("price", pivot.price),
            timestamp_ms = pivot.timestamp_ms,
            bar_index    = pivot.bar_index,
            layer        = pivot.layer,
            degree       = pivot.degree,
            fib_context  = overrides.get("fib_context"),
            volume_surge = overrides.get("volume_surge", pivot.volume_surge),
            rsi_value    = overrides.get("rsi_value"),
            macd_hist    = overrides.get("macd_hist"),
        )

    def _classify_bos_choch(
        self, pivot: PivotPoint, trend_state: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Standard SMC trend-state machine driven by HH/HL/LH/LL labels.

        Continuation in the current trend direction → BOS.
        First label that opposes the current trend direction → CHOCH,
        and flips trend_state.
        A pivot that merely confirms a pullback within the existing trend
        (HL while trending up, LH while trending down) emits nothing —
        it's expected structure, not a structural event.
        """
        label = pivot.structure_label

        if label == StructureLabel.UNKNOWN:
            return None, trend_state

        if label == StructureLabel.HH:
            new_state = "up"
            event = "BOS" if trend_state in (None, "up") else "CHOCH"
            return event, new_state

        if label == StructureLabel.LL:
            new_state = "down"
            event = "BOS" if trend_state in (None, "down") else "CHOCH"
            return event, new_state

        if label == StructureLabel.LH:
            if trend_state == "up":
                return "CHOCH", "down"
            return None, trend_state  # normal pullback in a downtrend

        if label == StructureLabel.HL:
            if trend_state == "down":
                return "CHOCH", "up"
            return None, trend_state  # normal pullback in an uptrend

        return None, trend_state

    def _check_fib_tag(
        self, leg_start: PivotPoint, leg_end: PivotPoint, current: PivotPoint
    ) -> Optional[StructureToken]:
        """
        leg_start → leg_end defines the prior swing leg.
        current is the retracement/extension of that leg.
        Tag FIB_T if the ratio lands within tolerance of a key Fib ratio.
        """
        leg_length = abs(leg_end.price - leg_start.price)
        if leg_length == 0:
            return None

        move = abs(current.price - leg_end.price)
        ratio = move / leg_length

        for key_ratio in KEY_FIB_RATIOS:
            if abs(ratio - key_ratio) <= key_ratio * self.fib_tolerance:
                return self._make_token("FIB_T", current, fib_context=round(ratio, 4))

        return None

    def _check_sweep(self, pivots: Sequence[PivotPoint], i: int) -> Optional[StructureToken]:
        """
        SWEEP = this pivot fast-breaks the prior same-type extreme
        (HH or LL, confirmed within sweep_max_bars of the prior pivot of
        the SAME type), and the very next pivot reverses back through a
        meaningful portion of the break before continuing.

        This approximates a liquidity-sweep / B-breach event using only
        pivot-to-pivot data, without requiring tick-level wick analysis.
        """
        pivot = pivots[i]
        if pivot.structure_label not in (StructureLabel.HH, StructureLabel.LL):
            return None

        # Find the prior pivot of the SAME type to measure "fast"-ness against.
        same_type_prior = None
        for j in range(i - 1, -1, -1):
            if pivots[j].swing_type == pivot.swing_type:
                same_type_prior = pivots[j]
                break
        if same_type_prior is None:
            return None

        bars_between = pivot.bar_index - same_type_prior.bar_index
        if bars_between > self.sweep_max_bars * 4:
            # not "fast" by any reasonable margin — skip cheaply
            pass  # still allow check below; bars_from_prior on opposite-type chain differs.

        break_size = abs(pivot.price - same_type_prior.price)
        if break_size == 0:
            return None

        next_pivot = pivots[i + 1]  # opposite type, the reversal leg
        reversal_size = abs(next_pivot.price - pivot.price)

        is_fast = pivot.bars_from_prior <= self.sweep_max_bars
        is_sharp_reversal = reversal_size >= (break_size * self.sweep_min_reversal_pct)

        if is_fast and is_sharp_reversal:
            return self._make_token("SWEEP", pivot)

        return None

    def _check_divergence(
        self,
        pivots: Sequence[PivotPoint],
        i: int,
        rsi_series: Optional[dict],
        macd_hist_series: Optional[dict],
    ) -> Optional[StructureToken]:
        """
        Compares this pivot's RSI/MACD-hist against the prior pivot of the
        SAME type. Requires an external indicator series keyed by bar_index
        since indicators.py is currently an empty stub in this repo.
        """
        pivot = pivots[i]
        if pivot.structure_label not in (StructureLabel.HH, StructureLabel.LL):
            return None  # divergence only meaningful at new structural extremes

        same_type_prior = None
        for j in range(i - 1, -1, -1):
            if pivots[j].swing_type == pivot.swing_type:
                same_type_prior = pivots[j]
                break
        if same_type_prior is None:
            return None

        def _lookup(series, bar_index):
            return series.get(bar_index) if series else None

        rsi_now, rsi_prior = _lookup(rsi_series, pivot.bar_index), _lookup(rsi_series, same_type_prior.bar_index)
        macd_now, macd_prior = _lookup(macd_hist_series, pivot.bar_index), _lookup(macd_hist_series, same_type_prior.bar_index)

        indicator_now = rsi_now if rsi_now is not None else macd_now
        indicator_prior = rsi_prior if rsi_prior is not None else macd_prior
        if indicator_now is None or indicator_prior is None:
            return None

        if pivot.structure_label == StructureLabel.HH and indicator_now < indicator_prior:
            return self._make_token("DIV_H", pivot, rsi_value=rsi_now, macd_hist=macd_now)

        if pivot.structure_label == StructureLabel.LL and indicator_now > indicator_prior:
            return self._make_token("DIV_L", pivot, rsi_value=rsi_now, macd_hist=macd_now)

        return None

    # ── persistence helpers ──────────────────────────────────

    def save_tokens(self, tokens: List[StructureToken], path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([t.to_dict() for t in tokens], f, indent=2)
        return path


# ─────────────────────────────────────────────────────────────
# CLI entry point — quick smoke test against saved pivot JSON
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.waveconf.pivots.pivot_schema import PivotPoint

    macro_path = sys.argv[1] if len(sys.argv) > 1 else "data/pivots/BTCUSD_1D_pivots_macro.json"

    with open(macro_path) as f:
        raw = json.load(f)
    pivots = [PivotPoint.from_dict(p) for p in raw["pivots"]]

    tokenizer = StructureTokenizer()
    tokens = tokenizer.run(pivots)

    print(f"Loaded {len(pivots)} pivots → emitted {len(tokens)} tokens")
    for t in tokens[-15:]:
        print(" ", t)
