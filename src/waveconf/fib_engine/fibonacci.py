"""
fibonacci.py
------------
FibonacciEngine — pure math module. No ML, no heuristics.

Computes all price targets from confirmed pivot inputs:
    1. Impulse wave ratio targets  (W2 retrace, W3 extension, W4 retrace, W5 extension)
    2. Correction wave ratio targets (WA, WB, WC per correction subtype)
    3. Pattern measured-move targets (ABW, wedge, triangle — scaled by empirical rate)
    4. Dual-tool Fibonacci cluster   (Tool A: 2.618 from C top / Tool B: 1.618 B→C)
    5. Cluster validation            (are two targets within 2% of each other?)

All ratio bounds and empirical rates are loaded from:
    config/completion_rates.yaml
    config/correction_rules.yaml  (tolerance)

Hard wave rules (W3 min, W2 max, W4 overlap) live in invalidation.py.
This module only produces targets — it does not validate counts.

Usage:
    from src.waveconf.fib_engine.fibonacci import FibonacciEngine, FibTarget, ClusterResult

    engine = FibonacciEngine()

    # Pattern measured move
    target = engine.measured_move(
        pattern_type='ascending_broadening_wedge',
        top_price=82720.71,
        support_price=68000.0,
        breakout_price=68000.0,
        direction='bearish'
    )

    # Dual-tool cluster
    cluster = engine.dual_cluster(
        c_top=82720.71,
        b_low=63000.0,
        direction='bearish'
    )
"""

from __future__ import annotations

import os
import yaml
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple



# ─────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────

def _load_yaml(relative_path: str) -> dict:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    full = os.path.join(base, relative_path)
    with open(full, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# Output types
# ─────────────────────────────────────────────────────────────

@dataclass
class FibTarget:
    """
    A single computed price target with full provenance.

    Fields
    ------
    price           float   The computed target price
    method          str     How it was computed (e.g. 'extension_2.618_from_c_top')
    ratio           float   The Fibonacci ratio applied (e.g. 2.618)
    anchor_price    float   The pivot price used as anchor
    anchor_label    str     Human label for the anchor (e.g. 'C_top', 'B_low')
    direction       str     'bearish' or 'bullish'
    pattern_type    str     Pattern context (e.g. 'ascending_broadening_wedge') or ''
    note            str     Optional explanatory note
    """
    price:          float
    method:         str
    ratio:          float
    anchor_price:   float
    anchor_label:   str
    direction:      str
    pattern_type:   str   = ""
    note:           str   = ""

    def __repr__(self) -> str:
        return (
            f"FibTarget(${self.price:,.2f} | {self.method} "
            f"| ratio={self.ratio} | anchor={self.anchor_label}@{self.anchor_price:,.2f})"
        )
@dataclass
class ClusterResult:
    """
    Output of FibonacciEngine.dual_cluster() — the core confluence output.

    Contains both independently-derived targets and the cluster assessment.
    This is the object that feeds into ConfluenceChecker.

    Fields
    ------
    target_a        FibTarget   Tool A result (2.618 extension from C top)
    target_b        FibTarget   Tool B result (1.618 extension from B→C)
    measured_move   FibTarget   Pattern measured-move target (optional)
    cluster_valid   bool        True if target_a and target_b are within cluster_threshold_pct
    proximity_pct   float       Actual % difference between target_a and target_b
    cluster_upper   float       Higher of the two cluster prices
    cluster_lower   float       Lower of the two cluster prices
    cluster_mid     float       Midpoint of the cluster zone
    cluster_strength float      0.0–1.0 (1.0 = perfectly coincident, 0.0 = beyond threshold)
    scenario_a      FibTarget   First reaction zone (target_a, partial entry 10–20%)
    scenario_b      FibTarget   Main target zone (target_b, full entry)
    """
    target_a:        FibTarget
    target_b:        FibTarget
    measured_move:   Optional[FibTarget]
    cluster_valid:   bool
    proximity_pct:   float
    cluster_upper:   float
    cluster_lower:   float
    cluster_mid:     float
    cluster_strength: float
    scenario_a:      FibTarget
    scenario_b:      FibTarget

    def __repr__(self) -> str:
        valid = "✅ CLUSTER" if self.cluster_valid else "❌ NO CLUSTER"
        return (
            f"ClusterResult({valid} | "
            f"A=${self.target_a.price:,.2f} / B=${self.target_b.price:,.2f} | "
            f"proximity={self.proximity_pct:.2f}% | strength={self.cluster_strength:.2f})"
        )


@dataclass
class ImpulseTargets:
    """All computed targets for one impulse wave sequence."""
    w2_retrace_zone:  Tuple[float, float]   # (lower_bound, upper_bound)
    w3_min_target:    float                  # hard minimum (1.0× W1)
    w3_typical_target: float                 # typical (1.618× W1)
    w4_retrace_zone:  Tuple[float, float]
    w5_equal_w1:      float                  # most common W5 target
    w5_zone:          Tuple[float, float]
    invalidation_w2:  float                  # W2 must not breach this
    invalidation_w4:  float                  # W4 must not breach this (non-diagonal)


@dataclass
class CorrectionTargets:
    """All computed targets for one correction sequence (ABC)."""
    correction_type:  str
    b_zone:           Tuple[float, float]    # expected B wave range
    c_typical:        float                  # typical C target
    c_zone:           Tuple[float, float]    # C range
    b_breach_price:   Optional[float]        # price B must exceed (expanded/running flat)
    b_breach_expected: bool
    # ─────────────────────────────────────────────────────────────
# FibonacciEngine
# ─────────────────────────────────────────────────────────────

class FibonacciEngine:
    """
    Pure math Fibonacci price target engine.

    All inputs are plain floats (prices). No PivotPoint objects here —
    the caller extracts prices from pivots and passes them in.
    This keeps the engine testable without any pivot infrastructure.

    Instantiate once per session. Config is loaded at init time.
    """

    # Standard Fibonacci ratios — immutable
    FIB_RATIOS = {
        "retrace": [0.236, 0.382, 0.500, 0.618, 0.786, 1.000],
        "extend":  [1.000, 1.272, 1.414, 1.618, 2.000, 2.618, 3.618, 4.236],
    }

    # Wave ratio rules from v2.0 spec Section 7
    WAVE_RULES = {
        "wave2_retrace":          (0.382, 0.786),
        "wave2_max":               1.000,          # HARD — never exceed 100%
        "wave3_extend":           (1.000, 4.236),
        "wave3_min":               1.000,          # HARD minimum (relative to W1)
        "wave4_retrace":          (0.236, 0.500),
        "wave4_no_overlap":        True,           # HARD — checked in invalidation.py
        "wave5_extend":           (0.382, 1.618),
        "wave5_equal_wave1":       1.000,
        "waveA_retrace":          (0.382, 0.618),
        "waveB_regular_retrace":  (0.810, 1.000),
        "waveB_expanded_retrace": (1.000, 1.382),
        "waveB_running_retrace":  (1.100, 1.382),
        "waveB_zigzag_retrace":   (0.382, 0.786),
        "waveC_regular":          (0.618, 1.000),
        "waveC_expanded":         (1.236, 1.618),
        "waveC_zigzag":           (0.618, 1.618),
        "fib_cluster_ext_A":       2.618,
        "fib_cluster_ext_B":       1.618,
    }

    def __init__(
        self,
        rates_config:      str = "config/completion_rates.yaml",
        rules_config:      str = "config/correction_rules.yaml",
    ):
        rates = _load_yaml(rates_config)
        rules = _load_yaml(rules_config)

        self.completion_rates:     Dict[str, float] = {
            k: v["rate"] for k, v in rates["patterns"].items()
        }
        self.cluster_threshold_pct: float = float(rates["cluster_threshold_pct"])
        self.invalidation_buffer:   float = float(rates["invalidation_buffer_pct"]) / 100.0
        self.wave_ratio_tolerance:  float = float(rules.get("tolerance", 0.05))

        # Cluster tool ratios from config
        fib_tools = rates.get("fibonacci_cluster_tools", {})
        self.tool_a_ratio: float = float(fib_tools.get("tool_a", {}).get("ratio", 2.618))
        self.tool_b_ratio: float = float(fib_tools.get("tool_b", {}).get("ratio", 1.618))

    # ─────────────────────────────────────────────────────────
    # 1. Generic Fibonacci extensions and retracements
    # ─────────────────────────────────────────────────────────

    def extension(
        self,
        anchor_price: float,
        swing_range:  float,
        ratio:        float,
        direction:    str,
        anchor_label: str = "anchor",
        method_note:  str = "",
        use_log_scale: bool = False,
    ) -> FibTarget:
        """
        Compute a Fibonacci extension target.

        target = anchor_price ± (swing_range × ratio)
        direction='bearish' → subtract (project downward)
        direction='bullish' → add (project upward)
        """
        if use_log_scale:
            log_anchor = math.log(anchor_price)
            log_offset = swing_range * ratio
            log_target = log_anchor - log_offset if direction == "bearish" else log_anchor + log_offset
            price = math.exp(log_target)
        else:
            offset = swing_range * ratio
            price  = anchor_price - offset if direction == "bearish" else anchor_price + offset
            
        return FibTarget(
            price        = round(price, 2),
            method       = method_note or f"extension_{ratio}_from_{anchor_label}",
            ratio        = ratio,
            anchor_price = anchor_price,
            anchor_label = anchor_label,
            direction    = direction,
        )

    def retracement(
        self,
        wave_start: float,
        wave_end:   float,
        ratio:      float,
        label:      str = "retrace",
    ) -> FibTarget:
        """
        Compute a Fibonacci retracement level.

        retrace_price = wave_end + (wave_range × ratio)  [for bearish wave, end < start]
        retrace_price = wave_end - (wave_range × ratio)  [for bullish wave, end > start]
        """
        wave_range = abs(wave_end - wave_start)
        if wave_end < wave_start:
            # bearish wave (falling) — retrace goes up
            price = wave_end + wave_range * ratio
        else:
            # bullish wave (rising) — retrace goes down
            price = wave_end - wave_range * ratio
        return FibTarget(
            price        = round(price, 2),
            method       = f"retracement_{ratio}",
            ratio        = ratio,
            anchor_price = wave_end,
            anchor_label = label,
            direction    = "bullish" if wave_end < wave_start else "bearish",
        )

    def all_retracements(self, wave_start: float, wave_end: float) -> List[FibTarget]:
        """Compute all standard retracement levels for a wave."""
        return [
            self.retracement(wave_start, wave_end, r, f"retrace_{r}")
            for r in self.FIB_RATIOS["retrace"]
        ]

    def all_extensions(
        self,
        anchor: float,
        swing_range: float,
        direction: str,
        anchor_label: str = "anchor",
    ) -> List[FibTarget]:
        """Compute all standard extension levels from an anchor."""
        return [
            self.extension(anchor, swing_range, r, direction, anchor_label)
            for r in self.FIB_RATIOS["extend"]
        ]
# ─────────────────────────────────────────────────────────
    # 2. Impulse wave targets
    # ─────────────────────────────────────────────────────────

    def impulse_targets(
        self,
        origin:    float,   # wave start price
        w1_end:    float,   # end of wave 1
        bullish:   bool = True,
    ) -> ImpulseTargets:
        """
        Compute all expected price targets for waves 2–5 given origin and W1.

        Only requires origin and W1 end — subsequent waves are projected
        from W1 ranges and standard Fibonacci ratios.
        All prices are returned as absolute price levels.
        """
        r1 = abs(w1_end - origin)

        def _proj(base: float, range_: float, ratio: float) -> float:
            """Project ratio × range_ from base, in trend direction."""
            return (base + range_ * ratio) if bullish else (base - range_ * ratio)

        def _ret(from_: float, range_: float, ratio: float) -> float:
            """Retrace ratio × range_ against trend direction from from_."""
            return (from_ - range_ * ratio) if bullish else (from_ + range_ * ratio)

        # Wave 2 retracement of Wave 1
        w2_lo = _ret(w1_end, r1, self.WAVE_RULES["wave2_retrace"][1])  # 78.6%
        w2_hi = _ret(w1_end, r1, self.WAVE_RULES["wave2_retrace"][0])  # 38.2%

        # Wave 3 — minimum hard floor is 1.0× W1 beyond W1's end
        w3_min     = _proj(w1_end, r1, self.WAVE_RULES["wave3_min"])    # 1.0×
        w3_typical = _proj(w1_end, r1, 1.618)                           # 1.618×

        # Wave 4 retracement of Wave 3 — depends on W3 actual; estimated from W3 typical
        r3_typical = r1 * 1.618
        w4_lo = _ret(w3_typical, r3_typical, self.WAVE_RULES["wave4_retrace"][1])  # 50%
        w4_hi = _ret(w3_typical, r3_typical, self.WAVE_RULES["wave4_retrace"][0])  # 23.6%

        # Wave 5 — most common = equal to Wave 1
        w5_eq_w1 = _proj(w4_lo, r1, 1.0)   # W4 low + W1 range (using W4 deep end)
        r5_lo = r1 * self.WAVE_RULES["wave5_extend"][0]  # 0.382× W1
        r5_hi = r1 * self.WAVE_RULES["wave5_extend"][1]  # 1.618× W1
        w5_lo = _proj(w4_lo, r5_lo, 1.0)
        w5_hi = _proj(w4_lo, r5_hi, 1.0)

        # Invalidation prices
        # Wave 2 must not retrace > 100% of Wave 1
        inv_w2 = origin   # if W2 breaches origin, count invalid
        # Wave 4 must not enter Wave 1 territory (non-diagonal)
        inv_w4 = w1_end   # if W4 breaches W1 end, count invalid

        return ImpulseTargets(
            w2_retrace_zone   = (round(w2_lo, 2), round(w2_hi, 2)),
            w3_min_target     = round(w3_min, 2),
            w3_typical_target = round(w3_typical, 2),
            w4_retrace_zone   = (round(w4_lo, 2), round(w4_hi, 2)),
            w5_equal_w1       = round(w5_eq_w1, 2),
            w5_zone           = (round(w5_lo, 2), round(w5_hi, 2)),
            invalidation_w2   = round(inv_w2, 2),
            invalidation_w4   = round(inv_w4, 2),
        )

    # ─────────────────────────────────────────────────────────
    # 3. Correction wave targets
    # ─────────────────────────────────────────────────────────

    def correction_targets(
        self,
        correction_type: str,
        a_start:         float,   # price where correction begins
        a_end:           float,   # price where Wave A ends
    ) -> CorrectionTargets:
        """
        Compute B and C wave targets given correction type and Wave A range.

        correction_type must be one of:
            regular_flat | expanded_flat | running_flat |
            single_zigzag | double_zigzag |
            contracting_symmetrical | ... (triangle variants)
        """
        r_A = abs(a_end - a_start)
        downward = a_end < a_start  # correction going down first

        def _b(ratio: float) -> float:
            """Project B from A_end back toward (and possibly past) A_start."""
            return (a_end + r_A * ratio) if downward else (a_end - r_A * ratio)

        def _c(ratio: float) -> float:
            """Project C from B zone estimate back in A direction."""
            return (a_end - r_A * ratio) if downward else (a_end + r_A * ratio)

        ct = correction_type.lower()

        if ct == "regular_flat":
            b_lo, b_hi = self.WAVE_RULES["waveB_regular_retrace"]
            c_lo, c_hi = self.WAVE_RULES["waveC_regular"]
            b_breach_expected = False
            b_breach_price    = None

        elif ct == "expanded_flat":
            b_lo, b_hi = self.WAVE_RULES["waveB_expanded_retrace"]
            c_lo, c_hi = self.WAVE_RULES["waveC_expanded"]
            b_breach_expected = True
            b_breach_price    = a_start   # B must exceed A's start

        elif ct == "running_flat":
            b_lo, b_hi = self.WAVE_RULES["waveB_running_retrace"]
            c_lo, c_hi = (0.382, 0.786)  # C truncated, typically 38–78% of A
            b_breach_expected = True
            b_breach_price    = a_start

        elif ct in ("single_zigzag", "zigzag"):
            b_lo, b_hi = self.WAVE_RULES["waveB_zigzag_retrace"]
            c_lo, c_hi = self.WAVE_RULES["waveC_zigzag"]
            b_breach_expected = False
            b_breach_price    = None

        elif ct in ("double_zigzag", "triple_zigzag"):
            # X wave (same as zigzag B)
            b_lo, b_hi = self.WAVE_RULES["waveB_zigzag_retrace"]
            c_lo, c_hi = (0.618, 1.000)  # Y ≈ W
            b_breach_expected = False
            b_breach_price    = None

        else:
            # Triangle and combination — use generic ranges
            b_lo, b_hi = (0.382, 0.786)
            c_lo, c_hi = (0.500, 1.000)
            b_breach_expected = False
            b_breach_price    = None

        return CorrectionTargets(
            correction_type   = ct,
            b_zone            = (round(_b(b_lo), 2), round(_b(b_hi), 2)),
            c_typical         = round(_c(1.000), 2),
            c_zone            = (round(_c(c_lo), 2), round(_c(c_hi), 2)),
            b_breach_price    = round(b_breach_price, 2) if b_breach_price else None,
            b_breach_expected = b_breach_expected,
        )
    # ─────────────────────────────────────────────────────────
    # 4. Pattern measured-move target
    # ─────────────────────────────────────────────────────────

    def measured_move(
        self,
        pattern_type:    str,
        top_price:       float,
        support_price:   float,
        breakout_price:  float,
        direction:       str = "bearish",
    ) -> FibTarget:
        """
        Compute pattern measured-move target scaled by empirical completion rate.

        Formula:
            pattern_height = abs(top_price - support_price)
            target = breakout_price - (pattern_height × completion_rate)  [bearish]
            target = breakout_price + (pattern_height × completion_rate)  [bullish]

        Parameters
        ----------
        pattern_type    str     Key matching config patterns (e.g. 'ascending_broadening_wedge')
        top_price       float   Structural high of the pattern (C top / e-wave top for ABW)
        support_price   float   Lower trendline price at the breakout bar
        breakout_price  float   Price where pattern support was broken
        direction       str     'bearish' (project down) or 'bullish' (project up)
        """
        rate = self.completion_rates.get(pattern_type)
        if rate is None:
            raise ValueError(
                f"Unknown pattern_type '{pattern_type}'. "
                f"Valid options: {list(self.completion_rates.keys())}"
            )

        pattern_height = abs(top_price - support_price)
        scaled_move    = pattern_height * rate

        if direction == "bearish":
            target = breakout_price - scaled_move
        else:
            target = breakout_price + scaled_move

        pct_from_breakout = scaled_move / breakout_price * 100

        return FibTarget(
            price        = round(target, 2),
            method       = f"measured_move_{rate:.0%}_of_{pattern_type}",
            ratio        = rate,
            anchor_price = breakout_price,
            anchor_label = "breakout_point",
            direction    = direction,
            pattern_type = pattern_type,
            note         = (
                f"pattern_height={pattern_height:,.2f} × {rate:.0%} = "
                f"{scaled_move:,.2f} ({pct_from_breakout:.1f}% from breakout)"
            ),
        )

    # ─────────────────────────────────────────────────────────
    # 5. Dual-tool Fibonacci cluster (the core confluence output)
    # ─────────────────────────────────────────────────────────

    def dual_cluster(
        self,
        c_top:     float,
        b_low:     float,
        direction: str = "bearish",
        ab_range:  Optional[float] = None,
        a_price:   Optional[float] = None,
        version:   str = "v2_linear",
    ) -> ClusterResult:
        """
        Compute the two-tool Fibonacci cluster confluence output.
        Supports 5 historical/experimental versions for comparison and regression checks:
        
        - 'v1_buggy': Original buggy implementation (Target A == Target B mathematically).
        - 'v2_linear': Linear 3-pivot logic (resolves the v1 bug using ab_range).
        - 'v3_gated': Linear 3-pivot logic with macro trend regime gating constraints.
        - 'v4_log': Log-scale calculations, protecting against negative prices for large swings.
        - 'v5_relaxed': Log-scale calculations with bc_range fallback when ab_range is missing,
                        and a wider cluster threshold (15%) to restore signal volume.
                        Keeps correct math while accepting approximate confluence.
        """
        if direction not in ("bearish", "bullish"):
            raise ValueError(f"direction must be 'bearish' or 'bullish', got '{direction}'")

        if version == "v1_buggy":
            # Buggy logic: Tool A and Tool B both use bc_range
            bc_range = abs(c_top - b_low)
            target_a = self.extension(
                anchor_price=c_top,
                swing_range=bc_range,
                ratio=self.tool_a_ratio,
                direction=direction,
                anchor_label="C_top",
                method_note=f"extension_{self.tool_a_ratio}_from_C_top"
            )
            target_b = self.extension(
                anchor_price=b_low,
                swing_range=bc_range,
                ratio=self.tool_b_ratio,
                direction=direction,
                anchor_label="B_low",
                method_note=f"extension_{self.tool_b_ratio}_from_B_low"
            )
            has_wave_a = True

        elif version in ("v2_linear", "v3_gated"):
            # Linear 3-pivot logic using ab_range (to decouple the swings)
            bc_range = abs(c_top - b_low)
            b_range = ab_range if ab_range is not None else bc_range
            
            target_a = self.extension(
                anchor_price=c_top,
                swing_range=bc_range,
                ratio=self.tool_a_ratio,
                direction=direction,
                anchor_label="C_top",
                method_note=f"extension_{self.tool_a_ratio}_from_C_top"
            )
            target_b = self.extension(
                anchor_price=b_low,
                swing_range=b_range,
                ratio=self.tool_b_ratio,
                direction=direction,
                anchor_label="B_low",
                method_note=f"extension_{self.tool_b_ratio}_from_B_low"
            )
            has_wave_a = ab_range is not None

        elif version == "v4_log":
            # Log-scale calculation (prevents negative price target projections)
            log_c = math.log(c_top)
            log_b = math.log(b_low)
            log_bc_range = abs(log_c - log_b)
            
            # Anchor B price in log is log_b for bearish, and log_c for bullish
            b_price_log = log_b if direction == "bearish" else log_c
            
            if a_price is not None:
                log_ab_range = abs(math.log(a_price) - b_price_log)
                log_b_range = log_ab_range
                has_wave_a = True
            elif ab_range is not None:
                # Approximated log range if only linear ab_range is provided
                b_price_val = b_low if direction == "bearish" else c_top
                log_b_range = abs(math.log(b_price_val + ab_range) - b_price_log)
                has_wave_a = True
            else:
                log_b_range = log_bc_range
                has_wave_a = False
                
            target_a = self.extension(
                anchor_price=c_top,
                swing_range=log_bc_range,
                ratio=self.tool_a_ratio,
                direction=direction,
                anchor_label="C_top",
                method_note=f"extension_{self.tool_a_ratio}_from_C_top",
                use_log_scale=True
            )
            target_b = self.extension(
                anchor_price=b_low,
                swing_range=log_b_range,
                ratio=self.tool_b_ratio,
                direction=direction,
                anchor_label="B_low",
                method_note=f"extension_{self.tool_b_ratio}_from_B_low",
                use_log_scale=True
            )
        elif version == "v5_relaxed":
            # v5_relaxed: log-scale math (correct) but uses bc_range as wave-A surrogate
            # when ab_range is missing, so signals fire even without full 3-pivot history.
            # Cluster threshold is widened to 15% (vs 2% default) to accept loose confluences.
            #
            # Direction-specific ratios:
            #   Bearish: 2.618× BC from C top (deep extension) / 1.618× AB from B low
            #   Bullish: 1.618× BC from C bottom (post-correction target) / 1.0× AB from C bottom (measured move)
            log_c = math.log(c_top)
            log_b = math.log(b_low)
            log_bc_range = abs(log_c - log_b)

            # AB range: the prior swing used for Tool B's measured move
            # Bearish: AB = A_high → B_low (the decline before C top)
            # Bullish: AB = A_low → B_high (the rally before C bottom)
            if a_price is not None:
                if direction == "bearish":
                    log_b_range = abs(math.log(a_price) - log_b)
                else:
                    # Bullish: AB rally = |log(A_low) - log(B_high)|
                    log_b_range = abs(math.log(a_price) - log_c)
                has_wave_a = True
            elif ab_range is not None:
                log_b_range = abs(math.log(b_low + ab_range) - log_b)
                has_wave_a = True
            else:
                # Fallback: use bc_range as surrogate for ab_range (wider zone accepted)
                log_b_range = log_bc_range
                has_wave_a = True  # Accept even without confirmed wave-A

            if direction == "bearish":
                # Bearish: project DOWN from the high (c_top) using config ratios
                ratio_a = self.tool_a_ratio       # 2.618
                ratio_b = self.tool_b_ratio       # 1.618
                anchor_a = c_top
                anchor_b = b_low
                label_a  = "C_top"
                label_b  = "B_low"
                range_a  = log_bc_range
                range_b  = log_b_range
            else:
                # Bullish: project UP from the corrective low (b_low)
                # Use tighter ratios appropriate for post-correction targets
                ratio_a = self.tool_b_ratio       # 1.618 (BC extension target)
                ratio_b = 1.000                    # 1.0× AB measured move
                anchor_a = b_low
                anchor_b = b_low
                label_a  = "C_bottom"
                label_b  = "C_bottom"
                range_a  = log_bc_range            # BC swing
                range_b  = log_b_range             # AB swing

            target_a = self.extension(
                anchor_price=anchor_a,
                swing_range=range_a,
                ratio=ratio_a,
                direction=direction,
                anchor_label=label_a,
                method_note=f"extension_{ratio_a}_from_{label_a}",
                use_log_scale=True
            )
            target_b = self.extension(
                anchor_price=anchor_b,
                swing_range=range_b,
                ratio=ratio_b,
                direction=direction,
                anchor_label=label_b,
                method_note=f"extension_{ratio_b}_from_{label_b}",
                use_log_scale=True
            )
        else:
            raise ValueError(f"Unknown version: {version}")

        # v5_relaxed uses a wider cluster threshold to restore signal volume
        effective_threshold = 15.0 if version == "v5_relaxed" else self.cluster_threshold_pct

        # Compute proximity and validation
        proximity_pct = abs(target_a.price - target_b.price) / max(target_a.price, target_b.price) * 100
        
        if not has_wave_a:
            cluster_valid = False
            cluster_strength = 0.0
        else:
            cluster_valid = proximity_pct <= effective_threshold
            if cluster_valid:
                cluster_strength = max(0.0, 1.0 - (proximity_pct / effective_threshold))
            else:
                cluster_strength = 0.0

        cluster_upper = max(target_a.price, target_b.price)
        cluster_lower = min(target_a.price, target_b.price)
        cluster_mid   = (cluster_upper + cluster_lower) / 2

        if direction == "bearish":
            scenario_a = target_a if target_a.price > target_b.price else target_b
            scenario_b = target_b if target_b.price < target_a.price else target_a
        else:
            scenario_a = target_a if target_a.price < target_b.price else target_b
            scenario_b = target_b if target_b.price > target_a.price else target_a

        return ClusterResult(
            target_a         = target_a,
            target_b         = target_b,
            measured_move    = None,
            cluster_valid    = cluster_valid,
            proximity_pct    = round(proximity_pct, 4),
            cluster_upper    = round(cluster_upper, 2),
            cluster_lower    = round(cluster_lower, 2),
            cluster_mid      = round(cluster_mid, 2),
            cluster_strength = round(cluster_strength, 4),
            scenario_a       = scenario_a,
            scenario_b       = scenario_b,
        )

    def add_measured_move_to_cluster(
        self,
        cluster:         ClusterResult,
        measured_move:   FibTarget,
    ) -> ClusterResult:
        """
        Attach a measured_move target to an existing ClusterResult.
        Recalculates cluster_valid to check if measured_move also falls
        within the cluster zone — triple confluence.
        """
        mm_in_cluster = (
            cluster.cluster_lower * (1 - self.cluster_threshold_pct / 100)
            <= measured_move.price
            <= cluster.cluster_upper * (1 + self.cluster_threshold_pct / 100)
        )
        cluster.measured_move = measured_move
        if mm_in_cluster:
            cluster.cluster_strength = min(1.0, cluster.cluster_strength + 0.20)
            cluster.notes_mm = "Measured move inside cluster zone — triple confluence"
        else:
            cluster.notes_mm = f"Measured move (${measured_move.price:,.2f}) outside cluster zone"
        return cluster

    # ─────────────────────────────────────────────────────────
    # 6. Nearest Fibonacci level context (for token metadata)
    # ─────────────────────────────────────────────────────────

    def nearest_fib_level(
        self,
        price:      float,
        wave_start: float,
        wave_end:   float,
        tolerance:  Optional[float] = None,
    ) -> Optional[float]:
        """
        Find the nearest Fibonacci retracement ratio at a given price.
        Returns the ratio (e.g. 0.618) if within tolerance, else None.
        Used to populate PivotPoint.fib_context.
        """
        tol = tolerance if tolerance is not None else self.wave_ratio_tolerance
        wave_range = abs(wave_end - wave_start)
        if wave_range == 0:
            return None

        retrace_dist = abs(price - wave_end)
        ratio = retrace_dist / wave_range

        key_ratios = self.FIB_RATIOS["retrace"] + [1.618, 2.618]
        best_ratio  = None
        best_delta  = float("inf")
        for r in key_ratios:
            delta = abs(ratio - r)
            if delta < best_delta:
                best_delta = delta
                best_ratio = r

        if best_delta <= best_ratio * tol:
            return best_ratio
        return None


# ─────────────────────────────────────────────────────────────
# Optional typing import (Python 3.8 compat)
# ─────────────────────────────────────────────────────────────
from typing import Optional