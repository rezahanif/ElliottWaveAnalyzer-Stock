"""
pattern_detector.py
--------------------
Classifies the geometric relationship between a resistance trendline
(fitted to highs) and a support trendline (fitted to lows) into one of
the pattern types defined in config/completion_rates.yaml:

    rising_wedge, falling_wedge,
    ascending_broadening_wedge, descending_broadening_wedge,
    symmetrical_triangle (contracting), ascending_triangle (contracting),
    descending_triangle (contracting), expanding_triangle,
    channel_ascending, channel_descending, channel_horizontal

Pipeline position: consumes TrendlineBuilder output (fib_engine/trendline.py).
Feeds the Confluence Scorer indirectly via FibonacciEngine's measured-move
projections, which look up completion_rates.yaml by pattern_type.

Classification logic (summary)
===============================
1. Normalize both trendline slopes to slope_pct_per_bar (already done
   by Trendline) so BTC's price-regime changes don't bias thresholds.
2. Measure width (resistance - support) at the start and end of the
   fitted window. Width change % tells us converge / diverge / constant.
3. Branch on (width behaviour) x (slope signs) per the decision table
   in config/pattern_thresholds.yaml.

This module is pure rule-based math — no ML. All thresholds live in
config/pattern_thresholds.yaml, not hardcoded here, so they can be
tuned without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import yaml

from src.waveconf.fib_engine.trendline import Trendline


# ─────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = os.path.join("config", "pattern_thresholds.yaml")


def _load_thresholds(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# Output record
# ─────────────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    pattern_type:       str            # key into config/completion_rates.yaml
    family:             str            # wedge / triangle / channel / broadening / ambiguous
    resistance:         Trendline
    support:            Trendline
    width_start:        float
    width_end:          float
    width_change_pct:   float          # signed: positive = diverging, negative = converging
    confidence:         float          # 0.0-1.0
    notes:              str = ""

    def __repr__(self) -> str:
        return (
            f"<PatternMatch {self.pattern_type} conf={self.confidence:.2f} "
            f"width_change={self.width_change_pct:+.1f}%>"
        )


# ─────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────

class PatternDetector:
    def __init__(self, thresholds: Optional[dict] = None, config_path: str = _DEFAULT_CONFIG_PATH):
        self.t = thresholds if thresholds is not None else _load_thresholds(config_path)

    # ── public API ────────────────────────────────────────────

    def detect(self, resistance: Optional[Trendline], support: Optional[Trendline]) -> Optional[PatternMatch]:
        """
        Classify the geometry formed by a resistance trendline (highs)
        and a support trendline (lows) fitted over the SAME pivot window.
        Returns None if either trendline is missing, has too few pivots,
        or fits too poorly (R^2 below min_r_squared_for_pattern) to mean
        anything geometrically.
        """
        if resistance is None or support is None:
            return None

        if resistance.pivot_count < self.t["min_pivots_per_side"] or support.pivot_count < self.t["min_pivots_per_side"]:
            return None

        if resistance.r_squared < self.t["min_r_squared_for_pattern"] and support.r_squared < self.t["min_r_squared_for_pattern"]:
            # Both fits are poor — no reliable geometry to classify.
            return None

        width_start = resistance.start_price - support.start_price
        width_end   = resistance.end_price - support.end_price

        if width_start <= 0 or width_end <= 0:
            # Lines have already crossed within the fitted window —
            # not a valid channel/wedge/triangle (support above resistance
            # means the "pattern" has already broken down structurally).
            return PatternMatch(
                pattern_type="invalid_crossed_lines", family="ambiguous",
                resistance=resistance, support=support,
                width_start=width_start, width_end=width_end,
                width_change_pct=0.0, confidence=0.0,
                notes="Support and resistance trendlines have crossed within the fitted window.",
            )

        mean_width = (width_start + width_end) / 2.0
        width_change_pct = ((width_end - width_start) / mean_width) * 100.0 if mean_width else 0.0

        flat_thr        = self.t["flat_slope_threshold_pct_per_bar"]
        parallel_thr    = self.t["parallel_slope_diff_threshold_pct_per_bar"]
        constant_thr    = self.t["constant_width_tolerance_pct"]
        min_converge    = self.t["min_convergence_pct"]

        r_is_flat = resistance.is_flat(flat_thr)
        s_is_flat = support.is_flat(flat_thr)
        r_rising  = resistance.is_rising(flat_thr)
        r_falling = resistance.is_falling(flat_thr)
        s_rising  = support.is_rising(flat_thr)
        s_falling = support.is_falling(flat_thr)

        slope_diff = abs(resistance.slope_pct_per_bar - support.slope_pct_per_bar)
        is_constant_width = abs(width_change_pct) < constant_thr
        is_converging      = width_change_pct <= -min_converge
        is_diverging        = width_change_pct >= min_converge

        pattern_type, family, notes = self._classify(
            r_is_flat, s_is_flat, r_rising, r_falling, s_rising, s_falling,
            slope_diff, parallel_thr, is_constant_width, is_converging, is_diverging,
            resistance, support, width_change_pct,
        )

        confidence = self._score_confidence(resistance, support, pattern_type)

        return PatternMatch(
            pattern_type      = pattern_type,
            family            = family,
            resistance        = resistance,
            support           = support,
            width_start       = width_start,
            width_end         = width_end,
            width_change_pct  = width_change_pct,
            confidence        = confidence,
            notes             = notes,
        )

    # ── internals ─────────────────────────────────────────────

    def _classify(
        self, r_flat, s_flat, r_rising, r_falling, s_rising, s_falling,
        slope_diff, parallel_thr, is_constant_width, is_converging, is_diverging,
        resistance: Trendline, support: Trendline, width_change_pct: float,
    ) -> tuple[str, str, str]:

        # ── CHANNELS: roughly parallel, constant width ──
        if is_constant_width and slope_diff <= parallel_thr:
            if r_rising and s_rising:
                return "channel_ascending", "channel", "Both boundaries rising, near-parallel, constant width."
            if r_falling and s_falling:
                return "channel_descending", "channel", "Both boundaries falling, near-parallel, constant width."
            if r_flat and s_flat:
                return "channel_horizontal", "channel", "Both boundaries flat, constant width — horizontal range."
            # parallel but mixed flat/angled with tiny slope_diff is still
            # effectively a channel; fall through to descending/ascending
            # based on resistance direction as the dominant signal.
            if r_rising or s_rising:
                return "channel_ascending", "channel", "Near-parallel boundaries, net upward drift."
            return "channel_descending", "channel", "Near-parallel boundaries, net downward drift."

        # ── CONTRACTING (converging width) ──
        if is_converging:
            if r_rising and s_rising:
                return "rising_wedge", "wedge", "Both boundaries rising, range narrowing — bearish wedge."
            if r_falling and s_falling:
                return "falling_wedge", "wedge", "Both boundaries falling, range narrowing — bullish wedge."
            if r_falling and s_rising:
                return "symmetrical_triangle", "triangle", "Resistance falling, support rising — converging to apex."
            if r_flat and s_rising:
                return "ascending_triangle", "triangle", "Flat resistance, rising support — bullish bias."
            if r_falling and s_flat:
                return "descending_triangle", "triangle", "Declining resistance, flat support — bearish bias."
            # Converging but slopes don't fit a clean category above
            # (e.g. both flat, or one barely angled) — call it out rather
            # than force-fit a label.
            return (
                "symmetrical_triangle", "triangle",
                f"Converging width but ambiguous slope combination "
                f"(r={resistance.slope_pct_per_bar:.3f}, s={support.slope_pct_per_bar:.3f}); "
                f"defaulted to symmetrical_triangle — verify manually.",
            )

        # ── EXPANDING (diverging width) ──
        if is_diverging:
            if r_rising and s_rising and resistance.slope_pct_per_bar > support.slope_pct_per_bar:
                return "ascending_broadening_wedge", "broadening", "Both rising, resistance steeper — diverging while net ascending (ABW)."
            if r_falling and s_falling and support.slope_pct_per_bar < resistance.slope_pct_per_bar:
                return "descending_broadening_wedge", "broadening", "Both falling, support steeper — diverging while net descending."
            if r_rising and s_falling:
                return "expanding_triangle", "triangle", "Resistance rising, support falling — symmetric divergence (reverse triangle)."
            return (
                "expanding_triangle", "triangle",
                f"Diverging width but ambiguous slope combination "
                f"(r={resistance.slope_pct_per_bar:.3f}, s={support.slope_pct_per_bar:.3f}); "
                f"defaulted to expanding_triangle — verify manually.",
            )

        # ── Neither clearly constant, converging, nor diverging ──
        return (
            "ambiguous", "ambiguous",
            f"Width change {width_change_pct:.1f}% falls between thresholds — "
            f"not a clean channel, wedge, or triangle. Re-check with a different "
            f"pivot window.",
        )

    def _score_confidence(self, resistance: Trendline, support: Trendline, pattern_type: str) -> float:
        if pattern_type in ("ambiguous", "invalid_crossed_lines"):
            return 0.0

        weights = self.t["confidence_weights"]
        saturation = self.t["pivot_count_bonus_saturation"]

        r_squared_avg = (resistance.r_squared + support.r_squared) / 2.0

        avg_pivots = (resistance.pivot_count + support.pivot_count) / 2.0
        pivot_bonus = min(avg_pivots / saturation, 1.0)

        # Convergence clarity: not directly tracked per-call here, so we
        # approximate using how clean the fits are (high R^2 on both sides
        # means the converge/diverge classification itself is trustworthy).
        convergence_clarity = min(resistance.r_squared, support.r_squared)

        confidence = (
            weights["r_squared_avg"] * r_squared_avg
            + weights["pivot_count_bonus"] * pivot_bonus
            + weights["convergence_clarity"] * convergence_clarity
        )
        return round(min(max(confidence, 0.0), 1.0), 4)
