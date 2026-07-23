"""
trendline.py
------------
Trendline dataclass + fit_trendline() builder.

pattern_detector.py's docstring describes itself as consuming
"TrendlineBuilder output (fib_engine/trendline.py)" — that builder
didn't exist until this function was added. fit_trendline() is what
turns a list of PivotPoint into the Trendline that PatternDetector.detect()
actually requires.

Usage:
    from src.waveconf.fib_engine.trendline import fit_trendline

    # Resistance: fit ONLY the high-type pivots in your window
    highs = [p for p in window if p.is_high()]
    resistance = fit_trendline(highs)

    # Support: fit ONLY the low-type pivots in the SAME window
    lows = [p for p in window if p.is_low()]
    support = fit_trendline(lows)

    pattern = PatternDetector().detect(resistance, support)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from src.waveconf.pivots.pivot_schema import PivotPoint


@dataclass
class Trendline:
    """
    Represents a fitted trendline over a set of pivot points.
    Generated using ordinary least squares linear regression of
    price against bar_index.
    """
    start_price: float
    end_price: float
    slope_pct_per_bar: float
    r_squared: float
    pivot_count: int

    def is_flat(self, threshold: float) -> bool:
        return abs(self.slope_pct_per_bar) <= threshold

    def is_rising(self, threshold: float) -> bool:
        return self.slope_pct_per_bar > threshold

    def is_falling(self, threshold: float) -> bool:
        return self.slope_pct_per_bar < -threshold


def fit_trendline(pivots: Sequence[PivotPoint]) -> Optional[Trendline]:
    """
    Fit a single trendline through a set of same-type pivots (all highs,
    or all lows -- do not mix, since the line is meant to represent one
    boundary of a channel/wedge/triangle, not a regression through both).

    Returns None if fewer than 2 pivots are given (a line needs 2+ points;
    PatternDetector additionally requires min_pivots_per_side from
    config/pattern_thresholds.yaml, but that threshold is PatternDetector's
    concern, not this function's -- this function only enforces the
    mathematical minimum).
    """
    n = len(pivots)
    if n < 2:
        return None

    ordered = sorted(pivots, key=lambda p: p.bar_index)
    xs = [p.bar_index for p in ordered]
    ys = [p.price for p in ordered]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)

    if ss_xx == 0:
        # All pivots on the same bar_index -- degenerate, can't fit a slope.
        return None

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    fitted = [slope * x + intercept for x in xs]
    ss_res = sum((y - f) ** 2 for y, f in zip(ys, fitted))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    start_price = slope * xs[0] + intercept
    end_price = slope * xs[-1] + intercept

    # Normalize slope to %-per-bar relative to mean price, so a $500/day
    # slope on a $20,000 BTC chart and a $5/day slope on a $200 chart are
    # comparable -- this is what lets flat_slope_threshold_pct_per_bar in
    # pattern_thresholds.yaml mean the same thing across price regimes.
    slope_pct_per_bar = (slope / y_mean) * 100 if y_mean != 0 else 0.0

    return Trendline(
        start_price=start_price,
        end_price=end_price,
        slope_pct_per_bar=slope_pct_per_bar,
        r_squared=max(0.0, min(1.0, r_squared)),
        pivot_count=n,
    )