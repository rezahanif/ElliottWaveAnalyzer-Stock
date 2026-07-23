"""
classifiers.py
--------------
Full v2.0 Elliott Wave taxonomy classifiers.

Two top-level classes:
    ImpulseClassifier   — identifies which of 6 impulse types a 5-wave
                          sequence matches (or rejects all)
    CorrectionClassifier— identifies which of 11 correction types a
                          sequence matches (or rejects all)

Both read tolerance and ratio bounds from:
    config/correction_rules.yaml
    config/completion_rates.yaml

Design:
  - Pure deterministic rule evaluation. No ML, no probability. 
  - Input: list of PivotPoint objects representing a candidate wave sequence.
  - Output: ClassificationResult (type label, confidence, flags, violations).
  - Hard rules return HARD_FAIL — no tolerance applied.
  - Soft rules return SOFT_FAIL — within ±tolerance is a warning, not rejection.
  - Classifiers are stateless — instantiate once, call repeatedly.

Usage:
    from src.waveconf.pivots.classifiers import ImpulseClassifier, CorrectionClassifier
    from src.waveconf.pivots.pivot_schema import PivotPoint

    impulse_clf    = ImpulseClassifier()
    correction_clf = CorrectionClassifier()

    result = impulse_clf.classify(pivots)      # 6 pivot points: start,1,2,3,4,5
    result = correction_clf.classify(pivots)   # 4 pivot points: A_start,A,B,C
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.waveconf.pivots.pivot_schema import PivotPoint, SwingType


# ─────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────

class FailSeverity(str, Enum):
    HARD = "HARD"   # inviolable rule violated — count is wrong
    SOFT = "SOFT"   # ratio outside expected range but within tolerance
    WARN = "WARN"   # unusual but not impossible


@dataclass
class RuleViolation:
    rule_id:    str
    severity:   FailSeverity
    message:    str
    actual:     float = 0.0
    expected:   str   = ""


@dataclass
class ClassificationResult:
    """
    Output of ImpulseClassifier.classify() or CorrectionClassifier.classify().
    """
    pattern_type:         str              # e.g. "expanded_flat", "standard_impulse"
    family:               str              # "impulse" / "flat" / "zigzag" / "triangle" / "combination"
    matched:              bool             # True = all hard rules passed
    confidence:           float            # 0.0–1.0, penalised per soft violation
    b_breach_expected:    bool  = False    # correction only
    diagonal_overlap_ok:  bool  = False    # impulse only
    truncation_flag:      bool  = False    # wave 5 truncation
    violations:           List[RuleViolation] = field(default_factory=list)
    notes:                List[str]            = field(default_factory=list)

    def hard_violations(self) -> List[RuleViolation]:
        return [v for v in self.violations if v.severity == FailSeverity.HARD]

    def soft_violations(self) -> List[RuleViolation]:
        return [v for v in self.violations if v.severity == FailSeverity.SOFT]

    def __repr__(self) -> str:
        status = "✅ MATCH" if self.matched else "❌ REJECT"
        return (
            f"ClassificationResult({status} | {self.pattern_type} "
            f"| confidence={self.confidence:.2f} "
            f"| hard_fails={len(self.hard_violations())} "
            f"| soft_fails={len(self.soft_violations())})"
        )


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _load_yaml(relative_path: str) -> dict:
    """Load a YAML config relative to the project root."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    full = os.path.join(base, relative_path)
    with open(full, "r") as f:
        return yaml.safe_load(f)


def _price_range(p_start: PivotPoint, p_end: PivotPoint) -> float:
    """Absolute price distance between two pivot prices."""
    return abs(p_end.price - p_start.price)


def _retrace_ratio(wave_start: float, wave_end: float, retrace_end: float) -> float:
    """
    Retracement ratio of a wave.
    wave_start → wave_end is the measured wave.
    retrace_end is where the retracement ends.
    Returns ratio relative to the wave's range.
    """
    wave_range = abs(wave_end - wave_start)
    if wave_range == 0:
        return 0.0
    retrace = abs(retrace_end - wave_end)
    return retrace / wave_range


def _extension_ratio(base_range: float, extended_range: float) -> float:
    if base_range == 0:
        return 0.0
    return extended_range / base_range


def _slope(x1: int, y1: float, x2: int, y2: float) -> float:
    dx = x2 - x1
    return (y2 - y1) / dx if dx != 0 else 0.0


def _within(value: float, low: float, high: float, tolerance: float) -> bool:
    """Soft check: value is within [low*(1-tol), high*(1+tol)]."""
    return (low * (1 - tolerance)) <= value <= (high * (1 + tolerance))


def _above(value: float, minimum: float, tolerance: float = 0.0) -> bool:
    return value >= minimum * (1 - tolerance)


def _below(value: float, maximum: float, tolerance: float = 0.0) -> bool:
    return value <= maximum * (1 + tolerance)


# ─────────────────────────────────────────────────────────────
# ImpulseClassifier
# ─────────────────────────────────────────────────────────────

class ImpulseClassifier:
    """
    Classifies a 5-wave impulse sequence into one of 6 types:
        standard_impulse
        impulse_w3_extension
        impulse_w5_extension
        impulse_w1_extension
        leading_diagonal
        ending_diagonal

    Input pivots: exactly 6 PivotPoint objects representing:
        [origin, w1_end, w2_end, w3_end, w4_end, w5_end]

    For a bullish impulse:
        origin  = swing low before wave 1
        w1_end  = swing high after wave 1 (HH)
        w2_end  = swing low after wave 2 (HL)
        w3_end  = swing high after wave 3 (HH, usually highest)
        w4_end  = swing low after wave 4 (HL, must not overlap w1)
        w5_end  = swing high after wave 5

    For a bearish impulse (inverted):
        All highs/lows are mirrored — the classifier handles both directions
        by working with absolute price ranges, not raw prices.
    """

    TYPES = [
        "standard_impulse",
        "impulse_w3_extension",
        "impulse_w5_extension",
        "impulse_w1_extension",
        "leading_diagonal",
        "ending_diagonal",
    ]

    def __init__(self, config_path: str = "config/correction_rules.yaml"):
        cfg = _load_yaml(config_path)
        self.tolerance = cfg.get("tolerance", 0.05)

    def classify(self, pivots: List[PivotPoint]) -> ClassificationResult:
        """
        Try all 6 impulse types. Return the matching type with the highest confidence.
        If no type matches, return a REJECT result with all violations from standard impulse.
        """
        if len(pivots) != 6:
            return ClassificationResult(
                pattern_type="invalid_input",
                family="impulse",
                matched=False,
                confidence=0.0,
                violations=[RuleViolation("input", FailSeverity.HARD,
                    f"ImpulseClassifier requires exactly 6 pivots, got {len(pivots)}")]
            )

        origin, w1, w2, w3, w4, w5 = pivots

        # Determine direction from origin → w1
        bullish = w1.price > origin.price

        # Compute wave price ranges
        r1 = abs(w1.price - origin.price)
        r2 = abs(w2.price - w1.price)
        r3 = abs(w3.price - w2.price)
        r4 = abs(w4.price - w3.price)
        r5 = abs(w5.price - w4.price)

        candidates = []
        for try_fn in [
            self._try_ending_diagonal,
            self._try_leading_diagonal,
            self._try_standard,
            self._try_w3_extension,
            self._try_w5_extension,
            self._try_w1_extension,
        ]:
            result = try_fn(pivots, bullish, r1, r2, r3, r4, r5)
            if result.matched:
                candidates.append(result)

        if candidates:
            # Sort by confidence descending. If equal, preserves the order of the loop.
            candidates.sort(key=lambda x: x.confidence, reverse=True)
            return candidates[0]

        # Nothing matched — collect all hard rules for the standard impulse
        # as the baseline rejection report
        return self._try_standard(pivots, bullish, r1, r2, r3, r4, r5)

    # ── Hard rules shared by all non-diagonal impulse types ──────────────

    def _hard_impulse_rules(
        self, pivots, bullish, r1, r2, r3, r4, r5
    ) -> List[RuleViolation]:
        origin, w1, w2, w3, w4, w5 = pivots
        violations = []
        tol = self.tolerance

        # HARD: Wave 2 never retraces more than 100% of Wave 1
        w2_retrace = r2 / r1 if r1 > 0 else 99
        if w2_retrace > 1.0 + tol:
            violations.append(RuleViolation(
                "w2_no_exceed_w1", FailSeverity.HARD,
                f"Wave 2 retraced {w2_retrace:.1%} of Wave 1 (max 100%)",
                actual=w2_retrace, expected="<= 1.000"
            ))

        # HARD: Wave 2 endpoint must not exceed origin (for bullish)
        if bullish and w2.price <= origin.price:
            violations.append(RuleViolation(
                "w2_above_origin", FailSeverity.HARD,
                "Wave 2 low breached Wave 1 origin — wave count invalid",
                actual=w2.price, expected=f"> {origin.price}"
            ))
        elif not bullish and w2.price >= origin.price:
            violations.append(RuleViolation(
                "w2_below_origin", FailSeverity.HARD,
                "Wave 2 high breached Wave 1 origin — wave count invalid",
                actual=w2.price, expected=f"< {origin.price}"
            ))

        # HARD: Wave 3 must not be the shortest among 1, 3, 5
        if r3 < r1 and r3 < r5:
            violations.append(RuleViolation(
                "w3_not_shortest", FailSeverity.HARD,
                f"Wave 3 ({r3:.2f}) is the shortest wave — hard violation",
                actual=r3, expected=f"not shorter than both {r1:.2f} and {r5:.2f}"
            ))

        # HARD: Wave 3 must exceed Wave 1's endpoint
        if bullish and w3.price <= w1.price:
            violations.append(RuleViolation(
                "w3_exceeds_w1", FailSeverity.HARD,
                "Wave 3 did not exceed Wave 1 high",
                actual=w3.price, expected=f"> {w1.price}"
            ))
        elif not bullish and w3.price >= w1.price:
            violations.append(RuleViolation(
                "w3_exceeds_w1", FailSeverity.HARD,
                "Wave 3 did not exceed Wave 1 low",
                actual=w3.price, expected=f"< {w1.price}"
            ))

        # HARD: Wave 4 must not overlap Wave 1 price territory (non-diagonal)
        if bullish and w4.price <= w1.price:
            violations.append(RuleViolation(
                "w4_no_overlap_w1", FailSeverity.HARD,
                f"Wave 4 low ({w4.price}) entered Wave 1 price territory (high={w1.price})",
                actual=w4.price, expected=f"> {w1.price}"
            ))
        elif not bullish and w4.price >= w1.price:
            violations.append(RuleViolation(
                "w4_no_overlap_w1", FailSeverity.HARD,
                f"Wave 4 high ({w4.price}) entered Wave 1 price territory (low={w1.price})",
                actual=w4.price, expected=f"< {w1.price}"
            ))

        return violations

    def _compute_confidence(self, violations: List[RuleViolation]) -> float:
        """Penalise 0.15 per soft violation, 0.00 if any hard violation."""
        if any(v.severity == FailSeverity.HARD for v in violations):
            return 0.0
        soft_count = sum(1 for v in violations if v.severity == FailSeverity.SOFT)
        return max(0.0, 1.0 - soft_count * 0.15)

    # ── Standard Impulse ─────────────────────────────────────────────────

    def _try_standard(self, pivots, bullish, r1, r2, r3, r4, r5) -> ClassificationResult:
        violations = self._hard_impulse_rules(pivots, bullish, r1, r2, r3, r4, r5)
        origin, w1, w2, w3, w4, w5 = pivots
        tol = self.tolerance

        # Soft: Wave 3 typically 1.0–1.618× Wave 1 for standard (not an extension)
        w3_ext = _extension_ratio(r1, r3)
        if w3_ext > 1.618 * (1 + tol):
            # Hard gate: this is an extension — standard impulse must not claim it
            violations.append(RuleViolation(
                "w3_ext_is_extension", FailSeverity.HARD,
                f"Wave 3 extension {w3_ext:.3f} > 1.618 — classify as impulse_w3_extension not standard",
                actual=w3_ext, expected="<= 1.618 for standard impulse"
            ))
        elif not _within(w3_ext, 1.000, 4.236, tol):
            violations.append(RuleViolation(
                "w3_ext_range", FailSeverity.SOFT,
                f"Wave 3 extension {w3_ext:.3f} outside typical [1.0, 4.236]",
                actual=w3_ext, expected="1.000 – 4.236"
            ))

        # Soft: Wave 4 retraces 23.6–50% of Wave 3
        w4_ret = _retrace_ratio(w2.price, w3.price, w4.price)
        if not _within(w4_ret, 0.236, 0.500, tol):
            violations.append(RuleViolation(
                "w4_retrace", FailSeverity.SOFT,
                f"Wave 4 retraced {w4_ret:.1%} of Wave 3 (typical 23.6–50%)",
                actual=w4_ret, expected="0.236 – 0.500"
            ))

        # Soft: Wave 5 typically equals Wave 1 or 0.382–1.618× Wave 1
        w5_ext = _extension_ratio(r1, r5)
        if not _within(w5_ext, 0.382, 1.618, tol):
            violations.append(RuleViolation(
                "w5_ext_range", FailSeverity.SOFT,
                f"Wave 5 extension {w5_ext:.3f} outside typical [0.382, 1.618]",
                actual=w5_ext, expected="0.382 – 1.618"
            ))

        # Truncation check: Wave 5 fails to exceed Wave 3
        truncation = (bullish and w5.price < w3.price) or (not bullish and w5.price > w3.price)

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        conf = self._compute_confidence(violations)

        return ClassificationResult(
            pattern_type="standard_impulse",
            family="impulse",
            matched=not hard_fail,
            confidence=conf,
            truncation_flag=truncation,
            violations=violations,
            notes=["TRUNCATION: Wave 5 failed to exceed Wave 3"] if truncation else [],
        )

    # ── Wave 3 Extension ─────────────────────────────────────────────────

    def _try_w3_extension(self, pivots, bullish, r1, r2, r3, r4, r5) -> ClassificationResult:
        violations = self._hard_impulse_rules(pivots, bullish, r1, r2, r3, r4, r5)
        tol = self.tolerance

        # Diagnostic: Wave 3 must be > 1.618× Wave 1 to qualify as extension
        w3_ext = _extension_ratio(r1, r3)
        if not _above(w3_ext, 1.618, tol):
            violations.append(RuleViolation(
                "w3_ext_min", FailSeverity.HARD,
                f"Wave 3 extension {w3_ext:.3f} < 1.618 — not a Wave 3 extension",
                actual=w3_ext, expected=">= 1.618"
            ))

        # Soft: Wave 3 extension often subdivides into 9 waves (momentum surge)
        # (cannot check subdivision count without sub-pivot data — note only)
        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        conf = self._compute_confidence(violations)

        return ClassificationResult(
            pattern_type="impulse_w3_extension",
            family="impulse",
            matched=not hard_fail,
            confidence=conf,
            violations=violations,
            notes=["Wave 3 extension: explosive momentum, may subdivide into 9 sub-waves"],
        )

    # ── Wave 5 Extension ─────────────────────────────────────────────────

    def _try_w5_extension(self, pivots, bullish, r1, r2, r3, r4, r5) -> ClassificationResult:
        violations = self._hard_impulse_rules(pivots, bullish, r1, r2, r3, r4, r5)
        tol = self.tolerance

        # Wave 5 must exceed Wave 3 in length AND Wave 3 ≈ Wave 1
        w5_ext = _extension_ratio(r1, r5)
        if not _above(w5_ext, 1.618, tol):
            violations.append(RuleViolation(
                "w5_ext_min", FailSeverity.HARD,
                f"Wave 5 extension {w5_ext:.3f} < 1.618 — not a Wave 5 extension",
                actual=w5_ext, expected=">= 1.618"
            ))

        # Wave 3 ≈ Wave 1 in a W5 extension
        w3_vs_w1 = _extension_ratio(r1, r3)
        if not _within(w3_vs_w1, 0.618, 1.382, tol):
            violations.append(RuleViolation(
                "w3_approx_w1", FailSeverity.SOFT,
                f"Wave 3 / Wave 1 = {w3_vs_w1:.3f}, expected ≈1.0 for W5 extension",
                actual=w3_vs_w1, expected="0.618 – 1.382"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        conf = self._compute_confidence(violations)

        return ClassificationResult(
            pattern_type="impulse_w5_extension",
            family="impulse",
            matched=not hard_fail,
            confidence=conf,
            violations=violations,
            notes=["Wave 5 extension: common at market tops, often shows bearish divergence"],
        )

    # ── Wave 1 Extension ─────────────────────────────────────────────────

    def _try_w1_extension(self, pivots, bullish, r1, r2, r3, r4, r5) -> ClassificationResult:
        violations = self._hard_impulse_rules(pivots, bullish, r1, r2, r3, r4, r5)
        tol = self.tolerance

        # Wave 1 must be longest
        w1_ext_vs_3 = _extension_ratio(r3, r1)
        w1_ext_vs_5 = _extension_ratio(r5, r1)
        if not (_above(w1_ext_vs_3, 1.618, tol) and _above(w1_ext_vs_5, 1.618, tol)):
            violations.append(RuleViolation(
                "w1_longest", FailSeverity.HARD,
                f"Wave 1 is not the longest wave (vs W3: {w1_ext_vs_3:.3f}, vs W5: {w1_ext_vs_5:.3f})",
                actual=min(w1_ext_vs_3, w1_ext_vs_5), expected=">= 1.618 vs both W3 and W5"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        conf = self._compute_confidence(violations)

        return ClassificationResult(
            pattern_type="impulse_w1_extension",
            family="impulse",
            matched=not hard_fail,
            confidence=conf,
            violations=violations,
            notes=["Wave 1 extension: less common, first waves of major new trends"],
        )

    # ── Leading Diagonal ─────────────────────────────────────────────────

    def _try_leading_diagonal(self, pivots, bullish, r1, r2, r3, r4, r5) -> ClassificationResult:
        """
        Leading Diagonal: Wave 1 or Wave A position only.
        Wave 4 MUST overlap Wave 1 territory.
        Trendlines 1-3 and 2-4 must converge.
        Sub-waves are 3-3-3-3-3.
        """
        origin, w1, w2, w3, w4, w5 = pivots
        violations = []
        tol = self.tolerance

        # HARD: Wave 2 must not exceed origin
        if bullish and w2.price <= origin.price:
            violations.append(RuleViolation(
                "ld_w2_origin", FailSeverity.HARD,
                "Wave 2 breached Wave 1 origin",
                actual=w2.price, expected=f"> {origin.price}"
            ))

        # HARD: Wave 3 must exceed Wave 1 end
        if bullish and w3.price <= w1.price:
            violations.append(RuleViolation(
                "ld_w3_exceeds_w1", FailSeverity.HARD,
                "Wave 3 did not exceed Wave 1 high",
                actual=w3.price, expected=f"> {w1.price}"
            ))

        # HARD: Wave 3 not shortest
        if r3 < r1 and r3 < r5:
            violations.append(RuleViolation(
                "ld_w3_not_shortest", FailSeverity.HARD,
                "Wave 3 is shortest — invalid for leading diagonal",
                actual=r3, expected="not shortest"
            ))

        # KEY DIAGNOSTIC: Wave 4 MUST overlap Wave 1 (opposite of standard impulse)
        if bullish and w4.price >= w1.price:
            violations.append(RuleViolation(
                "ld_w4_overlap_required", FailSeverity.HARD,
                f"Wave 4 ({w4.price}) must overlap Wave 1 territory ({w1.price}) for a leading diagonal",
                actual=w4.price, expected=f"< {w1.price}"
            ))
        elif not bullish and w4.price <= w1.price:
            violations.append(RuleViolation(
                "ld_w4_overlap_required", FailSeverity.HARD,
                f"Wave 4 ({w4.price}) must overlap Wave 1 territory ({w1.price}) for a leading diagonal",
                actual=w4.price, expected=f"> {w1.price}"
            ))

        # HARD: Trendlines must converge (slope of 1-3 > slope of 2-4 for bullish)
        slope_13 = _slope(w1.bar_index, w1.price, w3.bar_index, w3.price)
        slope_24 = _slope(w2.bar_index, w2.price, w4.bar_index, w4.price)
        if bullish:
            # Both positive and 1-3 slope > 2-4 slope means converging
            trendlines_converge = slope_13 > 0 and slope_24 > 0 and slope_13 > slope_24
        else:
            trendlines_converge = slope_13 < 0 and slope_24 < 0 and slope_13 < slope_24
        if not trendlines_converge:
            violations.append(RuleViolation(
                "ld_trendlines_converge", FailSeverity.HARD,
                f"Trendlines 1-3 (slope={slope_13:.4f}) and 2-4 (slope={slope_24:.4f}) do not converge",
                actual=slope_13 - slope_24, expected="converging"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        conf = self._compute_confidence(violations)

        return ClassificationResult(
            pattern_type="leading_diagonal",
            family="impulse",
            matched=not hard_fail,
            confidence=conf,
            diagonal_overlap_ok=True,
            violations=violations,
            notes=[
                "Leading diagonal: wave 4/1 overlap is VALID and EXPECTED here",
                "Appears as Wave 1 or Wave A only",
                "Sub-waves should be 3-3-3-3-3 (check sub-pivot count if available)",
            ],
        )

    # ── Ending Diagonal ──────────────────────────────────────────────────

    def _try_ending_diagonal(self, pivots, bullish, r1, r2, r3, r4, r5) -> ClassificationResult:
        """
        Ending Diagonal: Wave 5 or Wave C position only.
        Wave 4 ALWAYS overlaps Wave 1. All sub-waves are 3-wave structures.
        Signals exhaustion — reversal imminent after Wave 5.
        """
        origin, w1, w2, w3, w4, w5 = pivots
        violations = []
        tol = self.tolerance

        # HARD: Wave 4 must overlap Wave 1 (same as leading diagonal, same check)
        if bullish and w4.price >= w1.price:
            violations.append(RuleViolation(
                "ed_w4_overlap_required", FailSeverity.HARD,
                f"Wave 4 ({w4.price}) must overlap Wave 1 ({w1.price}) in ending diagonal",
                actual=w4.price, expected=f"< {w1.price}"
            ))
        elif not bullish and w4.price <= w1.price:
            violations.append(RuleViolation(
                "ed_w4_overlap_required", FailSeverity.HARD,
                f"Wave 4 ({w4.price}) must overlap Wave 1 ({w1.price}) in ending diagonal",
                actual=w4.price, expected=f"> {w1.price}"
            ))

        # HARD: Wave 3 must not be shortest
        if r3 < r1 and r3 < r5:
            violations.append(RuleViolation(
                "ed_w3_not_shortest", FailSeverity.HARD,
                "Wave 3 is shortest — invalid for ending diagonal",
                actual=r3, expected="not shortest"
            ))

        # HARD: Trendlines must converge (wedge shape)
        slope_13 = _slope(w1.bar_index, w1.price, w3.bar_index, w3.price)
        slope_24 = _slope(w2.bar_index, w2.price, w4.bar_index, w4.price)
        if bullish:
            trendlines_converge = slope_13 > 0 and slope_24 > 0 and slope_13 > slope_24
        else:
            trendlines_converge = slope_13 < 0 and slope_24 < 0 and slope_13 < slope_24
        if not trendlines_converge:
            violations.append(RuleViolation(
                "ed_trendlines_converge", FailSeverity.HARD,
                f"Ending diagonal trendlines do not converge (1-3={slope_13:.4f}, 2-4={slope_24:.4f})",
                actual=slope_13, expected="converging"
            ))

        # Soft: each wave smaller than prior in ending diagonal
        if r3 > r1:
            violations.append(RuleViolation(
                "ed_wave_shrink", FailSeverity.SOFT,
                f"Wave 3 ({r3:.2f}) > Wave 1 ({r1:.2f}) — ending diagonal waves typically shrink",
                actual=r3, expected=f"< {r1:.2f}"
            ))
        if r5 > r3:
            violations.append(RuleViolation(
                "ed_w5_shrink", FailSeverity.SOFT,
                f"Wave 5 ({r5:.2f}) > Wave 3 ({r3:.2f}) — ending diagonal waves typically shrink",
                actual=r5, expected=f"< {r3:.2f}"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        conf = self._compute_confidence(violations)

        return ClassificationResult(
            pattern_type="ending_diagonal",
            family="impulse",
            matched=not hard_fail,
            confidence=conf,
            diagonal_overlap_ok=True,
            violations=violations,
            notes=[
                "Ending diagonal: wave 4/1 overlap is VALID and REQUIRED",
                "Appears as Wave 5 or Wave C only",
                "Reversal imminent after Wave 5 completes",
                "Sub-waves should be 3-3-3-3-3 (check sub-pivot count if available)",
            ],
        )


# ─────────────────────────────────────────────────────────────
# CorrectionClassifier
# ─────────────────────────────────────────────────────────────

class CorrectionClassifier:
    """
    Classifies a corrective wave sequence into one of 11 types:

    Flat family (3-3-5):
        regular_flat, expanded_flat, running_flat

    Zigzag family (5-3-5 or linked):
        single_zigzag, double_zigzag, triple_zigzag

    Triangle family (3-3-3-3-3):
        contracting_symmetrical, contracting_ascending,
        contracting_descending, expanding_triangle

    Combination:
        double_three, triple_three

    Input pivots for ABC corrections (flat/zigzag):
        [correction_origin, A_end, B_end, C_end]
        4 PivotPoint objects.

    Input pivots for triangles (ABCDE):
        [triangle_origin, A_end, B_end, C_end, D_end, E_end]
        6 PivotPoint objects.

    For combination structures pass [W_start, W_end, X_end, Y_end]
    or [W_start, W_end, X_end, Y_end, X2_end, Z_end].
    """

    def __init__(self, config_path: str = "config/correction_rules.yaml"):
        cfg = _load_yaml(config_path)
        self.tolerance   = cfg.get("tolerance", 0.05)
        self.flat_rules  = cfg.get("flat_family", {})
        self.zz_rules    = cfg.get("zigzag_family", {})
        self.tri_rules   = cfg.get("triangle_family", {})
        self.combo_rules = cfg.get("combination_family", {})

    def classify(self, pivots: List[PivotPoint]) -> ClassificationResult:
        """
        Try all correction families in order. Return the best match.
        Priority: flat → zigzag → triangle → combination.
        (flat and zigzag share 4-pivot ABC structure;
         triangle uses 6-pivot ABCDE; combination uses 4 or 6 pivots)
        """
        n = len(pivots)

        if n == 4:
            # Pre-filter: compute B retrace and B-exceeds-origin to route
            # to the right family first before trying all types.
            # This prevents regular_flat swallowing zigzag counts and
            # expanded_flat swallowing running_flat counts.
            origin_p, A_p, B_p, C_p = pivots
            r_A = abs(A_p.price - origin_p.price)
            r_B = abs(B_p.price - A_p.price)
            b_ret_pre = r_B / r_A if r_A > 0 else 0.0
            downward_pre = A_p.price < origin_p.price
            b_exc_pre = (B_p.price > origin_p.price) if downward_pre else (B_p.price < origin_p.price)
            r_C = abs(C_p.price - B_p.price)
            c_vs_a_pre = r_C / r_A if r_A > 0 else 0.0

            # Route 1: B does NOT exceed origin + B retrace <= 78.6%
            #          → zigzag family first, then regular flat as fallback
            if not b_exc_pre and b_ret_pre <= (0.786 + self.tolerance):
                priority = [
                    self._try_single_zigzag,
                    self._try_double_zigzag,
                    self._try_regular_flat,
                    self._try_double_three,
                ]

            # Route 2: B DOES exceed origin + C is truncated (< A in length)
            #          → running_flat first (most specific), then expanded_flat
            elif b_exc_pre and c_vs_a_pre < 1.0:
                priority = [
                    self._try_running_flat,
                    self._try_expanded_flat,
                    self._try_double_three,
                ]

            # Route 3: B DOES exceed origin + C >= A
            #          → expanded_flat
            elif b_exc_pre:
                priority = [
                    self._try_expanded_flat,
                    self._try_running_flat,
                    self._try_double_three,
                ]

            # Route 4: B near origin (81–100% retrace) — regular flat territory
            else:
                priority = [
                    self._try_regular_flat,
                    self._try_single_zigzag,
                    self._try_double_zigzag,
                    self._try_double_three,
                ]

            candidates = []
            for try_fn in priority:
                result = try_fn(pivots)
                if result.matched:
                    candidates.append(result)

            if candidates:
                candidates.sort(key=lambda x: x.confidence, reverse=True)
                return candidates[0]

            # Exhausted priority — return baseline rejection
            return self._try_regular_flat(pivots)

        elif n == 6:
            # Triangle (ABCDE) or triple three (W-X-Y-X-Z)
            candidates = []
            for try_fn in [
                self._try_contracting_ascending,
                self._try_contracting_descending,
                self._try_contracting_symmetrical,
                self._try_expanding_triangle,
                self._try_triple_three,
            ]:
                result = try_fn(pivots)
                if result.matched:
                    candidates.append(result)

            if candidates:
                candidates.sort(key=lambda x: x.confidence, reverse=True)
                return candidates[0]

            return self._try_contracting_symmetrical(pivots)

        else:
            return ClassificationResult(
                pattern_type="invalid_input",
                family="correction",
                matched=False,
                confidence=0.0,
                violations=[RuleViolation("input", FailSeverity.HARD,
                    f"CorrectionClassifier requires 4 or 6 pivots, got {n}")]
            )

    def _compute_confidence(self, violations: List[RuleViolation]) -> float:
        if any(v.severity == FailSeverity.HARD for v in violations):
            return 0.0
        soft_count = sum(1 for v in violations if v.severity == FailSeverity.SOFT)
        return max(0.0, 1.0 - soft_count * 0.15)

    # ── Shared ABC geometry ───────────────────────────────────────────────

    def _abc_geometry(self, pivots):
        """Extract A, B, C wave ranges and key ratios from 4-pivot input."""
        origin, A, B, C = pivots
        r_A = abs(A.price - origin.price)
        r_B = abs(B.price - A.price)
        r_C = abs(C.price - B.price)
        # Downward correction: origin high → A low → B high → C low
        # B retrace ratio relative to A
        b_retrace = r_B / r_A if r_A > 0 else 0.0
        # Does B exceed A's origin?
        # For a downward correction: origin is HIGH, A is LOW, B is HIGH
        # B exceeds A's origin means B.price > origin.price
        downward = A.price < origin.price  # correction direction
        if downward:
            b_exceeds_origin = B.price > origin.price
        else:
            b_exceeds_origin = B.price < origin.price
        c_vs_a = r_C / r_A if r_A > 0 else 0.0
        return r_A, r_B, r_C, b_retrace, b_exceeds_origin, c_vs_a, downward

    # ── Flat family ───────────────────────────────────────────────────────

    def _try_regular_flat(self, pivots) -> ClassificationResult:
        origin, A, B, C = pivots
        r_A, r_B, r_C, b_ret, b_exc, c_vs_a, downward = self._abc_geometry(pivots)
        violations = []
        tol = self.tolerance

        # HARD: B must NOT significantly exceed A's origin
        if b_exc:
            violations.append(RuleViolation(
                "reg_flat_b_no_exceed", FailSeverity.HARD,
                f"B ({B.price}) exceeded A's origin ({origin.price}) — not a regular flat",
                actual=B.price, expected=f"<= {origin.price}"
            ))

        # Soft: B retraces 81–100% of A
        if not _within(b_ret, 0.810, 1.000, tol):
            violations.append(RuleViolation(
                "reg_flat_b_retrace", FailSeverity.SOFT,
                f"B retraced {b_ret:.1%} of A (regular flat expects 81–100%)",
                actual=b_ret, expected="0.810 – 1.000"
            ))

        # Soft: C ≈ A in length
        if not _within(c_vs_a, 0.900, 1.100, tol):
            violations.append(RuleViolation(
                "reg_flat_c_vs_a", FailSeverity.SOFT,
                f"C / A = {c_vs_a:.3f} (regular flat expects ≈1.0)",
                actual=c_vs_a, expected="0.900 – 1.100"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="regular_flat",
            family="flat",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=False,
            violations=violations,
        )

    def _try_expanded_flat(self, pivots) -> ClassificationResult:
        origin, A, B, C = pivots
        r_A, r_B, r_C, b_ret, b_exc, c_vs_a, downward = self._abc_geometry(pivots)
        violations = []
        tol = self.tolerance

        # HARD: B must exceed A's origin
        if not b_exc:
            violations.append(RuleViolation(
                "exp_flat_b_must_exceed", FailSeverity.HARD,
                f"B ({B.price}) did not exceed A's origin ({origin.price}) — not an expanded flat",
                actual=B.price, expected=f"> {origin.price}"
            ))

        # Soft: B retraces 100–138.2% of A
        if not _within(b_ret, 1.000, 1.382, tol):
            violations.append(RuleViolation(
                "exp_flat_b_retrace", FailSeverity.SOFT,
                f"B retraced {b_ret:.1%} of A (expanded flat expects 100–138.2%)",
                actual=b_ret, expected="1.000 – 1.382"
            ))

        # Soft: C extends 123.6–161.8% of A
        if not _within(c_vs_a, 1.236, 1.618, tol):
            violations.append(RuleViolation(
                "exp_flat_c_vs_a", FailSeverity.SOFT,
                f"C / A = {c_vs_a:.3f} (expanded flat expects 1.236 – 1.618)",
                actual=c_vs_a, expected="1.236 – 1.618"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="expanded_flat",
            family="flat",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=True,
            violations=violations,
            notes=["B-breach is EXPECTED — not an invalidation. Flag as SWEEP if fast reversal."],
        )

    def _try_running_flat(self, pivots) -> ClassificationResult:
        origin, A, B, C = pivots
        r_A, r_B, r_C, b_ret, b_exc, c_vs_a, downward = self._abc_geometry(pivots)
        violations = []
        tol = self.tolerance

        # HARD: B must significantly exceed A's origin (>110%)
        if not b_exc:
            violations.append(RuleViolation(
                "run_flat_b_must_exceed", FailSeverity.HARD,
                f"B did not exceed A's origin — not a running flat",
                actual=B.price, expected=f"> {origin.price}"
            ))
        elif not _above(b_ret, 1.100, tol):
            violations.append(RuleViolation(
                "run_flat_b_min", FailSeverity.HARD,
                f"B retraced only {b_ret:.1%} of A (running flat requires > 110%)",
                actual=b_ret, expected=">= 1.100"
            ))

        # HARD: C must NOT reach A's end (truncated)
        if c_vs_a >= 1.000:
            violations.append(RuleViolation(
                "run_flat_c_truncated", FailSeverity.HARD,
                f"C / A = {c_vs_a:.3f} — running flat requires C < A (truncated)",
                actual=c_vs_a, expected="< 1.000"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="running_flat",
            family="flat",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=True,
            violations=violations,
            notes=[
                "Running flat: C truncation signals very strong underlying demand.",
                "Rare — confirm with higher-degree structure before accepting.",
            ],
        )

    # ── Zigzag family ─────────────────────────────────────────────────────

    def _try_single_zigzag(self, pivots) -> ClassificationResult:
        origin, A, B, C = pivots
        r_A, r_B, r_C, b_ret, b_exc, c_vs_a, downward = self._abc_geometry(pivots)
        violations = []
        tol = self.tolerance

        # HARD: B must NOT exceed A's origin
        if b_exc:
            violations.append(RuleViolation(
                "zz_b_no_exceed", FailSeverity.HARD,
                f"B ({B.price}) exceeded A's origin — not a zigzag (reclassify as flat)",
                actual=B.price, expected=f"opposite side of {origin.price}"
            ))

        # HARD: B retraces at most 78.6% of A
        if b_ret > 0.786 + tol:
            violations.append(RuleViolation(
                "zz_b_cap", FailSeverity.HARD,
                f"B retraced {b_ret:.1%} of A (zigzag max 78.6%)",
                actual=b_ret, expected="<= 0.786"
            ))

        # Soft: B retraces 38.2–78.6% of A
        if not _within(b_ret, 0.382, 0.786, tol):
            violations.append(RuleViolation(
                "zz_b_retrace", FailSeverity.SOFT,
                f"B retraced {b_ret:.1%} of A (typical 38.2–78.6%)",
                actual=b_ret, expected="0.382 – 0.786"
            ))

        # Soft: C typically equals A or 0.618×/1.618× A
        if not _within(c_vs_a, 0.618, 1.618, tol):
            violations.append(RuleViolation(
                "zz_c_vs_a", FailSeverity.SOFT,
                f"C / A = {c_vs_a:.3f} (typical 0.618 – 1.618 for zigzag)",
                actual=c_vs_a, expected="0.618 – 1.618"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="single_zigzag",
            family="zigzag",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=False,
            violations=violations,
            notes=["Sharpest correction type. B rarely exceeds 78.6% of A."],
        )

    def _try_double_zigzag(self, pivots) -> ClassificationResult:
        """
        Double zigzag W-X-Y passed as [W_start, W_end, X_end, Y_end].
        The X wave is the connector (3-wave corrective).
        W and Y are each individual zigzags (5-3-5).
        """
        origin, W_end, X_end, Y_end = pivots
        violations = []
        tol = self.tolerance

        r_W = abs(W_end.price - origin.price)
        r_X = abs(X_end.price - W_end.price)
        r_Y = abs(Y_end.price - X_end.price)

        x_ret = r_X / r_W if r_W > 0 else 0.0
        y_vs_w = r_Y / r_W if r_W > 0 else 0.0

        # Soft: X retraces 38.2–78.6% of W
        if not _within(x_ret, 0.382, 0.786, tol):
            violations.append(RuleViolation(
                "dz_x_retrace", FailSeverity.SOFT,
                f"X retraced {x_ret:.1%} of W (typical 38.2–78.6%)",
                actual=x_ret, expected="0.382 – 0.786"
            ))

        # Soft: Y ≈ W or 0.618× W
        if not _within(y_vs_w, 0.618, 1.000, tol):
            violations.append(RuleViolation(
                "dz_y_vs_w", FailSeverity.SOFT,
                f"Y / W = {y_vs_w:.3f} (typical 0.618 – 1.0)",
                actual=y_vs_w, expected="0.618 – 1.000"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="double_zigzag",
            family="zigzag",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=False,
            violations=violations,
            notes=["Often mistaken for an impulse — verify sub-wave counts."],
        )

    def _try_triple_zigzag(self, pivots) -> ClassificationResult:
        """Triple zigzag — rare. Treated with low confidence by default."""
        violations = [RuleViolation(
            "triple_zz_rare", FailSeverity.WARN,
            "Triple zigzag is extremely rare. Re-examine count before accepting.",
            actual=0, expected="n/a"
        )]
        # Minimal geometric check only — same X retrace logic as double
        return ClassificationResult(
            pattern_type="triple_zigzag",
            family="zigzag",
            matched=True,  # passes with warning only
            confidence=0.30,  # low confidence cap for rare structure
            b_breach_expected=False,
            violations=violations,
            notes=["Triple zigzag: extremely rare. Exhaust all other counts first."],
        )

    # ── Triangle family ───────────────────────────────────────────────────

    def _try_contracting_symmetrical(self, pivots) -> ClassificationResult:
        """
        Contracting symmetrical triangle — 6 pivots: [origin, A, B, C, D, E]
        Both trendlines converge. Each leg smaller than prior.
        """
        origin, A, B, C, D, E = pivots
        violations = []
        tol = self.tolerance

        r_A = abs(A.price - origin.price)
        r_B = abs(B.price - A.price)
        r_C = abs(C.price - B.price)
        r_D = abs(D.price - C.price)
        r_E = abs(E.price - D.price)

        # Each successive leg must be smaller
        for (label, prev, curr) in [("B<A", r_A, r_B), ("C<B", r_B, r_C),
                                      ("D<C", r_C, r_D), ("E<D", r_D, r_E)]:
            if curr >= prev * (1 + tol):
                violations.append(RuleViolation(
                    f"sym_tri_{label}", FailSeverity.HARD,
                    f"Triangle leg {label}: {curr:.2f} >= {prev:.2f} — not contracting",
                    actual=curr, expected=f"< {prev:.2f}"
                ))

        # Trendline convergence: slope of highs (A, C, E) and lows (B, D) must converge
        # For a downward A, highs are B/D and lows are A/C/E
        slope_highs = _slope(B.bar_index, B.price, D.bar_index, D.price)
        slope_lows  = _slope(A.bar_index, A.price, C.bar_index, C.price)
        # For symmetrical: both slopes should move toward each other
        if not ((slope_highs < 0 and slope_lows > 0) or
                (slope_highs > 0 and slope_lows < 0)):
            violations.append(RuleViolation(
                "sym_tri_convergence", FailSeverity.SOFT,
                f"Trendlines may not be converging (high slope={slope_highs:.4f}, low slope={slope_lows:.4f})",
                actual=slope_highs, expected="opposite signs, converging"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="contracting_symmetrical",
            family="triangle",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=False,
            violations=violations,
            notes=["Breakout expected in direction of prior trend. Volume contracts during formation."],
        )

    def _try_contracting_ascending(self, pivots) -> ClassificationResult:
        origin, A, B, C, D, E = pivots
        violations = []
        tol = self.tolerance

        # Flat upper resistance: B and D highs should be approximately equal
        b_d_diff = abs(B.price - D.price) / max(B.price, D.price)
        if b_d_diff > 0.03:  # 3% tolerance on flat resistance
            violations.append(RuleViolation(
                "asc_tri_flat_top", FailSeverity.SOFT,
                f"Upper resistance not flat: B={B.price}, D={D.price} ({b_d_diff:.1%} apart)",
                actual=b_d_diff, expected="< 3%"
            ))

        # Rising lower support: A.price < C.price
        if A.price >= C.price:
            violations.append(RuleViolation(
                "asc_tri_rising_support", FailSeverity.HARD,
                f"Lower support not rising: A={A.price} >= C={C.price}",
                actual=A.price, expected=f"< {C.price}"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="contracting_ascending",
            family="triangle",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=False,
            violations=violations,
            notes=["Bullish bias. Buyers absorbing at flat resistance."],
        )

    def _try_contracting_descending(self, pivots) -> ClassificationResult:
        origin, A, B, C, D, E = pivots
        violations = []
        tol = self.tolerance

        # Flat lower support: A and C lows approximately equal
        a_c_diff = abs(A.price - C.price) / max(A.price, C.price)
        if a_c_diff > 0.03:
            violations.append(RuleViolation(
                "desc_tri_flat_bottom", FailSeverity.SOFT,
                f"Lower support not flat: A={A.price}, C={C.price} ({a_c_diff:.1%} apart)",
                actual=a_c_diff, expected="< 3%"
            ))

        # Declining upper resistance: B.price > D.price
        if B.price <= D.price:
            violations.append(RuleViolation(
                "desc_tri_declining_resistance", FailSeverity.HARD,
                f"Upper resistance not declining: B={B.price} <= D={D.price}",
                actual=B.price, expected=f"> {D.price}"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="contracting_descending",
            family="triangle",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=False,
            violations=violations,
            notes=["Bearish bias. Sellers absorbing at declining resistance."],
        )

    def _try_expanding_triangle(self, pivots) -> ClassificationResult:
        origin, A, B, C, D, E = pivots
        violations = []
        tol = self.tolerance

        r_A = abs(A.price - origin.price)
        r_B = abs(B.price - A.price)
        r_C = abs(C.price - B.price)
        r_D = abs(D.price - C.price)
        r_E = abs(E.price - D.price)

        # Each leg must be LARGER than prior (expanding)
        for (label, prev, curr) in [("B>A", r_A, r_B), ("C>B", r_B, r_C),
                                      ("D>C", r_C, r_D), ("E>D", r_D, r_E)]:
            if curr <= prev * (1 - tol):
                violations.append(RuleViolation(
                    f"exp_tri_{label}", FailSeverity.HARD,
                    f"Expanding triangle leg {label}: {curr:.2f} <= {prev:.2f} — not expanding",
                    actual=curr, expected=f"> {prev:.2f}"
                ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="expanding_triangle",
            family="triangle",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=False,
            violations=violations,
            notes=["Rare. Typically appears in Wave B positions. Signals major indecision."],
        )

    # ── Combination family ────────────────────────────────────────────────

    def _try_double_three(self, pivots) -> ClassificationResult:
        """W-X-Y passed as [W_start, W_end, X_end, Y_end]."""
        origin, W_end, X_end, Y_end = pivots
        violations = []
        tol = self.tolerance

        r_W = abs(W_end.price - origin.price)
        r_X = abs(X_end.price - W_end.price)
        r_Y = abs(Y_end.price - X_end.price)

        x_ret = r_X / r_W if r_W > 0 else 0.0

        if not _within(x_ret, 0.382, 0.786, tol):
            violations.append(RuleViolation(
                "d3_x_retrace", FailSeverity.SOFT,
                f"X connector retraced {x_ret:.1%} of W (typical 38.2–78.6%)",
                actual=x_ret, expected="0.382 – 0.786"
            ))

        hard_fail = any(v.severity == FailSeverity.HARD for v in violations)
        return ClassificationResult(
            pattern_type="double_three",
            family="combination",
            matched=not hard_fail,
            confidence=self._compute_confidence(violations),
            b_breach_expected=False,
            violations=violations,
            notes=[
                "W and Y are any simple correction (flat/zigzag/triangle).",
                "Y cannot be same type as W if W is a zigzag.",
                "b_breach_expected inherited from the Y wave type.",
            ],
        )

    def _try_triple_three(self, pivots) -> ClassificationResult:
        """W-X-Y-X-Z passed as [W_start, W_end, X_end, Y_end, X2_end, Z_end]."""
        violations = [RuleViolation(
            "triple_three_rare", FailSeverity.WARN,
            "Triple three is very rare. Exhaust double_three and triangle first.",
            actual=0, expected="n/a"
        )]
        return ClassificationResult(
            pattern_type="triple_three",
            family="combination",
            matched=True,
            confidence=0.25,
            b_breach_expected=False,
            violations=violations,
            notes=["Triple three: extremely time-consuming correction. Very rare."],
        )