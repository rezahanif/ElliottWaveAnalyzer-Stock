"""
scorer.py
---------
ConfluenceChecker — pipeline step [5].

Compares TFT quantile bands against Fibonacci target zones,
applies economic calendar adjustments, and computes scenario probabilities.

Signal tiers:
  AGGRESSIVE — original loose criteria, captures more signals with MFE potential.
  SELECTIVE  — tightened criteria for highest-conviction entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import List, Optional

from src.waveconf.fib_engine.fibonacci import ClusterResult
from src.waveconf.ingestion.economic_calender import CalendarContext, EconomicCalendarEngine
from src.waveconf.confluence.cluster_check import is_confluent, compute_probability_mass
from src.waveconf.confluence.entry_plan import generate_entry_zones


class SignalTier(str, Enum):
    """Signal conviction tier."""
    SELECTIVE  = "selective"   # Tight: 1.5% overlap, prob >= 15%, horizon <= 14d
    AGGRESSIVE = "aggressive"  # Original: 2.0% overlap, no prob gate


# ── Tier thresholds (centralised for easy tuning) ─────────────────────
SELECTIVE_CLUSTER_TOLERANCE_PCT  = 1.5   # q50 must be within ±1.5% of cluster
SELECTIVE_MIN_COMBINED_PROB      = 0.15  # at least 15% probability mass overlap
SELECTIVE_MAX_HORIZON_DAYS       = 14    # prefer shorter horizons for path safety
SELECTIVE_MIN_STRENGTH            = 0.50  # minimum adjusted strength

AGGRESSIVE_CLUSTER_TOLERANCE_PCT = 2.0   # original 2% tolerance


@dataclass
class TFTPrediction:
    horizon_days: int   # e.g., 7, 14, 30, 60
    q10: float
    q50: float
    q90: float


@dataclass
class ConfluenceReport:
    as_of: date
    confluence_valid: bool
    cluster_valid: bool
    proximity_pct: float
    raw_strength: float
    adjusted_strength: float

    # Scenario A
    scenario_a_target: float
    scenario_a_lower: float
    scenario_a_upper: float
    scenario_a_prob: float

    # Scenario B
    scenario_b_target: float
    scenario_b_lower: float
    scenario_b_upper: float
    scenario_b_prob: float

    combined_prob: float
    best_horizon_days: int

    # Signal tier classification
    signal_tier: SignalTier = SignalTier.AGGRESSIVE
    tier_reasons: str = ""  # human-readable explanation of why tier was assigned

    calendar_ctx: Optional[CalendarContext] = None

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "confluence_valid": int(self.confluence_valid),
            "cluster_valid": int(self.cluster_valid),
            "proximity_pct": self.proximity_pct,
            "raw_strength": self.raw_strength,
            "adjusted_strength": self.adjusted_strength,
            "scenario_a_target": self.scenario_a_target,
            "scenario_a_lower": self.scenario_a_lower,
            "scenario_a_upper": self.scenario_a_upper,
            "scenario_a_prob": round(self.scenario_a_prob, 4),
            "scenario_b_target": self.scenario_b_target,
            "scenario_b_lower": self.scenario_b_lower,
            "scenario_b_upper": self.scenario_b_upper,
            "scenario_b_prob": round(self.scenario_b_prob, 4),
            "combined_prob": round(self.combined_prob, 4),
            "best_horizon_days": self.best_horizon_days,
            "signal_tier": self.signal_tier.value,
            "tier_reasons": self.tier_reasons,
        }


class ConfluenceChecker:

    def __init__(self, calendar_config_path: str = "config/economic_calender.yaml"):
        self.calendar_engine = EconomicCalendarEngine(calendar_config_path)

    def analyze(
        self,
        as_of: date,
        cluster: ClusterResult,
        tft_predictions: List[TFTPrediction],
        zone_tolerance_pct: float = 1.0,
        cluster_overlap_tolerance_pct: float = AGGRESSIVE_CLUSTER_TOLERANCE_PCT,
    ) -> ConfluenceReport:
        """
        Analyze predictions against Fibonacci zones and produce a ConfluenceReport.

        Every signal that passes the AGGRESSIVE gate is valid.  A subset of
        those signals additionally qualify for the SELECTIVE tier when they
        satisfy tighter overlap, probability, and horizon constraints.
        """
        if not tft_predictions:
            raise ValueError("tft_predictions list cannot be empty")

        # 1. Identify if any horizon is confluent (q50 overlaps the cluster zone)
        #    Uses the AGGRESSIVE (loose) tolerance so we don't reject any signal
        confluent_preds = []
        for pred in tft_predictions:
            if cluster.cluster_valid:
                if is_confluent(pred.q50, cluster.cluster_lower, cluster.cluster_upper, cluster_overlap_tolerance_pct):
                    confluent_preds.append(pred)

        confluence_valid = len(confluent_preds) > 0 and cluster.cluster_valid

        # 2. Pick the best horizon (prediction closest to cluster midpoint)
        if confluent_preds:
            best_pred = min(confluent_preds, key=lambda p: abs(p.q50 - cluster.cluster_mid))
        else:
            best_pred = min(tft_predictions, key=lambda p: abs(p.q50 - cluster.cluster_mid))

        # 3. Generate Scenario Zones
        plan_a, plan_b = generate_entry_zones(
            target_a=cluster.scenario_a.price,
            target_b=cluster.scenario_b.price,
            cluster_lower=cluster.cluster_lower,
            cluster_upper=cluster.cluster_upper,
            cluster_valid=cluster.cluster_valid,
            zone_tolerance_pct=zone_tolerance_pct,
        )

        # 4. Compute Scenario Probabilities based on the best matching prediction's distribution
        prob_a = compute_probability_mass(plan_a.zone_lower, plan_a.zone_upper, best_pred.q10, best_pred.q50, best_pred.q90)
        prob_b = compute_probability_mass(plan_b.zone_lower, plan_b.zone_upper, best_pred.q10, best_pred.q50, best_pred.q90)
        combined_prob = min(1.0, prob_a + prob_b)

        # 5. Apply Calendar Adjustments to strength
        cal_ctx = self.calendar_engine.get_context(as_of)
        adjusted_strength = self.calendar_engine.adjust_confidence(cluster.cluster_strength, cal_ctx)

        # 6. ── Tier classification ────────────────────────────────────────
        signal_tier, tier_reasons = self._classify_tier(
            confluence_valid=confluence_valid,
            cluster=cluster,
            best_pred=best_pred,
            combined_prob=combined_prob,
            adjusted_strength=adjusted_strength,
        )

        return ConfluenceReport(
            as_of=as_of,
            confluence_valid=confluence_valid,
            cluster_valid=cluster.cluster_valid,
            proximity_pct=cluster.proximity_pct,
            raw_strength=cluster.cluster_strength,
            adjusted_strength=round(adjusted_strength, 4),
            scenario_a_target=plan_a.target_price,
            scenario_a_lower=plan_a.zone_lower,
            scenario_a_upper=plan_a.zone_upper,
            scenario_a_prob=prob_a,
            scenario_b_target=plan_b.target_price,
            scenario_b_lower=plan_b.zone_lower,
            scenario_b_upper=plan_b.zone_upper,
            scenario_b_prob=prob_b,
            combined_prob=combined_prob,
            best_horizon_days=best_pred.horizon_days,
            signal_tier=signal_tier,
            tier_reasons=tier_reasons,
            calendar_ctx=cal_ctx,
        )

    # ── private ─────────────────────────────────────────────────────────

    @staticmethod
    def _classify_tier(
        confluence_valid: bool,
        cluster: ClusterResult,
        best_pred: TFTPrediction,
        combined_prob: float,
        adjusted_strength: float,
    ) -> tuple[SignalTier, str]:
        """
        Classify a signal as SELECTIVE or AGGRESSIVE.

        SELECTIVE criteria (ALL must pass):
          1. q50 within ±SELECTIVE_CLUSTER_TOLERANCE_PCT of cluster bounds
          2. combined_prob >= SELECTIVE_MIN_COMBINED_PROB
          3. best_horizon_days <= SELECTIVE_MAX_HORIZON_DAYS
          4. adjusted_strength >= SELECTIVE_MIN_STRENGTH

        Everything else that still has confluence_valid=True is AGGRESSIVE.
        """
        if not confluence_valid:
            return SignalTier.AGGRESSIVE, "No confluence — default aggressive"

        fail_reasons: list[str] = []

        # Check 1: Tight cluster overlap
        tight_overlap = is_confluent(
            best_pred.q50,
            cluster.cluster_lower,
            cluster.cluster_upper,
            SELECTIVE_CLUSTER_TOLERANCE_PCT,
        )
        if not tight_overlap:
            fail_reasons.append(
                f"q50 outside ±{SELECTIVE_CLUSTER_TOLERANCE_PCT}% of cluster"
            )

        # Check 2: Minimum probability mass
        if combined_prob < SELECTIVE_MIN_COMBINED_PROB:
            fail_reasons.append(
                f"combined_prob {combined_prob:.2%} < {SELECTIVE_MIN_COMBINED_PROB:.0%}"
            )

        # Check 3: Horizon preference
        if best_pred.horizon_days > SELECTIVE_MAX_HORIZON_DAYS:
            fail_reasons.append(
                f"horizon {best_pred.horizon_days}d > {SELECTIVE_MAX_HORIZON_DAYS}d"
            )

        # Check 4: Minimum adjusted strength
        if adjusted_strength < SELECTIVE_MIN_STRENGTH:
            fail_reasons.append(
                f"strength {adjusted_strength:.2f} < {SELECTIVE_MIN_STRENGTH}"
            )

        if not fail_reasons:
            return SignalTier.SELECTIVE, "All selective criteria passed"

        return SignalTier.AGGRESSIVE, "; ".join(fail_reasons)
