"""
multi_tf.py
-----------
Multi-Timeframe Confluence Layer — aligns and aggregates signals across
1D, 4H, and 1W timeframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MultiTFReport:
    dominant_bias: str  # "bullish" or "bearish"
    agreement_count: int  # number of TFs agreeing with dominant bias (2 or 3)
    confluence_score: float  # 0.0 to 1.0 depending on agreement and overlaps
    confluent_zone: Optional[Tuple[float, float]]  # overlapping target zone if any
    notes: str


def compute_multi_tf_confluence(
    tf_results: Dict[str, Dict],
    tolerance_pct: float = 5.0,
) -> MultiTFReport:
    """
    Scrutinize prediction and pivot target records across 1D, 4H, and 1W timeframes.
    Checks:
    - Directional bias alignment.
    - Proximity/overlap of Fibonacci target zones.
    """
    valid_results = {tf: res for tf, res in tf_results.items() if res is not None}
    if len(valid_results) < 2:
        return MultiTFReport(
            dominant_bias="neutral",
            agreement_count=0,
            confluence_score=0.0,
            confluent_zone=None,
            notes="Insufficient timeframe data for multi-TF evaluation.",
        )

    # 1. Count directional bias
    biases = [res.get("direction") for res in valid_results.values()]
    bullish_count = biases.count("bullish")
    bearish_count = biases.count("bearish")

    if bullish_count >= bearish_count:
        dominant_bias = "bullish"
        agreement_count = bullish_count
    else:
        dominant_bias = "bearish"
        agreement_count = bearish_count

    agreeing_tfs = [tf for tf, res in valid_results.items() if res.get("direction") == dominant_bias]

    # If only 1 TF agrees, or no agreement (e.g. 1 bullish, 1 bearish, 1 neutral), score is 0
    if len(agreeing_tfs) < 2:
        return MultiTFReport(
            dominant_bias=dominant_bias,
            agreement_count=len(agreeing_tfs),
            confluence_score=0.0,
            confluent_zone=None,
            notes=f"No multi-TF confluence: TFs disagree ({biases}).",
        )

    # 2. Check for target zone overlaps among agreeing timeframes
    # For each agreeing timeframe, target bounds are [target_a, target_b] or cluster bounds
    zones: List[Tuple[float, float]] = []
    for tf in agreeing_tfs:
        res = valid_results[tf]
        # Use cluster bounds if valid, else Scenario targets
        if res.get("cluster_valid") and res.get("cluster_lower") and res.get("cluster_upper"):
            lo = float(res["cluster_lower"])
            hi = float(res["cluster_upper"])
        else:
            lo = min(float(res.get("target_a", 0)), float(res.get("target_b", 0)))
            hi = max(float(res.get("target_a", 0)), float(res.get("target_b", 0)))

        if lo > 0 and hi > 0:
            zones.append((lo, hi))

    # Compute overlapping zone intersection
    overlap_zone = None
    if len(zones) >= 2:
        # Start with the first zone expanded by tolerance
        current_lo = zones[0][0] * (1.0 - tolerance_pct / 100.0)
        current_hi = zones[0][1] * (1.0 + tolerance_pct / 100.0)

        has_overlap = True
        for z in zones[1:]:
            z_lo = z[0] * (1.0 - tolerance_pct / 100.0)
            z_hi = z[1] * (1.0 + tolerance_pct / 100.0)

            # Intersection
            inter_lo = max(current_lo, z_lo)
            inter_hi = min(current_hi, z_hi)

            if inter_lo <= inter_hi:
                current_lo, current_hi = inter_lo, inter_hi
            else:
                has_overlap = False
                break

        if has_overlap:
            overlap_zone = (round(current_lo, 2), round(current_hi, 2))

    # 3. Calculate score
    # Base score on agreement count: 3/3 = 0.6, 2/3 = 0.4
    base_score = 0.6 if agreement_count == 3 else 0.4
    # Boost if target zone overlaps
    if overlap_zone is not None:
        base_score += 0.4
    confluence_score = round(min(1.0, base_score), 2)

    tfs_str = ", ".join(agreeing_tfs)
    if overlap_zone:
        notes = f"Confluence confirmed ({agreement_count}/3 TFs: {tfs_str}) pointing {dominant_bias.upper()} with overlapping targets at ${overlap_zone[0]:,.2f} - ${overlap_zone[1]:,.2f}."
    else:
        notes = f"Directional agreement ({agreement_count}/3 TFs: {tfs_str}) pointing {dominant_bias.upper()} but no target zone overlap."

    return MultiTFReport(
        dominant_bias=dominant_bias,
        agreement_count=agreement_count,
        confluence_score=confluence_score,
        confluent_zone=overlap_zone,
        notes=notes,
    )
