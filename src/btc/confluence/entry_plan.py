"""
entry_plan.py
-------------
Helper logic to define price entry zones and trade sizes for Scenario A and B.
"""

from dataclasses import dataclass

@dataclass
class ScenarioPlan:
    label: str
    target_price: float
    zone_lower: float
    zone_upper: float
    allocation_note: str


def generate_entry_zones(
    target_a: float,
    target_b: float,
    cluster_lower: float,
    cluster_upper: float,
    cluster_valid: bool,
    zone_tolerance_pct: float = 1.0,
) -> tuple[ScenarioPlan, ScenarioPlan]:
    """
    Generate price zones and trade allocations for Scenario A (Target A)
    and Scenario B (Target B).
    """
    # Scenario A: First reaction zone (Target A)
    # Zone: +/- zone_tolerance_pct around target_a
    a_lower = target_a * (1.0 - zone_tolerance_pct / 100.0)
    a_upper = target_a * (1.0 + zone_tolerance_pct / 100.0)
    if a_lower > a_upper:
        a_lower, a_upper = a_upper, a_lower

    plan_a = ScenarioPlan(
        label="Scenario A",
        target_price=target_a,
        zone_lower=round(a_lower, 2),
        zone_upper=round(a_upper, 2),
        allocation_note="10-20% entry",
    )

    # Scenario B: Main Target Zone (Target B)
    # If cluster is valid, the zone is the cluster bounds [cluster_lower, cluster_upper]
    # Otherwise, it is +/- zone_tolerance_pct around target_b
    if cluster_valid:
        b_lower = cluster_lower
        b_upper = cluster_upper
    else:
        b_lower = target_b * (1.0 - zone_tolerance_pct / 100.0)
        b_upper = target_b * (1.0 + zone_tolerance_pct / 100.0)

    if b_lower > b_upper:
        b_lower, b_upper = b_upper, b_lower

    plan_b = ScenarioPlan(
        label="Scenario B",
        target_price=target_b,
        zone_lower=round(b_lower, 2),
        zone_upper=round(b_upper, 2),
        allocation_note="full entry",
    )

    return plan_a, plan_b
