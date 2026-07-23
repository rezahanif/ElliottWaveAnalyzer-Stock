from datetime import date
import pytest

from src.waveconf.fib_engine.fibonacci import ClusterResult, FibTarget
from src.waveconf.confluence.scorer import ConfluenceChecker, TFTPrediction
from src.waveconf.confluence.cluster_check import compute_probability_mass, is_confluent
from src.waveconf.confluence.entry_plan import generate_entry_zones


def _mk_target(price):
    return FibTarget(
        price=price, method="test_method", ratio=1.618, anchor_price=price,
        anchor_label="test", direction="bearish", pattern_type="", note=""
    )


def _mk_cluster(target_a, target_b, valid=True):
    cluster_upper = max(target_a, target_b)
    cluster_lower = min(target_a, target_b)
    return ClusterResult(
        target_a=_mk_target(target_a),
        target_b=_mk_target(target_b),
        measured_move=None,
        cluster_valid=valid,
        proximity_pct=1.0,
        cluster_upper=cluster_upper,
        cluster_lower=cluster_lower,
        cluster_mid=(cluster_upper + cluster_lower) / 2,
        cluster_strength=0.80,
        scenario_a=_mk_target(target_a),
        scenario_b=_mk_target(target_b),
    )


def test_is_confluent_boundary_conditions():
    assert is_confluent(60000, 59000, 61000, 2.0) is True
    # Exactly at boundary
    # lower bound: 59000 * 0.98 = 57820
    # upper bound: 61000 * 1.02 = 62220
    assert is_confluent(57820, 59000, 61000, 2.0) is True
    assert is_confluent(62220, 59000, 61000, 2.0) is True
    # Outside boundary
    assert is_confluent(57800, 59000, 61000, 2.0) is False
    assert is_confluent(62300, 59000, 61000, 2.0) is False


def test_probability_mass_basic_regions():
    # q10=100, q50=150, q90=200
    # d1 = 50, d2 = 50
    # lower_bound = 50, upper_bound = 250
    # CDF:
    # 50 -> 0.0
    # 100 -> 0.10
    # 150 -> 0.50
    # 200 -> 0.90
    # 250 -> 1.0
    
    # Exact quantile targets
    assert abs(compute_probability_mass(50, 100, 100, 150, 200) - 0.10) < 1e-9
    assert abs(compute_probability_mass(100, 150, 100, 150, 200) - 0.40) < 1e-9
    assert abs(compute_probability_mass(150, 200, 100, 150, 200) - 0.40) < 1e-9
    assert abs(compute_probability_mass(200, 250, 100, 150, 200) - 0.10) < 1e-9
    
    # Combined target spanning multiple intervals
    assert abs(compute_probability_mass(100, 200, 100, 150, 200) - 0.80) < 1e-9
    
    # Outside distribution bounds
    assert compute_probability_mass(0, 40, 100, 150, 200) == 0.0
    assert compute_probability_mass(260, 300, 100, 150, 200) == 0.0


def test_generate_entry_zones_logic():
    # Clustered case
    plan_a, plan_b = generate_entry_zones(
        target_a=63000, target_b=60000,
        cluster_lower=59500, cluster_upper=60500,
        cluster_valid=True, zone_tolerance_pct=1.0,
    )
    assert plan_a.target_price == 63000
    assert plan_a.zone_lower == 63000 * 0.99
    assert plan_a.zone_upper == 63000 * 1.01

    assert plan_b.target_price == 60000
    assert plan_b.zone_lower == 59500
    assert plan_b.zone_upper == 60500

    # Non-clustered case
    plan_a, plan_b = generate_entry_zones(
        target_a=63000, target_b=55000,
        cluster_lower=0, cluster_upper=0,
        cluster_valid=False, zone_tolerance_pct=2.0,
    )
    assert plan_b.zone_lower == 55000 * 0.98
    assert plan_b.zone_upper == 55000 * 1.02


def test_confluence_checker_valid_confluence():
    checker = ConfluenceChecker()
    cluster = _mk_cluster(target_a=61000, target_b=59000, valid=True) # cluster_mid = 60000
    
    predictions = [
        TFTPrediction(horizon_days=7, q10=55000, q50=57000, q90=59000),   # not confluent
        TFTPrediction(horizon_days=30, q10=52000, q50=60100, q90=68000),  # confluent (q50=60100 inside 59000-61000)
    ]
    
    report = checker.analyze(date(2026, 6, 20), cluster, predictions)
    
    assert report.confluence_valid is True
    assert report.best_horizon_days == 30
    assert report.raw_strength == 0.80
    assert report.scenario_a_target == 61000
    assert report.scenario_b_target == 59000
    assert report.scenario_b_lower == 59000  # cluster lower
    assert report.scenario_b_upper == 61000  # cluster upper


def test_confluence_checker_invalid_confluence_no_overlap():
    checker = ConfluenceChecker()
    cluster = _mk_cluster(target_a=61000, target_b=59000, valid=True)
    
    # None of the predictions have q50 overlapping the cluster within 2%
    predictions = [
        TFTPrediction(horizon_days=7, q10=55000, q50=57000, q90=59000),
        TFTPrediction(horizon_days=30, q10=45000, q50=50000, q90=55000),
    ]
    
    report = checker.analyze(date(2026, 6, 20), cluster, predictions)
    assert report.confluence_valid is False
    assert report.best_horizon_days in (7, 30)  # still maps a best horizon
    assert report.combined_prob >= 0.0


def test_confluence_checker_invalid_confluence_invalid_cluster():
    checker = ConfluenceChecker()
    # Cluster itself is invalid (target_a and target_b are too far apart)
    cluster = _mk_cluster(target_a=70000, target_b=50000, valid=False)
    
    predictions = [
        TFTPrediction(horizon_days=30, q10=52000, q50=60000, q90=68000),
    ]
    
    report = checker.analyze(date(2026, 6, 20), cluster, predictions)
    assert report.confluence_valid is False
    assert report.cluster_valid is False


def test_confluence_checker_calendar_discounting():
    checker = ConfluenceChecker()
    cluster = _mk_cluster(target_a=61000, target_b=59000, valid=True)
    predictions = [TFTPrediction(horizon_days=30, q10=52000, q50=60000, q90=68000)]
    
    # As of 2026-07-27: 2 days before the 2026-07-29 FOMC -> 0.60 discount
    report = checker.analyze(date(2026, 7, 27), cluster, predictions)
    assert report.raw_strength == 0.80
    assert round(report.adjusted_strength, 4) == round(0.80 * 0.60, 4)


def test_degenerate_quantiles_handled_gracefully():
    # If the quantiles are degenerate (e.g. model output is flat / all quantiles equal)
    prob = compute_probability_mass(100, 110, 100, 100, 100)
    assert 0.0 <= prob <= 1.0  # didn't crash with ZeroDivisionError
