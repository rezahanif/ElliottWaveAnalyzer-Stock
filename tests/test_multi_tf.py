import pytest

from src.waveconf.confluence.multi_tf import compute_multi_tf_confluence


def test_multi_tf_all_agree_with_overlap():
    tf_results = {
        "1D": {
            "direction": "bullish",
            "cluster_valid": True,
            "cluster_lower": 59000.0,
            "cluster_upper": 61000.0,
        },
        "1W": {
            "direction": "bullish",
            "cluster_valid": True,
            "cluster_lower": 58000.0,
            "cluster_upper": 60500.0,
        },
        "4H": {
            "direction": "bullish",
            "cluster_valid": True,
            "cluster_lower": 59500.0,
            "cluster_upper": 62000.0,
        },
    }

    report = compute_multi_tf_confluence(tf_results, tolerance_pct=5.0)

    assert report.dominant_bias == "bullish"
    assert report.agreement_count == 3
    # 3/3 agree (0.6) + overlap (0.4) = 1.0
    assert report.confluence_score == 1.0
    assert report.confluent_zone is not None
    # 59000, 58000, 59500: max(lo - tol) -> max(56050, 55100, 56525) -> 56525
    # 61000, 60500, 62000: min(hi + tol) -> min(64050, 63525, 65100) -> 63525
    # So overlap should be around 56525 to 63525 expanded, or the exact overlap bounds.
    assert report.confluent_zone[0] < report.confluent_zone[1]


def test_multi_tf_partial_agreement_no_overlap():
    tf_results = {
        "1D": {
            "direction": "bullish",
            "cluster_valid": False,
            "target_a": 100000.0,
            "target_b": 110000.0,
        },
        "1W": {
            "direction": "bullish",
            "cluster_valid": False,
            "target_a": 20000.0,
            "target_b": 30000.0,
        },
        "4H": {
            "direction": "bearish",
            "cluster_valid": False,
            "target_a": 50000.0,
            "target_b": 52000.0,
        },
    }

    # 1D & 1W agree on BULLISH, but target zones [100k, 110k] and [20k, 30k] are far apart (even with 5% tol)
    report = compute_multi_tf_confluence(tf_results, tolerance_pct=5.0)

    assert report.dominant_bias == "bullish"
    assert report.agreement_count == 2
    # 2/3 agree (0.4) + no overlap (0.0) = 0.4
    assert report.confluence_score == 0.4
    assert report.confluent_zone is None


def test_multi_tf_disagreement():
    tf_results = {
        "1D": {"direction": "bullish"},
        "1W": {"direction": "bearish"},
    }

    report = compute_multi_tf_confluence(tf_results)

    # 1 bullish, 1 bearish -> dominant could be bullish (since it picks bullish on tie) but agreement_count is 1
    assert report.agreement_count == 1
    assert report.confluence_score == 0.0
    assert report.confluent_zone is None
