"""
test_rule_engine.py
-------------------
Verify the trading rule engine logic under different scenarios.
"""

from __future__ import annotations

from src.btc.fib_engine.fibonacci import ClusterResult, FibTarget
from src.stock.forecast.rule_engine import evaluate_rules


def test_rule_engine_buy_signal():
    # Setup mock inputs that satisfy all BUY criteria
    dummy_target = FibTarget(100.0, "A", 1.618, 80.0, "C", "bullish")
    mock_cluster = ClusterResult(
        target_a=dummy_target,
        target_b=dummy_target,
        measured_move=None,
        cluster_valid=True,
        proximity_pct=2.0,
        cluster_upper=105.0,
        cluster_lower=95.0,
        cluster_mid=100.0,
        cluster_strength=0.8,
        scenario_a=dummy_target,
        scenario_b=dummy_target,
    )
    
    analysis_res = {
        "direction": "bullish",
        "fibonacci": mock_cluster,
        "invalidation": 80.0,
    }
    
    market_ctx = {
        "ihsg": {"bias": "BULLISH", "recent_change_pct": 1.2},
        "sector": {"outperforming_market": True, "recent_relative_change_pct": 0.5},
        "stock": {"outperforming_sector": True, "recent_relative_change_pct": 0.3},
        "composite_alignment": "STRONG_BULLISH",
    }
    
    fundamentals = {
        "pe_ratio": 10.5,
        "roe": 0.18,
        "pb_ratio": 1.9,
    }
    
    news_sentiment = {
        "sentiment_score": 0.3,
        "sentiment_class": "POSITIVE",
    }
    
    # Evaluate rules when price is at 99.0 (within 5% of cluster_mid=100.0)
    result = evaluate_rules(
        symbol="BMRI.JK",
        current_price=99.0,
        analysis_res=analysis_res,
        market_ctx=market_ctx,
        fundamentals=fundamentals,
        news_sentiment=news_sentiment,
    )
    
    print("\nRule Engine BUY signal test:")
    print("Signal:", result["signal"])
    print("Reasons:", result["reasons"])
    
    assert result["signal"] == "BUY"
    assert "invalidation" in result


def test_rule_engine_sell_regime():
    # Setup mock inputs where IHSG is bearish -> triggers SELL
    dummy_target = FibTarget(100.0, "A", 1.618, 80.0, "C", "bullish")
    mock_cluster = ClusterResult(
        target_a=dummy_target,
        target_b=dummy_target,
        measured_move=None,
        cluster_valid=True,
        proximity_pct=2.0,
        cluster_upper=105.0,
        cluster_lower=95.0,
        cluster_mid=100.0,
        cluster_strength=0.8,
        scenario_a=dummy_target,
        scenario_b=dummy_target,
    )
    
    analysis_res = {
        "direction": "bullish",
        "fibonacci": mock_cluster,
        "invalidation": 80.0,
    }
    
    market_ctx = {
        "ihsg": {"bias": "BEARISH", "recent_change_pct": -2.0},
        "sector": {"outperforming_market": False, "recent_relative_change_pct": -0.5},
        "stock": {"outperforming_sector": False, "recent_relative_change_pct": -0.3},
        "composite_alignment": "WEAK_BEARISH",
    }
    
    fundamentals = {"pe_ratio": 10.5, "roe": 0.18}
    news_sentiment = {"sentiment_score": 0.0, "sentiment_class": "NEUTRAL"}
    
    result = evaluate_rules(
        symbol="BMRI.JK",
        current_price=99.0,
        analysis_res=analysis_res,
        market_ctx=market_ctx,
        fundamentals=fundamentals,
        news_sentiment=news_sentiment,
    )
    
    print("\nRule Engine BEARISH SELL signal test:")
    print("Signal:", result["signal"])
    
    assert result["signal"] == "SELL"


if __name__ == "__main__":
    test_rule_engine_buy_signal()
    test_rule_engine_sell_regime()
