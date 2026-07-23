"""
rule_engine.py
--------------
Stock trading rule engine.
Combines wave structures, Fibonacci clusters, fundamental ratios, news sentiment,
and the IHSG/sector market cascade into a final action signal: BUY, SELL, or WATCH.
"""

from __future__ import annotations

from typing import Dict, Any, Optional


def evaluate_rules(
    symbol: str,
    current_price: float,
    analysis_res: Dict[str, Any],
    market_ctx: Dict[str, Any],
    fundamentals: Dict[str, Any],
    news_sentiment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate the rules matrix to determine final swing trading signal.
    """
    # 1. Technical Signals
    direction = analysis_res.get("direction")  # "bullish" or "bearish"
    fib_res = analysis_res.get("fibonacci")
    invalidation = analysis_res.get("invalidation")

    near_fib_support = False
    near_fib_resistance = False
    fib_zones_text = "No valid zones"

    if fib_res and fib_res.cluster_valid:
        # Check if current price is close to the cluster support/resistance
        cluster_mid = (fib_res.cluster_lower + fib_res.cluster_upper) / 2.0
        deviation = abs(current_price - cluster_mid) / cluster_mid
        fib_zones_text = f"${fib_res.cluster_lower:,.2f} - ${fib_res.cluster_upper:,.2f}"

        if direction == "bullish":
            # In bullish wave, cluster acts as buy/support zone
            if deviation <= 0.05:  # within 5% of cluster
                near_fib_support = True
        elif direction == "bearish":
            # In bearish wave, cluster acts as sell/resistance zone
            if deviation <= 0.05:
                near_fib_resistance = True

    # 2. Market Context
    ihsg_bias = market_ctx["ihsg"]["bias"]
    sector_outperforming = market_ctx["sector"]["outperforming_market"]
    stock_outperforming = market_ctx["stock"]["outperforming_sector"]
    composite_market = market_ctx["composite_alignment"]

    # 3. Fundamentals
    pe = fundamentals.get("pe_ratio", 99.0)
    roe = fundamentals.get("roe", 0.0)
    pb = fundamentals.get("pb_ratio", 99.0)

    # Criteria: Good value (P/E < 12) + strong profitability (ROE > 15% / 0.15)
    fundamental_ok = pe < 12.0 and roe >= 0.15

    # 4. Sentiment
    sentiment_class = news_sentiment.get("sentiment_class", "NEUTRAL")
    sentiment_score = news_sentiment.get("sentiment_score", 0.0)
    sentiment_ok = sentiment_class in ["POSITIVE", "NEUTRAL"]

    # ─────────────────────────────────────────────────────────
    # Signal Decisions (PRD §13)
    # ─────────────────────────────────────────────────────────
    reasons = []
    signal = "WATCH"

    # Check Stop Loss/Invalidation first
    stop_triggered = False
    if invalidation is not None:
        if direction == "bullish" and current_price < invalidation:
            stop_triggered = True
        elif direction == "bearish" and current_price > invalidation:
            stop_triggered = True

    if stop_triggered:
        signal = "SELL"
        reasons.append(f"Technical invalidation level triggered (Stop Loss at ${invalidation:,.2f})")

    elif ihsg_bias == "BEARISH":
        signal = "SELL"
        reasons.append("IHSG market bias is BEARISH. General risk-off regime.")

    elif direction == "bearish" and near_fib_resistance:
        signal = "SELL"
        reasons.append(f"Bearish wave structure + price near Fibonacci resistance zone ({fib_zones_text})")

    elif (
        direction == "bullish"
        and near_fib_support
        and ihsg_bias == "BULLISH"
        and (sector_outperforming or stock_outperforming)
        and fundamental_ok
        and sentiment_ok
    ):
        signal = "BUY"
        reasons.append(
            f"Bullish wave structure + price in Fibonacci support zone ({fib_zones_text}) "
            f"+ positive/neutral sentiment + strong fundamentals (P/E {pe:.1f}, ROE {roe*100:.1f}%)"
        )
    else:
        # Default to WATCH: compile reasons why it is a watch
        if direction == "bullish" and not near_fib_support:
            reasons.append(f"Bullish wave confirmed, but price is not near Fibonacci support zone yet ({fib_zones_text})")
        elif direction == "bullish" and not fundamental_ok:
            reasons.append(f"Technical setup ready, but fundamentals do not meet filters (P/E: {pe:.1f}, ROE: {roe*100:.1f}%)")
        elif direction == "bullish" and not sentiment_ok:
            reasons.append(f"Technical setup ready, but news sentiment is negative ({sentiment_class})")
        elif direction is None:
            reasons.append("No active wave count or pivots detected to determine trend structure.")
        else:
            reasons.append("Mixed signals across technicals, market context, and macro filters.")

    return {
        "symbol": symbol.upper(),
        "price": current_price,
        "signal": signal,
        "invalidation": invalidation,
        "fib_zone": fib_zones_text,
        "reasons": reasons,
        "metrics": {
            "pe": pe,
            "roe": roe,
            "sentiment_score": sentiment_score,
            "ihsg_bias": ihsg_bias,
        },
    }
