"""
confidence.py
-------------
Component-weighted Confidence Model.

Computes per-component scores and an overall confidence score:
  Technical / Wave / Market / Fundamental / News / AI Forecast → Overall

Each component returns a score 0.0–1.0 and a weight. The overall score is
the weighted sum. When AI Forecast is unavailable, its weight is zeroed
(neutral contribution) — deterministic components fully determine the score.
"""

from __future__ import annotations

from typing import Dict, Any, Optional


# Default weights (sum = 1.0 when AI available; redistributed when not)
DEFAULT_WEIGHTS = {
    "technical":   0.25,
    "wave":        0.20,
    "market":      0.20,
    "fundamental": 0.15,
    "news":        0.10,
    "ai_forecast": 0.10,
}

# When AI unavailable, redistribute its weight proportionally across others
WEIGHTS_NO_AI = {
    "technical":   0.25 / 0.90,   # 0.278
    "wave":        0.20 / 0.90,   # 0.222
    "market":      0.20 / 0.90,   # 0.222
    "fundamental": 0.15 / 0.90,   # 0.167
    "news":        0.10 / 0.90,   # 0.111
    "ai_forecast": 0.0,
}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def score_technical(analysis_res: Dict[str, Any]) -> float:
    """Score based on pattern detection confidence."""
    pattern = analysis_res.get("pattern")
    if pattern and hasattr(pattern, "confidence"):
        return _clamp(pattern.confidence)
    return 0.3  # no pattern detected → low-moderate


def score_wave(analysis_res: Dict[str, Any]) -> float:
    """Score based on wave direction clarity and Fibonacci cluster validity."""
    direction = analysis_res.get("direction")
    fib = analysis_res.get("fibonacci")
    if direction is None:
        return 0.2
    base = 0.5
    if fib and hasattr(fib, "cluster_valid") and fib.cluster_valid:
        base += 0.3
    if hasattr(fib, "cluster_strength"):
        base += _clamp(fib.cluster_strength) * 0.2
    return _clamp(base)


def score_market(market_ctx: Dict[str, Any]) -> float:
    """Score based on IHSG/sector/stock alignment."""
    alignment = market_ctx.get("composite_alignment", "NEUTRAL_MIXED")
    if alignment == "STRONG_BULLISH":
        return 0.9
    elif alignment == "WEAK_BEARISH":
        return 0.15
    return 0.5


def score_fundamental(fundamentals: Dict[str, Any]) -> float:
    """Score based on P/E, ROE quality."""
    pe = fundamentals.get("pe_ratio", 99.0)
    roe = fundamentals.get("roe", 0.0)
    score = 0.5
    if pe < 12:
        score += 0.2
    elif pe > 20:
        score -= 0.2
    if roe >= 0.15:
        score += 0.2
    elif roe < 0.05:
        score -= 0.2
    return _clamp(score)


def score_news(news_sentiment: Dict[str, Any]) -> float:
    """Score based on sentiment class."""
    s = news_sentiment.get("sentiment_class", "NEUTRAL")
    if s == "POSITIVE":
        return 0.8
    elif s == "NEGATIVE":
        return 0.25
    return 0.5


def score_ai_forecast(fusion_result) -> float:
    """Score based on TFT confidence. Returns 0.0 when unavailable."""
    if fusion_result is None or not fusion_result.ai_enabled:
        return 0.0
    ai_data = fusion_result.ai_forecast_data or {}
    return _clamp(ai_data.get("confidence", 0.0))


def compute_confidence(
    analysis_res: Dict[str, Any],
    market_ctx: Dict[str, Any],
    fundamentals: Dict[str, Any],
    news_sentiment: Dict[str, Any],
    fusion_result=None,
) -> Dict[str, Any]:
    """
    Compute component-weighted confidence model.

    Returns dict with individual scores, weights, and overall score.
    """
    ai_available = fusion_result is not None and fusion_result.ai_enabled
    weights = DEFAULT_WEIGHTS if ai_available else WEIGHTS_NO_AI

    components = {
        "technical":   score_technical(analysis_res),
        "wave":        score_wave(analysis_res),
        "market":      score_market(market_ctx),
        "fundamental": score_fundamental(fundamentals),
        "news":        score_news(news_sentiment),
        "ai_forecast": score_ai_forecast(fusion_result),
    }

    overall = sum(components[k] * weights[k] for k in components)

    return {
        "components": {k: round(v, 3) for k, v in components.items()},
        "weights": {k: round(v, 3) for k, v in weights.items()},
        "overall": round(overall, 3),
        "ai_available": ai_available,
    }
