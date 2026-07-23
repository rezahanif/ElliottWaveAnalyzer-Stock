"""
market_context.py
------------------
Computes the 3-tier market context cascade:
IHSG (composite market) -> Sector (LQ45 liquid/financials proxy) -> Stock (BMRI.JK).
Determines relative strength and trend alignment.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd


def compute_trend(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> Dict[str, Any]:
    """Determine the trend direction and moving average alignment of a price series."""
    if len(df) < long_window:
        # Fallback if history is short
        return {"bias": "NEUTRAL", "ma_alignment": "NEUTRAL", "slope": 0.0}

    close = df["close"]
    ma_short = close.rolling(short_window).mean().iloc[-1]
    ma_long = close.rolling(long_window).mean().iloc[-1]
    current = close.iloc[-1]

    # Calculate price change slope over the last 5 days
    recent_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0.0

    if current > ma_short > ma_long:
        bias = "BULLISH"
    elif current < ma_short < ma_long:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "bias": bias,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "current": current,
        "recent_change_pct": round(recent_change * 100, 2),
    }


def generate_market_context(
    df_stock: pd.DataFrame,
    df_sector: pd.DataFrame,
    df_ihsg: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate the composite market context cascade:
    IHSG trend -> Sector relative strength -> Stock relative strength.
    """
    # 1. Benchmark Market (IHSG) Trend
    ihsg_ctx = compute_trend(df_ihsg)

    # 2. Sector relative strength to Market
    # Align dates using merge
    m_sector = pd.merge(df_sector[["date", "close"]], df_ihsg[["date", "close"]], on="date", suffixes=("_sector", "_ihsg"))
    m_sector["ratio"] = m_sector["close_sector"] / m_sector["close_ihsg"]
    
    sector_ratio_recent = m_sector["ratio"].iloc[-1]
    sector_ratio_ma = m_sector["ratio"].rolling(20).mean().iloc[-1] if len(m_sector) >= 20 else sector_ratio_recent
    
    sector_outperforming = sector_ratio_recent > sector_ratio_ma
    sector_recent_change = (m_sector["ratio"].iloc[-1] - m_sector["ratio"].iloc[-5]) / m_sector["ratio"].iloc[-5] if len(m_sector) >= 5 else 0.0

    # 3. Stock relative strength to Sector
    m_stock = pd.merge(df_stock[["date", "close"]], df_sector[["date", "close"]], on="date", suffixes=("_stock", "_sector"))
    m_stock["ratio"] = m_stock["close_stock"] / m_stock["close_sector"]
    
    stock_ratio_recent = m_stock["ratio"].iloc[-1]
    stock_ratio_ma = m_stock["ratio"].rolling(20).mean().iloc[-1] if len(m_stock) >= 20 else stock_ratio_recent
    
    stock_outperforming = stock_ratio_recent > stock_ratio_ma
    stock_recent_change = (m_stock["ratio"].iloc[-1] - m_stock["ratio"].iloc[-5]) / m_stock["ratio"].iloc[-5] if len(m_stock) >= 5 else 0.0

    # Cascade state decision
    if ihsg_ctx["bias"] == "BULLISH" and sector_outperforming and stock_outperforming:
        composite_alignment = "STRONG_BULLISH"
    elif ihsg_ctx["bias"] == "BEARISH" or (not sector_outperforming and not stock_outperforming):
        composite_alignment = "WEAK_BEARISH"
    else:
        composite_alignment = "NEUTRAL_MIXED"

    return {
        "ihsg": {
            "bias": ihsg_ctx["bias"],
            "recent_change_pct": ihsg_ctx["recent_change_pct"],
        },
        "sector": {
            "outperforming_market": sector_outperforming,
            "recent_relative_change_pct": round(sector_recent_change * 100, 2),
        },
        "stock": {
            "outperforming_sector": stock_outperforming,
            "recent_relative_change_pct": round(stock_recent_change * 100, 2),
        },
        "composite_alignment": composite_alignment,
    }
