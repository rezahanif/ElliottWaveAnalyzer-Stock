"""
test_market_context.py
----------------------
Verify the IHSG -> sector -> Stock context cascade logic.
"""

from __future__ import annotations

import pandas as pd
from src.stock.features.market_context import generate_market_context


def test_market_context_smoke():
    # Setup simple dataframes with overlapping dates
    dates = pd.date_range(start="2025-01-01", periods=60).strftime("%Y-%m-%d")
    
    # Simulate upward trending IHSG
    df_ihsg = pd.DataFrame({
        "date": dates,
        "close": [7000 + i * 5 for i in range(60)]
    })
    
    # Simulate sector outperforming IHSG slightly
    df_sector = pd.DataFrame({
        "date": dates,
        "close": [800 + i * 1.2 for i in range(60)]
    })
    
    # Simulate stock outperforming sector slightly
    df_stock = pd.DataFrame({
        "date": dates,
        "close": [5000 + i * 12 for i in range(60)]
    })
    
    ctx = generate_market_context(df_stock, df_sector, df_ihsg)
    
    print("\nMarket Context Verification:")
    print("IHSG Bias:", ctx["ihsg"]["bias"])
    print("Sector Outperforming:", ctx["sector"]["outperforming_market"])
    print("Stock Outperforming:", ctx["stock"]["outperforming_sector"])
    print("Composite:", ctx["composite_alignment"])
    
    assert ctx["ihsg"]["bias"] == "BULLISH"
    assert "composite_alignment" in ctx
    print("Market Context test PASSED! ✅")


if __name__ == "__main__":
    test_market_context_smoke()
