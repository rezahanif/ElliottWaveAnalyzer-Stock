"""
test_stock_analyzer.py
----------------------
Smoke test for StockAnalyzer adapter.
Generates synthetic stock data offline and runs the technical analysis pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.stock.analyzer import StockAnalyzer


def generate_synthetic_data(n_bars: int = 500) -> pd.DataFrame:
    """Generate a clean synthetic stock dataset (Geometric Brownian Motion + Volatility)."""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_bars)]
    
    # Simulate price walk
    price = 100.0
    prices = []
    for _ in range(n_bars):
        # 1% daily drift/volatility
        price = price * (1.0 + np.random.normal(0.0005, 0.015))
        prices.append(price)
        
    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "close": prices
    })
    
    # Generate mock high, low, open around close
    df["open"] = df["close"] * (1.0 + np.random.uniform(-0.01, 0.01, n_bars))
    df["high"] = df[["open", "close"]].max(axis=1) * (1.0 + np.random.uniform(0.0, 0.02, n_bars))
    df["low"] = df[["open", "close"]].min(axis=1) * (1.0 - np.random.uniform(0.0, 0.02, n_bars))
    df["volume"] = np.random.randint(100000, 1000000, n_bars).astype(float)
    df["timestamp_ms"] = pd.to_datetime(df["date"]).astype("int64") // 10**6
    
    return df


def test_stock_analyzer_smoke():
    print("Generating synthetic stock data...")
    df = generate_synthetic_data(500)
    print(f"Generated {len(df)} rows. Columns: {list(df.columns)}")
    
    # Instantiate StockAnalyzer
    analyzer = StockAnalyzer(symbol="MOCK_STOCK", timeframe="1D")
    
    # Run pipeline
    res = analyzer.analyze(df)
    
    # Verify outputs
    print("\nVerification:")
    print(f"Symbol: {res['symbol']}")
    print(f"Timeframe: {res['timeframe']}")
    print(f"Calculated layers shape: {res['df_layers'].shape}")
    print(f"ZigZag results: {res['zigzag'].summary()}")
    print(f"Pattern detected: {res['pattern']}")
    print(f"Fibonacci result: {res['fibonacci']}")
    
    assert res["symbol"] == "MOCK_STOCK"
    assert "wall_street_threshold_pct" in res["df_layers"].columns
    assert "behavioral_threshold_pct" in res["df_layers"].columns
    
    # We should have found some pivots
    assert len(res["zigzag"].macro) > 0
    print("Smoke test PASSED! ✅")


if __name__ == "__main__":
    test_stock_analyzer_smoke()
