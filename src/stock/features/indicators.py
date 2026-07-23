"""
indicators.py
-------------
Feature calculation for stock data. Reuses clean, generic indicators from src/btc/ingestion/indicators.py.
"""

from __future__ import annotations

import pandas as pd
from src.btc.ingestion.indicators import add_indicators as _add_indicators


def calculate_stock_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators (RSI, MACD, ATR, Bollinger, Price normalization) to stock data."""
    return _add_indicators(df)
