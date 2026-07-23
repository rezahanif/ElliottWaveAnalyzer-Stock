"""
ihsg.py
-------
Fetches IHSG benchmark index data (^JKSE) using the price collector.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

from src.stock.collectors.price import fetch_price_data


def fetch_ihsg_data(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch Jakarta Composite Index (^JKSE) data."""
    return fetch_price_data("^JKSE", start_date=start_date, end_date=end_date)
