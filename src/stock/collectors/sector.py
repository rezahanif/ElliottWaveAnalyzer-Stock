"""
sector.py
---------
Fetches IDX Financial index data (^JKFIN) using the price collector.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

from src.stock.collectors.price import fetch_price_data


def fetch_sector_data(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch IDX LQ45 Index (^JKLQ45) as sector proxy."""
    return fetch_price_data("^JKLQ45", start_date=start_date, end_date=end_date)
