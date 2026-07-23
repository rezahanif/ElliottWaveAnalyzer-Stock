"""
price.py
--------
Fetches historical stock/index price data using Yahoo Finance Chart API (v8).
Bypasses typical ISP blocks and is highly reliable.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("stock_price_collector")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
]


def fetch_yahoo_chart_api(symbol: str, start_ts: int, end_ts: int) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from Yahoo Finance v8 chart API."""
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
    }
    headers = {"User-Agent": USER_AGENTS[0]}

    logger.info(f"Fetching from Yahoo Chart API: {symbol} ({start_ts} to {end_ts})")
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            logger.warning(f"Yahoo Chart API returned status {r.status_code}")
            return None

        result = r.json()["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        if not timestamps:
            logger.warning(f"No timestamps in response for {symbol}")
            return None

        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame(
            {
                "timestamp_ms": [int(ts) * 1000 for ts in timestamps],
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", []),
            }
        )

        # Forward-fill any occasional missing/NaN prices, drop rows with all NaNs
        df = df.dropna(subset=["close", "high", "low", "open"], how="all")
        df = df.ffill().bfill()

        # Generate date string column for technical calculations compatibility
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.strftime("%Y-%m-%d")

        logger.info(f"Successfully fetched {len(df)} rows from Yahoo Chart API")
        return df

    except Exception as e:
        logger.error(f"Yahoo Chart API fetch failed for {symbol}: {e}")
        return None


def fetch_price_data(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Unified fetcher with fallback logic.
    Returns a pandas DataFrame sorted by timestamp_ms.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    # Primary: Yahoo Chart API (fast, works inside Indonesian ISPs)
    df = fetch_yahoo_chart_api(symbol, start_ts, end_ts)
    if df is not None:
        return df

    # Fallback: yfinance (standard HTTP query)
    logger.info(f"Attempting fallback yfinance fetch for {symbol}...")
    try:
        import yfinance as yf
        df_yf = yf.download(symbol, start=start_dt, end=end_dt, progress=False)
        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            df_yf = df_yf.reset_index()
            df_yf = df_yf.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df_yf["date"] = pd.to_datetime(df_yf["date"])
            df_yf["timestamp_ms"] = df_yf["date"].view("int64") // 10**6
            logger.info(f"yfinance fetch successful: {len(df_yf)} rows")
            return df_yf
    except Exception as e:
        logger.warning(f"Fallback yfinance fetch failed: {e}")

    return None
