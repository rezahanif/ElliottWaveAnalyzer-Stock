"""
providers.py
------------
Provider fallback chain for BMRI.JK and IDX indices.
Implements: Yahoo → AlphaVantage → TwelveData → IDX → Playwright

Failure policy: if all providers fail, keep operating on last verified local data
and log a warning — never hard-fail.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Callable

import pandas as pd
import requests

logger = logging.getLogger("stock_providers")

# API keys from environment
ALPHAVANTAGE_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
TWELVEDATA_KEY = os.environ.get("TWELVEDATA_API_KEY", "")

# User agents for bypassing ISP blocks
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]


class ProviderResult:
    """Result from a provider fetch attempt."""
    
    def __init__(self, df: Optional[pd.DataFrame], provider: str, success: bool, error: str = ""):
        self.df = df
        self.provider = provider
        self.success = success
        self.error = error


def fetch_yahoo_chart(symbol: str, start_ts: int, end_ts: int) -> ProviderResult:
    """Yahoo Finance v8 Chart API - primary provider (bypasses Indonesian ISP blocks)."""
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
    }
    headers = {"User-Agent": USER_AGENTS[0]}
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return ProviderResult(None, "yahoo", False, f"HTTP {r.status_code}")
        
        result = r.json()["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        if not timestamps:
            return ProviderResult(None, "yahoo", False, "No timestamps in response")
        
        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "timestamp_ms": [int(ts) * 1000 for ts in timestamps],
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        })
        
        df = df.dropna(subset=["close", "high", "low", "open"], how="all")
        df = df.ffill().bfill()
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.strftime("%Y-%m-%d")
        
        logger.info(f"Yahoo fetched {len(df)} bars for {symbol}")
        return ProviderResult(df, "yahoo", True)
        
    except Exception as e:
        return ProviderResult(None, "yahoo", False, str(e))


def fetch_alphavantage(symbol: str, outputsize: str = "full") -> ProviderResult:
    """AlphaVantage TIME_SERIES_DAILY endpoint."""
    if not ALPHAVANTAGE_KEY:
        return ProviderResult(None, "alphavantage", False, "No API key configured")
    
    # Convert BMRI.JK to BMRI.JK format for AlphaVantage
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": ALPHAVANTAGE_KEY,
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return ProviderResult(None, "alphavantage", False, f"HTTP {r.status_code}")
        
        data = r.json()
        if "Time Series (Daily)" not in data:
            return ProviderResult(None, "alphavantage", False, data.get("Note", "No time series data"))
        
        ts = data["Time Series (Daily)"]
        rows = []
        for date_str, values in ts.items():
            rows.append({
                "date": date_str,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": float(values["5. volume"]),
            })
        
        df = pd.DataFrame(rows)
        df["timestamp_ms"] = pd.to_datetime(df["date"]).astype("int64") // 1_000_000
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        
        logger.info(f"AlphaVantage fetched {len(df)} bars for {symbol}")
        return ProviderResult(df, "alphavantage", True)
        
    except Exception as e:
        return ProviderResult(None, "alphavantage", False, str(e))


def fetch_twelvedata(symbol: str, start_date: str, end_date: str) -> ProviderResult:
    """TwelveData time_series endpoint."""
    if not TWELVEDATA_KEY:
        return ProviderResult(None, "twelvedata", False, "No API key configured")
    
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start_date,
        "end_date": end_date,
        "apikey": TWELVEDATA_KEY,
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return ProviderResult(None, "twelvedata", False, f"HTTP {r.status_code}")
        
        data = r.json()
        if "values" not in data:
            return ProviderResult(None, "twelvedata", False, data.get("message", "No values"))
        
        rows = []
        for v in data["values"]:
            rows.append({
                "date": v["datetime"],
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
                "volume": float(v["volume"]),
            })
        
        df = pd.DataFrame(rows)
        df["timestamp_ms"] = pd.to_datetime(df["date"]).astype("int64") // 1_000_000
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        
        logger.info(f"TwelveData fetched {len(df)} bars for {symbol}")
        return ProviderResult(df, "twelvedata", True)
        
    except Exception as e:
        return ProviderResult(None, "twelvedata", False, str(e))


def fetch_playwright_fallback(symbol: str, start_date: str, end_date: str) -> ProviderResult:
    """
    Playwright-based scraper for when all APIs fail.
    Uses cached Chromium binary with stealth mode.
    """
    try:
        from playwright.sync_api import sync_playwright
        from src.stock.collectors.price import _chromium_executable
        
        chromium_path = _chromium_executable()
        if not chromium_path:
            return ProviderResult(None, "playwright", False, "No Chromium binary found")
        
        # Use Yahoo Finance CSV download endpoint
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        
        csv_url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
        csv_url += f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=chromium_path,
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            page = browser.new_page()
            
            # Try stealth if available
            try:
                from playwright_stealth import Stealth
                Stealth().apply_stealth_sync(page)
            except ImportError:
                pass
            
            page.goto(csv_url, timeout=30000)
            content = page.content()
            browser.close()
            
            # Parse CSV from page content
            if "Date,Open,High,Low,Close" not in content:
                return ProviderResult(None, "playwright", False, "CSV not in response")
            
            # Extract text content
            import re
            csv_match = re.search(r'<pre[^>]*>(.*?)</pre>', content, re.DOTALL)
            if not csv_match:
                return ProviderResult(None, "playwright", False, "Could not extract CSV")
            
            csv_text = csv_match.group(1)
            import io
            df = pd.read_csv(io.StringIO(csv_text))
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adj_close",
            })
            df = df.dropna(subset=["close"])
            df["timestamp_ms"] = pd.to_datetime(df["date"]).astype("int64") // 1_000_000
            
            logger.info(f"Playwright fetched {len(df)} bars for {symbol}")
            return ProviderResult(df, "playwright", True)
            
    except Exception as e:
        return ProviderResult(None, "playwright", False, str(e))


class ProviderChain:
    """
    Fallback chain for historical data fetch.
    Tries providers in order, logs failures, never hard-fails.
    """
    
    PROVIDERS: List[tuple] = [
        ("yahoo", fetch_yahoo_chart),
        ("alphavantage", fetch_alphavantage),
        ("twelvedata", fetch_twelvedata),
        ("playwright", fetch_playwright_fallback),
    ]
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.attempts: List[ProviderResult] = []
    
    def fetch(self, start_date: str, end_date: str, metadata: Optional[Any] = None) -> Optional[pd.DataFrame]:
        """
        Try each provider in order until one succeeds.
        Returns None if all fail (graceful degradation).
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        
        for name, fetcher in self.PROVIDERS:
            logger.info(f"Trying provider {name} for {self.symbol}...")
            
            # Call appropriate fetcher based on signature
            if name == "yahoo":
                result = fetcher(self.symbol, start_ts, end_ts)
            elif name == "alphavantage":
                result = fetcher(self.symbol)
            elif name == "twelvedata":
                result = fetcher(self.symbol, start_date, end_date)
            elif name == "playwright":
                result = fetcher(self.symbol, start_date, end_date)
            else:
                continue
            
            self.attempts.append(result)
            
            if result.success and result.df is not None and len(result.df) > 0:
                logger.info(f"Provider {name} succeeded with {len(result.df)} bars")
                
                # Update metadata
                if metadata:
                    metadata.log_provider(name, is_fallback=(name != "yahoo"))
                
                return result.df
            
            logger.warning(f"Provider {name} failed: {result.error}")
            time.sleep(1)  # Brief pause between attempts
        
        # All providers failed - log warning but don't crash
        logger.error(f"All providers failed for {self.symbol}. Operating on last verified local data.")
        return None
    
    def summary(self) -> str:
        lines = [f"Provider chain for {self.symbol}:"]
        for a in self.attempts:
            status = "✓" if a.success else "✗"
            lines.append(f"  {status} {a.provider}: {a.error if not a.success else 'OK'}")
        return "\n".join(lines)
