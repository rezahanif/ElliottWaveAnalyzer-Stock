"""
storage.py
----------
BMRI historical data storage layer.
Implements parquet-based storage with metadata tracking per the Addendum spec.

Storage layout:
  data/stocks/BMRI/
    ├── daily.parquet
    ├── weekly.parquet      # Resampled from daily
    ├── monthly.parquet     # Resampled from daily
    └── metadata.json       # Coverage, providers, last update, validators
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("stock_storage")

# Storage root
STORAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "stocks" / "BMRI"


class BMRIMetadata:
    """Metadata tracking for BMRI historical data."""
    
    def __init__(self, path: Optional[Path] = None):
        self.path = path or STORAGE_ROOT / "metadata.json"
        self.data = self._load_or_create()
    
    def _load_or_create(self) -> Dict[str, Any]:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {
            "symbol": "BMRI.JK",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_updated": None,
            "coverage": {
                "daily": {"start": None, "end": None, "count": 0},
                "weekly": {"start": None, "end": None, "count": 0},
                "monthly": {"start": None, "end": None, "count": 0},
            },
            "providers": {
                "primary": None,
                "fallbacks_used": [],
            },
            "corporate_actions": {
                "splits": [],
                "dividends": [],
                "bonus_shares": [],
            },
            "validation": {
                "last_run": None,
                "errors": [],
                "warnings": [],
            },
            "incremental_updates": [],
        }
    
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def update_coverage(self, freq: str, df: pd.DataFrame):
        if len(df) == 0:
            return
        self.data["coverage"][freq] = {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
            "count": len(df),
        }
        self.data["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    def log_provider(self, provider: str, is_fallback: bool = False):
        if not is_fallback:
            self.data["providers"]["primary"] = provider
        elif provider not in self.data["providers"]["fallbacks_used"]:
            self.data["providers"]["fallbacks_used"].append(provider)
    
    def log_validation(self, errors: List[str], warnings: List[str]):
        self.data["validation"]["last_run"] = datetime.utcnow().isoformat() + "Z"
        self.data["validation"]["errors"] = errors
        self.data["validation"]["warnings"] = warnings
    
    def log_incremental_update(self, bars_added: int, timestamp: Optional[str] = None):
        entry = {
            "timestamp": timestamp or datetime.utcnow().isoformat() + "Z",
            "bars_added": bars_added,
        }
        self.data["incremental_updates"].append(entry)
        # Keep last 30 entries
        if len(self.data["incremental_updates"]) > 30:
            self.data["incremental_updates"] = self.data["incremental_updates"][-30:]


class BMRIStorage:
    """Parquet-based storage for BMRI OHLCV data."""
    
    REQUIRED_COLUMNS = ["timestamp_ms", "date", "open", "high", "low", "close", "volume"]
    
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else STORAGE_ROOT
        self.metadata = BMRIMetadata(self.root / "metadata.json")
    
    def _path(self, freq: str) -> Path:
        return self.root / f"{freq}.parquet"
    
    def exists(self, freq: str = "daily") -> bool:
        return self._path(freq).exists()
    
    def load(self, freq: str = "daily") -> Optional[pd.DataFrame]:
        """Load parquet file, return None if not exists."""
        path = self._path(freq)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    
    def save(self, df: pd.DataFrame, freq: str = "daily"):
        """Save DataFrame to parquet with compression."""
        path = self._path(freq)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure required columns exist
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by timestamp
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        
        # Save with snappy compression
        df.to_parquet(path, compression="snappy", index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        
        # Update metadata
        self.metadata.update_coverage(freq, df)
        self.metadata.save()
    
    def append(self, new_df: pd.DataFrame, freq: str = "daily") -> int:
        """Append new bars to existing data, deduplicate by timestamp."""
        existing = self.load(freq)
        if existing is None:
            self.save(new_df, freq)
            return len(new_df)
        
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp_ms"], keep="last")
        combined = combined.sort_values("timestamp_ms").reset_index(drop=True)
        
        bars_added = len(combined) - len(existing)
        if bars_added > 0:
            self.save(combined, freq)
            self.metadata.log_incremental_update(bars_added)
            self.metadata.save()
        
        return bars_added
    
    def resample_to_weekly(self, daily_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Resample daily to weekly (Monday-based)."""
        if daily_df is None:
            daily_df = self.load("daily")
        if daily_df is None or len(daily_df) == 0:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        
        df = daily_df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df = df.set_index("datetime")
        
        weekly = df.resample("W-MON").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        weekly = weekly.reset_index()
        weekly["timestamp_ms"] = weekly["datetime"].astype("int64") // 1_000_000
        weekly["date"] = weekly["datetime"].dt.strftime("%Y-%m-%d")
        weekly = weekly[self.REQUIRED_COLUMNS]
        
        return weekly
    
    def resample_to_monthly(self, daily_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Resample daily to monthly (month-start)."""
        if daily_df is None:
            daily_df = self.load("daily")
        if daily_df is None or len(daily_df) == 0:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        
        df = daily_df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df = df.set_index("datetime")
        
        monthly = df.resample("MS").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        monthly = monthly.reset_index()
        monthly["timestamp_ms"] = monthly["datetime"].astype("int64") // 1_000_000
        monthly["date"] = monthly["datetime"].dt.strftime("%Y-%m-%d")
        monthly = monthly[self.REQUIRED_COLUMNS]
        
        return monthly
    
    def generate_all_frequencies(self):
        """Generate weekly and monthly from daily."""
        daily = self.load("daily")
        if daily is None:
            logger.warning("No daily data to resample")
            return
        
        weekly = self.resample_to_weekly(daily)
        monthly = self.resample_to_monthly(daily)
        
        if len(weekly) > 0:
            self.save(weekly, "weekly")
        if len(monthly) > 0:
            self.save(monthly, "monthly")
        
        logger.info(f"Generated weekly ({len(weekly)} bars) and monthly ({len(monthly)} bars)")
