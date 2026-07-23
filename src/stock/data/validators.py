"""
validators.py
------------
Data quality validators for BMRI historical data.
Checks: duplicate timestamps, missing dates, invalid OHLC, zero-volume anomalies, timezone consistency.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Tuple, Optional

import pandas as pd

logger = logging.getLogger("stock_validators")


class ValidationResult:
    """Result from validation checks."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    @property
    def valid(self) -> bool:
        return len(self.errors) == 0
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        logger.error(f"Validation error: {msg}")
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"Validation warning: {msg}")
    
    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines.extend([f"  ✗ {e}" for e in self.errors[:10]])
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend([f"  ⚠ {w}" for w in self.warnings[:10]])
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        return "\n".join(lines) if lines else "All validations passed ✓"


def check_duplicate_timestamps(df: pd.DataFrame, result: ValidationResult):
    """Check for duplicate timestamp_ms values."""
    dupes = df[df.duplicated(subset=["timestamp_ms"], keep=False)]
    if len(dupes) > 0:
        result.add_error(f"Duplicate timestamps: {len(dupes)} rows affected")
        # Log sample duplicates
        sample = dupes.head(5)["date"].tolist()
        result.add_warning(f"Sample duplicate dates: {sample}")


def check_missing_dates(df: pd.DataFrame, result: ValidationResult):
    """Check for missing trading days (weekdays only, excludes holidays)."""
    if len(df) < 2:
        return
    
    dates = pd.to_datetime(df["date"])
    date_range = pd.date_range(start=dates.min(), end=dates.max(), freq="B")  # Business days
    
    existing = set(dates.dt.date)
    expected = set(date_range.date)
    
    missing = expected - existing
    
    # Filter out known IDX holidays (simplified - could load from idx_holidays.py)
    # For now, just flag if > 5 consecutive missing weekdays
    if len(missing) > 5:
        result.add_warning(f"Missing {len(missing)} trading days in range")
    elif len(missing) > 0:
        result.add_warning(f"Missing {len(missing)} trading days (likely holidays)")


def check_invalid_ohlc(df: pd.DataFrame, result: ValidationResult):
    """Check for invalid OHLC relationships (high < low, etc.)."""
    invalid_high_low = df[df["high"] < df["low"]]
    if len(invalid_high_low) > 0:
        result.add_error(f"high < low in {len(invalid_high_low)} rows")
    
    # Check if close is outside high/low range
    invalid_close = df[(df["close"] > df["high"]) | (df["close"] < df["low"])]
    if len(invalid_close) > 0:
        result.add_error(f"close outside [low, high] in {len(invalid_close)} rows")
    
    # Check for negative prices
    for col in ["open", "high", "low", "close"]:
        neg = df[df[col] < 0]
        if len(neg) > 0:
            result.add_error(f"Negative {col} in {len(neg)} rows")


def check_zero_volume(df: pd.DataFrame, result: ValidationResult):
    """Check for zero-volume anomalies (could indicate data gap or halt)."""
    zero_vol = df[df["volume"] == 0]
    if len(zero_vol) > 0:
        pct = len(zero_vol) / len(df) * 100
        if pct > 10:
            result.add_warning(f"Zero volume in {len(zero_vol)} rows ({pct:.1f}%)")
        elif len(zero_vol) > 3:
            result.add_warning(f"Zero volume in {len(zero_vol)} rows (possible trading halts)")
    
    # Check for negative volume
    neg_vol = df[df["volume"] < 0]
    if len(neg_vol) > 0:
        result.add_error(f"Negative volume in {len(neg_vol)} rows")


def check_timezone_consistency(df: pd.DataFrame, result: ValidationResult):
    """Check that timestamps are consistent (all in same timezone assumption)."""
    # All timestamps should be at market open time (09:00 WIB = 02:00 UTC)
    # or at midnight depending on source
    
    ts_hours = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.hour
    
    # Most sources use either 00:00 or market open time
    unique_hours = ts_hours.unique()
    
    if len(unique_hours) > 2:
        result.add_warning(f"Multiple timestamp hours: {sorted(unique_hours)}")
    
    # Check for timezone drift (timestamps not aligning)
    ts_diff = df["timestamp_ms"].diff()
    expected_diff = 86400000  # 1 day in ms
    
    drift = ts_diff[(ts_diff < expected_diff * 0.9) | (ts_diff > expected_diff * 1.1)]
    if len(drift) > 1:  # Allow 1 for the first row
        result.add_warning(f"Timestamp gaps/drift in {len(drift) - 1} bars")


def check_nan_values(df: pd.DataFrame, result: ValidationResult):
    """Check for NaN values in critical columns."""
    for col in ["open", "high", "low", "close", "volume"]:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            result.add_error(f"NaN values in {col}: {nan_count} rows")


def check_price_continuity(df: pd.DataFrame, result: ValidationResult):
    """Check for unrealistic price jumps (potential data errors or unadjusted splits)."""
    if len(df) < 2:
        return
    
    df = df.copy()
    df["pct_change"] = df["close"].pct_change().abs()
    
    # Flag > 20% daily moves (could be data error or unadjusted corporate action)
    large_moves = df[df["pct_change"] > 0.20]
    if len(large_moves) > 0:
        result.add_warning(f"Large price moves (>20%): {len(large_moves)} bars")
        for _, row in large_moves.head(3).iterrows():
            result.add_warning(f"  {row['date']}: {row['pct_change']*100:.1f}%")


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Run all validation checks on a DataFrame.
    Returns ValidationResult with errors and warnings.
    """
    result = ValidationResult()
    
    if df is None or len(df) == 0:
        result.add_error("DataFrame is empty or None")
        return result
    
    # Required columns check
    required = ["timestamp_ms", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        result.add_error(f"Missing required columns: {missing}")
        return result
    
    # Run all validators
    check_duplicate_timestamps(df, result)
    check_nan_values(df, result)
    check_invalid_ohlc(df, result)
    check_zero_volume(df, result)
    check_timezone_consistency(df, result)
    check_missing_dates(df, result)
    check_price_continuity(df, result)
    
    return result


def validate_and_report(df: pd.DataFrame, name: str = "data") -> Tuple[bool, List[str], List[str]]:
    """
    Validate DataFrame and return tuple of (valid, errors, warnings).
    Convenience function for quick validation.
    """
    result = validate_dataframe(df)
    
    if result.valid:
        logger.info(f"{name} validation passed ✓")
    else:
        logger.error(f"{name} validation failed")
        logger.error(result.summary())
    
    return result.valid, result.errors, result.warnings
