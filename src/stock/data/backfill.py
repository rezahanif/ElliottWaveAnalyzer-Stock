"""
backfill.py
-----------
Historical data backfill and incremental update for BMRI.
Target: 10 years, minimum: 5 years.
Implements provider fallback chain with graceful degradation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.stock.data.storage import BMRIStorage, BMRIMetadata
from src.stock.data.providers import ProviderChain
from src.stock.data.corporate_actions import CorporateActionRegistry
from src.stock.data.validators import validate_and_report

logger = logging.getLogger("stock_backfill")

# Target history
TARGET_YEARS = 10
MIN_YEARS = 5


class BMRIBackfill:
    """
    Manages historical data backfill and incremental updates for BMRI.
    """
    
    def __init__(self, storage: Optional[BMRIStorage] = None):
        self.storage = storage or BMRIStorage()
        self.provider = ProviderChain("BMRI.JK")
        self.corp_actions = CorporateActionRegistry("BMRI.JK")
    
    def _get_start_date(self, target_years: int = TARGET_YEARS) -> str:
        """Calculate start date for backfill."""
        start = datetime.utcnow() - timedelta(days=target_years * 365)
        return start.strftime("%Y-%m-%d")
    
    def backfill(self, force: bool = False) -> bool:
        """
        Perform historical backfill.
        
        Args:
            force: If True, re-download even if data exists.
        
        Returns:
            True if successful, False otherwise.
        """
        existing = self.storage.load("daily")
        
        if existing is not None and len(existing) > 0 and not force:
            # Check if we have enough history
            dates = pd.to_datetime(existing["date"])
            years_covered = (dates.max() - dates.min()).days / 365.0
            
            if years_covered >= MIN_YEARS:
                logger.info(f"Existing data covers {years_covered:.1f} years (min: {MIN_YEARS})")
                logger.info("Use force=True to re-download")
                return True
        
        start_date = self._get_start_date(TARGET_YEARS)
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        logger.info(f"Starting backfill from {start_date} to {end_date}")
        
        # Fetch from provider chain
        df = self.provider.fetch(start_date, end_date, self.storage.metadata)
        
        if df is None or len(df) == 0:
            logger.error("Backfill failed: no data from any provider")
            self.storage.metadata.save()
            return False
        
        # Validate data
        valid, errors, warnings = validate_and_report(df, "backfill")
        
        if not valid:
            logger.error(f"Backfill validation failed: {errors}")
            # Still save metadata about the attempt
            self.storage.metadata.log_validation(errors, warnings)
            self.storage.metadata.save()
            return False
        
        # Apply corporate actions
        df = self.corp_actions.apply_all(df)
        
        # Save to storage
        self.storage.save(df, "daily")
        
        # Generate weekly and monthly
        self.storage.generate_all_frequencies()
        
        # Update corporate actions in metadata
        self.storage.metadata.data["corporate_actions"] = self.corp_actions.to_metadata()
        self.storage.metadata.log_validation([], warnings)
        self.storage.metadata.save()
        
        logger.info(f"Backfill complete: {len(df)} daily bars")
        logger.info(self.provider.summary())
        
        return True
    
    def incremental_update(self) -> int:
        """
        Fetch new bars since last stored timestamp.
        Appends to existing data, never re-downloads full history.
        
        Returns:
            Number of new bars added (0 if none or failed).
        """
        existing = self.storage.load("daily")
        
        if existing is None or len(existing) == 0:
            logger.warning("No existing data - run backfill first")
            return 0
        
        # Get last timestamp
        last_ts = existing["timestamp_ms"].max()
        last_date = datetime.utcfromtimestamp(last_ts / 1000)
        
        # Skip if last update was today (already up to date)
        now = datetime.utcnow()
        if last_date.date() == now.date():
            logger.info("Data already up to date")
            return 0
        
        # Fetch from last+1 day to now
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
        
        logger.info(f"Incremental update: {start_date} to {end_date}")
        
        df = self.provider.fetch(start_date, end_date, self.storage.metadata)
        
        if df is None or len(df) == 0:
            logger.warning("No new data from providers")
            return 0
        
        # Validate new data
        valid, errors, warnings = validate_and_report(df, "incremental")
        if not valid:
            logger.error(f"Incremental validation failed: {errors}")
            return 0
        
        # Apply corporate actions
        df = self.corp_actions.apply_all(df)
        
        # Append to existing
        bars_added = self.storage.append(df, "daily")
        
        if bars_added > 0:
            # Regenerate weekly/monthly
            self.storage.generate_all_frequencies()
            
            # Update metadata
            self.storage.metadata.save()
            
            logger.info(f"Incremental update complete: {bars_added} new bars")
        else:
            logger.info("No new bars to add (likely duplicates)")
        
        return bars_added
    
    def check_coverage(self) -> dict:
        """
        Check data coverage and report.
        
        Returns:
            Dict with coverage details.
        """
        coverage = self.storage.metadata.data.get("coverage", {})
        
        daily = coverage.get("daily", {})
        if daily.get("count", 0) == 0:
            return {"status": "no_data", "message": "No data loaded"}
        
        start = daily.get("start")
        end = daily.get("end")
        count = daily.get("count", 0)
        
        if start and end:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            years = (end_dt - start_dt).days / 365.0
        else:
            years = 0
        
        status = "ok" if years >= MIN_YEARS else "insufficient"
        
        return {
            "status": status,
            "years": round(years, 2),
            "target": TARGET_YEARS,
            "minimum": MIN_YEARS,
            "start": start,
            "end": end,
            "count": count,
            "meets_minimum": years >= MIN_YEARS,
        }
    
    def regenerate_feature_cache(self):
        """
        Regenerate feature cache after data update.
        Features: RSI, MACD, EMA, ATR, OBV, ADX, wave labels, fib levels, market context.
        """
        from src.btc.ingestion.indicators import add_indicators
        
        df = self.storage.load("daily")
        if df is None:
            logger.warning("No data to regenerate features")
            return
        
        # Add technical indicators
        df_features = add_indicators(df)
        
        # Save feature cache
        cache_path = self.storage.root / "features_daily.parquet"
        df_features.to_parquet(cache_path, compression="snappy", index=False)
        
        logger.info(f"Feature cache regenerated: {len(df_features)} rows, {len(df_features.columns)} columns")
        
        # Update metadata
        self.storage.metadata.data["feature_cache"] = {
            "last_regenerated": datetime.utcnow().isoformat() + "Z",
            "rows": len(df_features),
            "columns": list(df_features.columns),
        }
        self.storage.metadata.save()


def run_backfill():
    """CLI entry point for backfill."""
    backfill = BMRIBackfill()
    success = backfill.backfill()
    
    if success:
        coverage = backfill.check_coverage()
        print(f"Coverage: {coverage}")
        
        # Regenerate feature cache
        backfill.regenerate_feature_cache()
    
    return success


def run_incremental():
    """CLI entry point for incremental update."""
    backfill = BMRIBackfill()
    bars_added = backfill.incremental_update()
    
    if bars_added > 0:
        backfill.regenerate_feature_cache()
    
    return bars_added


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--incremental":
        run_incremental()
    else:
        run_backfill()
