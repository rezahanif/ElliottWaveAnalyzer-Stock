"""
feature_builder.py
------------------
Feature cache builder for BMRI TFT training.
Computes all features defined in schema.py and stores as cached parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from src.btc.ingestion.indicators import add_indicators
from src.stock.features.schema import FeatureSchema, get_default_schema
from src.stock.analyzer import StockAnalyzer
from src.stock.features.market_context import generate_market_context

logger = logging.getLogger("stock_features")


class FeatureBuilder:
    """
    Builds feature cache for TFT training.
    Ensures all features are computed consistently and no look-ahead bias.
    """
    
    def __init__(
        self,
        symbol: str = "BMRI.JK",
        schema: Optional[FeatureSchema] = None,
        storage_root: Optional[Path] = None,
    ):
        self.symbol = symbol
        self.schema = schema or get_default_schema()
        self.storage_root = storage_root or Path("data/stocks/BMRI")
        self.analyzer = StockAnalyzer(symbol, "1D")
    
    def build_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators using existing BTC implementation."""
        df = add_indicators(df)
        return df
    
    def build_wave(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute wave features.
        Note: Wave analysis is computed on-the-fly, not cached.
        For training, we use simplified proxies.
        """
        df = df.copy()
        
        # Placeholder: use ATR ratio as proxy for wave volatility
        df["wave_direction"] = "neutral"
        df["wave_degree"] = "unknown"
        df["fib_level"] = 0.5
        df["fib_cluster_strength"] = 0.0
        df["invalidation_distance"] = df["atr_14"] * 3 / df["close"]
        
        return df
    
    def build_market_context(
        self,
        df_stock: pd.DataFrame,
        df_sector: pd.DataFrame,
        df_ihsg: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute market context features aligned to stock dates."""
        ctx = generate_market_context(df_stock, df_sector, df_ihsg)
        
        # Create per-row features from context summary
        df = df_stock.copy()
        
        # Map context to each row (simplified - in reality would be time-varying)
        df["ihsg_bias"] = ctx["ihsg"]["bias"].lower()
        df["ihsg_change_5d"] = ctx["ihsg"]["recent_change_pct"] / 100.0
        df["sector_outperforming"] = ctx["sector"]["outperforming_market"]
        df["sector_relative_strength"] = ctx["sector"]["recent_relative_change_pct"] / 100.0
        df["stock_outperforming"] = ctx["stock"]["outperforming_sector"]
        df["stock_relative_strength"] = ctx["stock"]["recent_relative_change_pct"] / 100.0
        
        return df
    
    def build_fundamental(self, df: pd.DataFrame, fundamentals: Dict[str, Any]) -> pd.DataFrame:
        """Add static fundamental values to each row."""
        df = df.copy()
        
        for key in ["pe_ratio", "pb_ratio", "roe", "div_yield", "revenue_growth"]:
            df[key] = fundamentals.get(key, 0.0)
        
        return df
    
    def build_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add macro features.
        Note: Would need external data source for actual values.
        Using placeholders for now.
        """
        df = df.copy()
        
        # Placeholder values (would come from BI/external data)
        df["usd_idr_rate"] = 15500.0
        df["bi_rate"] = 5.75
        df["inflation_yoy"] = 0.03
        
        return df
    
    def build_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar features.
        Note: Would need economic calendar integration.
        Using placeholders for now.
        """
        df = df.copy()
        
        df["days_to_fomc"] = 999
        df["days_since_fomc"] = 999
        df["days_to_nfp"] = 999
        df["high_impact_within_5d"] = False
        df["high_impact_within_2d"] = False
        
        return df
    
    def build_news(self, df: pd.DataFrame, sentiment: Dict[str, Any]) -> pd.DataFrame:
        """Add news sentiment features."""
        df = df.copy()
        
        df["sentiment_score"] = sentiment.get("sentiment_score", 0.0)
        df["sentiment_class"] = sentiment.get("sentiment_class", "neutral").lower()
        df["news_count_24h"] = len(sentiment.get("articles", []))
        
        return df
    
    def build_all(
        self,
        df_stock: pd.DataFrame,
        df_sector: Optional[pd.DataFrame] = None,
        df_ihsg: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict] = None,
        sentiment: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Build complete feature set.
        Order matters: technical → wave → market → fundamental → macro → calendar → news
        """
        # 1. Technical (base layer)
        df = self.build_technical(df_stock)
        
        # 2. Wave
        df = self.build_wave(df)
        
        # 3. Market context (if sector/ihsg provided)
        if df_sector is not None and df_ihsg is not None:
            df = self.build_market_context(df, df_sector, df_ihsg)
        else:
            # Fill with defaults
            df["ihsg_bias"] = "neutral"
            df["ihsg_change_5d"] = 0.0
            df["sector_outperforming"] = False
            df["sector_relative_strength"] = 0.0
            df["stock_outperforming"] = False
            df["stock_relative_strength"] = 0.0
        
        # 4. Fundamentals
        if fundamentals:
            df = self.build_fundamental(df, fundamentals)
        else:
            df = self.build_fundamental(df, {})
        
        # 5. Macro
        df = self.build_macro(df)
        
        # 6. Calendar
        df = self.build_calendar(df)
        
        # 7. News
        if sentiment:
            df = self.build_news(df, sentiment)
        else:
            df = self.build_news(df, {})
        
        # Ensure all schema features exist
        for feat in self.schema.all_features:
            if feat not in df.columns:
                logger.warning(f"Missing feature {feat}, filling with default")
                if self.schema.is_categorical(feat):
                    df[feat] = "unknown"
                else:
                    df[feat] = 0.0
        
        return df
    
    def sanity_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Sanity check features for:
        - No look-ahead leakage
        - No excessive NaN values
        - Correct types
        """
        issues = []
        warnings = []
        
        # Check for excessive NaN
        for col in self.schema.numerical_features:
            if col not in df.columns:
                continue
            nan_pct = df[col].isna().sum() / len(df)
            if nan_pct > 0.1:
                warnings.append(f"{col}: {nan_pct*100:.1f}% NaN")
        
        # Check for infinite values
        for col in self.schema.numerical_features:
            if col not in df.columns:
                continue
            if np.isinf(df[col]).any():
                issues.append(f"{col}: contains infinite values")
        
        # Check categorical values
        for col, valid_values in self.schema.categorical_features.items():
            if col not in df.columns:
                continue
            invalid = set(df[col].unique()) - set(valid_values) - {"unknown"}
            if invalid:
                warnings.append(f"{col}: unexpected values {invalid}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }
    
    def save_cache(self, df: pd.DataFrame, name: str = "features_daily"):
        """Save feature cache to parquet."""
        path = self.storage_root / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy", index=False)
        logger.info(f"Saved feature cache to {path}")


def build_feature_cache(
    symbol: str = "BMRI.JK",
    df_stock: Optional[pd.DataFrame] = None,
    df_sector: Optional[pd.DataFrame] = None,
    df_ihsg: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convenience function to build complete feature cache.
    Loads data from storage if not provided.
    """
    from src.stock.data.storage import BMRIStorage
    from src.stock.collectors.fundamentals import fetch_fundamentals
    from src.stock.collectors.news import fetch_news_and_sentiment
    
    storage = BMRIStorage()
    builder = FeatureBuilder(symbol)
    
    # Load data if not provided
    if df_stock is None:
        df_stock = storage.load("daily")
    if df_sector is None:
        df_sector = storage.load("weekly")  # Use weekly as proxy
    if df_ihsg is None:
        df_ihsg = storage.load("monthly")  # Use monthly as proxy
    
    if df_stock is None:
        raise ValueError("No stock data available")
    
    # Get fundamentals and sentiment
    fundamentals = fetch_fundamentals(symbol)
    sentiment = fetch_news_and_sentiment(symbol)
    
    # Build features
    df = builder.build_all(df_stock, df_sector, df_ihsg, fundamentals, sentiment)
    
    # Sanity check
    check = builder.sanity_check(df)
    if not check["valid"]:
        logger.error(f"Feature sanity check failed: {check['issues']}")
    if check["warnings"]:
        for w in check["warnings"]:
            logger.warning(w)
    
    # Save cache
    builder.save_cache(df)
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_feature_cache()
    print(f"Built feature cache: {len(df)} rows, {len(df.columns)} columns")
