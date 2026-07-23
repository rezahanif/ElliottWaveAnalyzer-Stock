"""
analyzer.py
-----------
Instrument-agnostic StockAnalyzer adapter.
Applies wave conf pivots, trendlines, geometric patterns, and Fibonacci
projections to generic OHLCV data.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse clean, generic math engines from src/btc
from src.btc.pivots.zigzag import ZigZagDetector, ZigZagResult
from src.btc.pivots.pattern_detector import PatternDetector, PatternMatch
from src.btc.fib_engine.trendline import Trendline, fit_trendline
from src.btc.fib_engine.fibonacci import FibonacciEngine, ClusterResult
from src.btc.confluence.cluster_check import is_confluent
from src.stock.predict import predict_tft
from src.stock.forecast.fusion import fuse, FusionResult


class StockAnalyzer:
    """
    Adapter that takes arbitrary OHLCV data and computes technical layers,
    zigzag pivots, trendlines, geometric patterns, and Fibonacci zones.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1D",
        pattern_thresholds: Optional[dict] = None,
        rates_config: str = "config/completion_rates.yaml",
        rules_config: str = "config/correction_rules.yaml",
    ):
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.zigzag_detector = ZigZagDetector(timeframe=self.timeframe)
        self.pattern_detector = PatternDetector(thresholds=pattern_thresholds)
        self.fib_engine = FibonacciEngine(rates_config=rates_config, rules_config=rules_config)

    def calculate_volatility_layers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute dual ATR-based thresholds (wall_street_threshold_pct and behavioral_threshold_pct)
        required by the ZigZag state machine.
        """
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        cpd = {"1W": 1 / 7, "1D": 1, "4H": 6, "1H": 24}.get(self.timeframe, 1)

        atr_20_p = max(1, int(20 * cpd))
        atr_14_p = max(1, int(14 * cpd))

        df["prev_close"] = df["close"].shift(1)
        df["h_l"] = df["high"] - df["low"]
        df["h_pc"] = (df["high"] - df["prev_close"]).abs()
        df["l_pc"] = (df["low"] - df["prev_close"]).abs()
        df["true_range"] = df[["h_l", "h_pc", "l_pc"]].max(axis=1)

        df["atr_20"] = df["true_range"].ewm(alpha=1 / atr_20_p, adjust=False).mean()
        df["wall_street_threshold_pct"] = round((df["atr_20"] * 3 / df["close"]) * 100, 2)

        df["atr_14"] = df["true_range"].ewm(alpha=1 / atr_14_p, adjust=False).mean()
        df["behavioral_threshold_pct"] = round((df["atr_14"] * 1.5 / df["close"]) * 100, 2)

        # Drop rows with NaN if any (from shift(1))
        return df.dropna(subset=["wall_street_threshold_pct", "behavioral_threshold_pct"]).copy()

    def analyze(self, df: pd.DataFrame, version: str = "v5_relaxed") -> Dict[str, Any]:
        """
        Run the complete technical pipeline on generic OHLCV DataFrame.
        """
        # 1. Volatility layers
        df_layers = self.calculate_volatility_layers(df)
        if len(df_layers) < 50:
            raise ValueError(f"Insufficient data for analysis: {len(df_layers)} rows")

        # 2. Run ZigZag
        zigzag_res = self.zigzag_detector.run(df_layers, asset=self.symbol)
        macro_pivots = zigzag_res.macro
        micro_pivots = zigzag_res.micro

        # 3. Fit trendlines and detect patterns if pivots exist
        pattern_match = None
        if len(macro_pivots) >= 4:
            high_pivots = [p for p in macro_pivots if p.is_high()]
            low_pivots = [p for p in macro_pivots if p.is_low()]
            if len(high_pivots) >= 2 and len(low_pivots) >= 2:
                res_tl = fit_trendline(high_pivots[-2:])
                sup_tl = fit_trendline(low_pivots[-2:])
                if res_tl and sup_tl:
                    pattern_match = self.pattern_detector.detect(res_tl, sup_tl)

        # 4. Fibonacci targets
        cluster = None
        invalidation = None
        direction = None
        if len(macro_pivots) >= 4:
            highs = [p for p in macro_pivots if p.is_high()]
            lows = [p for p in macro_pivots if p.is_low()]
            if highs and lows:
                last_pivot = macro_pivots[-1]
                if last_pivot.is_high():
                    direction = "bearish"
                    c_top_val = last_pivot.price
                    b_low_pivot = max((p for p in lows if p.bar_index < last_pivot.bar_index), key=lambda p: p.bar_index, default=None)
                    if b_low_pivot:
                        b_low_val = b_low_pivot.price
                        a_high_pivot = max((p for p in highs if p.bar_index < b_low_pivot.bar_index), key=lambda p: p.bar_index, default=None)
                        ab_range = abs(a_high_pivot.price - b_low_val) if a_high_pivot else None
                        a_price = a_high_pivot.price if a_high_pivot else None
                        invalidation = round(last_pivot.price * (1 + self.fib_engine.invalidation_buffer), 2)
                        cluster = self.fib_engine.dual_cluster(
                            c_top=c_top_val,
                            b_low=b_low_val,
                            direction=direction,
                            ab_range=ab_range,
                            a_price=a_price,
                            version=version,
                        )
                else:
                    direction = "bullish"
                    c_top_val = last_pivot.price  # C bottom
                    b_low_pivot = max((p for p in highs if p.bar_index < last_pivot.bar_index), key=lambda p: p.bar_index, default=None)  # B high
                    if b_low_pivot:
                        b_low_val = b_low_pivot.price
                        a_low_pivot = max((p for p in lows if p.bar_index < b_low_pivot.bar_index), key=lambda p: p.bar_index, default=None)
                        ab_range = abs(a_low_pivot.price - b_low_val) if a_low_pivot else None
                        a_price = a_low_pivot.price if a_low_pivot else None
                        invalidation = round(last_pivot.price * (1 - self.fib_engine.invalidation_buffer), 2)
                        cluster = self.fib_engine.dual_cluster(
                            c_top=b_low_val,  # B high
                            b_low=c_top_val,  # C bottom
                            direction=direction,
                            ab_range=ab_range,
                            a_price=a_price,
                            version=version,
                        )

        # 5. Deterministic result (unchanged — Rule Engine decision authority)
        deterministic = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "df_layers": df_layers,
            "zigzag": zigzag_res,
            "pattern": pattern_match,
            "fibonacci": cluster,
            "invalidation": invalidation,
            "direction": direction,
        }

        # 6. Attempt TFT forecast (returns None if no checkpoint — graceful)
        tft_result = predict_tft(self.symbol, window_df=df_layers)

        # 7. Fusion Layer — pass-through when TFT unavailable
        fusion_result = fuse(deterministic, tft_result)

        # 8. Return deterministic fields + AI status fields
        #    Existing fields are byte-for-byte identical (pass-through guarantee)
        result = dict(deterministic)  # copy all deterministic fields
        result["ai_forecast_status"] = fusion_result.ai_forecast_status
        result["ai_forecast_reason"] = fusion_result.ai_forecast_reason
        result["ai_forecast_data"] = fusion_result.ai_forecast_data
        result["fusion"] = fusion_result
        return result
