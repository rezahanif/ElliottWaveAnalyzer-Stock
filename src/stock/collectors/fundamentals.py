"""
fundamentals.py
---------------
Collects fundamental ratios (P/E, P/B, ROE, Dividend Yield) for stock analysis.
Loads static values from config/stock.yaml, with support for future API integration.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger("stock_fundamentals")


def fetch_fundamentals(symbol: str, config_path: str = "config/stock.yaml") -> Dict[str, Any]:
    """
    Retrieve fundamental indicators.
    Default fallback returns static/semi-static metrics configured in config/stock.yaml.
    """
    default_metrics = {
        "symbol": symbol.upper(),
        "pe_ratio": 11.5,       # Historical average BMRI trailing P/E
        "pb_ratio": 2.1,        # Historical average BMRI price-to-book
        "roe": 0.185,           # Historical average BMRI ROE (18.5%)
        "div_yield": 0.045,     # Dividend yield (4.5%)
        "revenue_growth": 0.08,  # Year-on-year growth (8%)
    }

    if not os.path.exists(config_path):
        return default_metrics

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # Look for symbol-specific fundamental configurations
        fundamentals_cfg = cfg.get("fundamentals", {})
        if symbol.upper() in fundamentals_cfg:
            metrics = default_metrics.copy()
            metrics.update(fundamentals_cfg[symbol.upper()])
            logger.info(f"Loaded fundamentals for {symbol} from config")
            return metrics
    except Exception as e:
        logger.warning(f"Failed to parse fundamentals from config: {e}")

    logger.info(f"Using default fundamental metrics for {symbol}")
    return default_metrics
