"""
schema.py
---------
Config-driven feature schema for BMRI TFT training.
Groups: Technical, Wave, Market Context, Fundamental, Macro, Calendar, News.

This replaces hardcoded feature lists in dataset.py with a declarative schema
that can be modified without touching model code.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Default feature schema
DEFAULT_SCHEMA = {
    "technical": {
        "description": "Price-based technical indicators",
        "features": [
            {"name": "rsi_14", "type": "float", "normalize": True},
            {"name": "macd", "type": "float", "normalize": True},
            {"name": "macd_signal", "type": "float", "normalize": True},
            {"name": "macd_hist", "type": "float", "normalize": True},
            {"name": "atr_14", "type": "float", "normalize": True},
            {"name": "atr_20", "type": "float", "normalize": True},
            {"name": "ema_20", "type": "float", "normalize": False},
            {"name": "ema_50", "type": "float", "normalize": False},
            {"name": "ema_200", "type": "float", "normalize": False},
            {"name": "bb_upper", "type": "float", "normalize": False},
            {"name": "bb_lower", "type": "float", "normalize": False},
            {"name": "bb_width", "type": "float", "normalize": True},
            {"name": "obv", "type": "float", "normalize": True},
            {"name": "adx_14", "type": "float", "normalize": True},
        ],
    },
    "wave": {
        "description": "Elliott Wave and Fibonacci features",
        "features": [
            {"name": "wave_direction", "type": "categorical", "values": ["bullish", "bearish", "neutral"]},
            {"name": "wave_degree", "type": "categorical", "values": ["impulse", "correction", "unknown"]},
            {"name": "fib_level", "type": "float", "normalize": True},
            {"name": "fib_cluster_strength", "type": "float", "normalize": True},
            {"name": "invalidation_distance", "type": "float", "normalize": True},
        ],
    },
    "market_context": {
        "description": "Market cascade features (IHSG → Sector → Stock)",
        "features": [
            {"name": "ihsg_bias", "type": "categorical", "values": ["bullish", "bearish", "neutral"]},
            {"name": "ihsg_change_5d", "type": "float", "normalize": True},
            {"name": "sector_outperforming", "type": "bool"},
            {"name": "sector_relative_strength", "type": "float", "normalize": True},
            {"name": "stock_outperforming", "type": "bool"},
            {"name": "stock_relative_strength", "type": "float", "normalize": True},
        ],
    },
    "fundamental": {
        "description": "Company fundamentals (static or slowly-changing)",
        "features": [
            {"name": "pe_ratio", "type": "float", "normalize": True},
            {"name": "pb_ratio", "type": "float", "normalize": True},
            {"name": "roe", "type": "float", "normalize": True},
            {"name": "div_yield", "type": "float", "normalize": True},
            {"name": "revenue_growth", "type": "float", "normalize": True},
        ],
    },
    "macro": {
        "description": "Macro economic indicators",
        "features": [
            {"name": "usd_idr_rate", "type": "float", "normalize": True},
            {"name": "bi_rate", "type": "float", "normalize": True},  # Bank Indonesia rate
            {"name": "inflation_yoy", "type": "float", "normalize": True},
        ],
    },
    "calendar": {
        "description": "Economic calendar proximity",
        "features": [
            {"name": "days_to_fomc", "type": "int", "normalize": True},
            {"name": "days_since_fomc", "type": "int", "normalize": True},
            {"name": "days_to_nfp", "type": "int", "normalize": True},
            {"name": "high_impact_within_5d", "type": "bool"},
            {"name": "high_impact_within_2d", "type": "bool"},
        ],
    },
    "news": {
        "description": "News sentiment features",
        "features": [
            {"name": "sentiment_score", "type": "float", "normalize": True},
            {"name": "sentiment_class", "type": "categorical", "values": ["positive", "neutral", "negative"]},
            {"name": "news_count_24h", "type": "int", "normalize": True},
        ],
    },
}


class FeatureSchema:
    """
    Config-driven feature schema for TFT training.
    Loads from YAML or uses defaults.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.schema = self._load_schema()
        self._build_lookups()
    
    def _load_schema(self) -> Dict[str, Any]:
        if self.config_path and self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        return DEFAULT_SCHEMA
    
    def _build_lookups(self):
        """Build fast lookup tables."""
        self.all_features: List[str] = []
        self.feature_to_group: Dict[str, str] = {}
        self.categorical_features: Dict[str, List[str]] = {}
        self.numerical_features: List[str] = []
        
        for group_name, group_data in self.schema.items():
            for feat in group_data.get("features", []):
                name = feat["name"]
                self.all_features.append(name)
                self.feature_to_group[name] = group_name
                
                if feat.get("type") == "categorical":
                    self.categorical_features[name] = feat.get("values", [])
                else:
                    self.numerical_features.append(name)
    
    def get_features_for_group(self, group: str) -> List[str]:
        """Get feature names for a specific group."""
        if group not in self.schema:
            return []
        return [f["name"] for f in self.schema[group].get("features", [])]
    
    def get_categorical_values(self, feature: str) -> List[str]:
        """Get possible values for a categorical feature."""
        return self.categorical_features.get(feature, [])
    
    def is_categorical(self, feature: str) -> bool:
        """Check if feature is categorical."""
        return feature in self.categorical_features
    
    def get_input_dim(self) -> int:
        """Get total number of input features."""
        return len(self.all_features)
    
    def to_config(self) -> Dict[str, Any]:
        """Export schema to config dict."""
        return self.schema
    
    def save(self, path: Path):
        """Save schema to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.schema, f, default_flow_style=False)
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = ["Feature Schema Summary", "=" * 40]
        for group, data in self.schema.items():
            feat_count = len(data.get("features", []))
            lines.append(f"{group}: {feat_count} features")
        lines.append(f"Total: {len(self.all_features)} features")
        lines.append(f"Categorical: {len(self.categorical_features)}")
        lines.append(f"Numerical: {len(self.numerical_features)}")
        return "\n".join(lines)


def get_default_schema() -> FeatureSchema:
    """Get default feature schema instance."""
    return FeatureSchema()


if __name__ == "__main__":
    schema = FeatureSchema()
    print(schema.summary())
