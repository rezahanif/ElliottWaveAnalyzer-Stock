"""
schema.py
---------
Config-driven feature schema for BTC TFT training — mirrors src/stock/features/schema.py
so the coin-agnostic recipe UI can drive both asset classes through one FeatureSchema API.
Groups: Known Future Reals, Unknown Past Reals, Unknown Past Categoricals.

Replaces the hardcoded KNOWN_FUTURE_REALS / UNKNOWN_PAST_REALS /
UNKNOWN_PAST_CATEGORICALS lists that used to live in model.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.stock.features.schema import FeatureSchema  # generic class, shared lookup logic

# ponytail: FeatureSchema class lives under stock/; if a third asset class
# appears, hoist it to a shared src/common module and re-export from both sides.

BTC_DEFAULT_SCHEMA = {
    "known_future": {
        "description": "Known-future reals (astro + economic calendar) — predictable at inference",
        "features": [
            {"name": "lunar_phase_sin", "type": "float"},
            {"name": "lunar_phase_cos", "type": "float"},
            {"name": "lunar_anomalistic_normalized", "type": "float"},
            {"name": "lunar_node_distance", "type": "float"},
            {"name": "mercury_retrograde", "type": "float"},
            {"name": "aspect_jupiter_uranus_intensity", "type": "float"},
            {"name": "aspect_mars_uranus_intensity", "type": "float"},
            {"name": "days_to_fomc", "type": "float"},
            {"name": "days_since_last_fomc", "type": "float"},
            {"name": "days_to_nfp", "type": "float"},
            {"name": "high_impact_within_5d", "type": "float"},
            {"name": "high_impact_within_2d", "type": "float"},
            {"name": "post_event_window", "type": "float"},
        ],
    },
    "unknown_past": {
        "description": "Unknown-past reals (price/indicator) — observed, not forecastable ahead",
        "features": [
            {"name": "open_norm", "type": "float"},
            {"name": "high_norm", "type": "float"},
            {"name": "low_norm", "type": "float"},
            {"name": "close_norm", "type": "float"},
            {"name": "volume_norm", "type": "float"},
            {"name": "rsi_14", "type": "float"},
            {"name": "macd_line", "type": "float"},
            {"name": "macd_signal", "type": "float"},
            {"name": "macd_hist", "type": "float"},
            {"name": "atr_14_norm", "type": "float"},
            {"name": "bb_width", "type": "float"},
            {"name": "pattern_confidence", "type": "float"},
            {"name": "wave_match_confidence", "type": "float"},
        ],
    },
    "categoricals": {
        "description": "Unknown-past categoricals (structure/wave/pattern tokens)",
        "features": [
            {"name": "structure_token_id", "type": "categorical"},
            {"name": "wave_degree_id", "type": "categorical"},
            {"name": "pattern_type_id", "type": "categorical"},
            {"name": "correction_or_impulse_type_id", "type": "categorical"},
        ],
    },
}


def get_btc_schema(config_path: Optional[Path] = None) -> FeatureSchema:
    """Get BTC feature schema instance (YAML override or defaults)."""
    if config_path is not None:
        return FeatureSchema(config_path=config_path)
    return FeatureSchema(schema=BTC_DEFAULT_SCHEMA)


if __name__ == "__main__":
    schema = get_btc_schema()
    print(schema.summary())
