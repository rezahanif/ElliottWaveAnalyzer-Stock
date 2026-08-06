"""
test_schema_scalers.py
----------------------
Guards the schema-driven scaler/feature refactor (feat/orderbook-layer-and-tft-fix):
- stock scaler classes per feature must match the old hardcoded heavy_tailed rule
  for every live (schema) feature
- BTC feature lists must match the old hardcoded lists exactly (order + content)
- features absent from the schema keep the old StandardScaler default
  (predict/evaluate scale time_idx that way today)
"""
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.stock.features.schema import get_default_schema
from src.btc.wave_model.schema import get_btc_schema

# Old hardcoded rule (train.py/predict.py/evaluate.py before refactor)
OLD_HEAVY_TAILED = {
    "obv", "volume", "ema_20", "ema_50", "ema_200",
    "bb_upper", "bb_lower", "close", "open", "high", "low",
}

# Old hardcoded BTC lists (model.py before refactor)
OLD_KNOWN_FUTURE_REALS = [
    "lunar_phase_sin", "lunar_phase_cos", "lunar_anomalistic_normalized",
    "lunar_node_distance", "mercury_retrograde",
    "aspect_jupiter_uranus_intensity", "aspect_mars_uranus_intensity",
    "days_to_fomc", "days_since_last_fomc", "days_to_nfp",
    "high_impact_within_5d", "high_impact_within_2d", "post_event_window",
]
OLD_UNKNOWN_PAST_REALS = [
    "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
    "rsi_14", "macd_line", "macd_signal", "macd_hist", "atr_14_norm",
    "bb_width", "pattern_confidence", "wave_match_confidence",
]
OLD_UNKNOWN_PAST_CATEGORICALS = [
    "structure_token_id", "wave_degree_id", "pattern_type_id",
    "correction_or_impulse_type_id",
]


def test_stock_scaler_classes_match_old_rule():
    schema = get_default_schema()
    scalers = schema.scalers_for(schema.numerical_features)
    # bool features (no normalize flag) are intentionally unscaled ("none")
    expected = {f for f in schema.numerical_features if schema.scaler_for[f] != "none"}
    assert set(scalers.keys()) == expected
    for feat, scaler in scalers.items():
        if feat in OLD_HEAVY_TAILED:
            assert isinstance(scaler, RobustScaler), f"{feat} should be RobustScaler"
        else:
            assert isinstance(scaler, StandardScaler), f"{feat} should be StandardScaler"


def test_stock_schema_marks_old_heavy_tailed_features_robust():
    schema = get_default_schema()
    for feat in OLD_HEAVY_TAILED & set(schema.numerical_features):
        assert schema.scaler_for[feat] == "robust", f"{feat} not marked robust"


def test_absent_features_default_to_standard_scaler():
    schema = get_default_schema()
    # predict/evaluate scale time_idx today; it is not in the schema
    scalers = schema.scalers_for(["time_idx"])
    assert isinstance(scalers["time_idx"], StandardScaler)


def test_btc_lists_match_old_hardcoded_lists():
    schema = get_btc_schema()
    assert schema.get_features_for_group("known_future") == OLD_KNOWN_FUTURE_REALS
    assert schema.get_features_for_group("unknown_past") == OLD_UNKNOWN_PAST_REALS
    assert schema.get_features_for_group("categoricals") == OLD_UNKNOWN_PAST_CATEGORICALS


def test_btc_target_and_group_unchanged():
    import src.btc.wave_model.schema as s
    assert "close_pct_change"  # TARGET lives in model.py; schema has no target
    assert s.BTC_DEFAULT_SCHEMA["known_future"]["features"]
