"""
model.py
--------
Defines the Temporal Fusion Transformer (TFT) model structure and helper
functions to construct PyTorch Forecasting TimeSeriesDataSets.
"""

from __future__ import annotations

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from src.btc.wave_model.schema import get_btc_schema

# ── Feature Configurations ──────────────────────────────────────────────────
# Schema-driven (see schema.py) — same FeatureSchema pattern as the stock side,
# so the coin-agnostic recipe UI drives both asset classes through one API.
_schema = get_btc_schema()

KNOWN_FUTURE_REALS = _schema.get_features_for_group("known_future")
UNKNOWN_PAST_REALS = _schema.get_features_for_group("unknown_past")
UNKNOWN_PAST_CATEGORICALS = _schema.get_features_for_group("categoricals")

TARGET = "close_pct_change"
GROUP_IDS = ["asset_timeframe"]
TIME_IDX = "time_idx"


def prepare_df_for_tft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the raw dataset DataFrame for TFT training/inference.
    - Fills early technical indicator NaNs with 0.0
    - Encodes ID columns as strings for categorical variables
    """
    df = df.copy()

    # Default values matching DatasetBuilder initializations
    defaults = {
        "structure_token_id": -1,
        "wave_degree_id": -1,
        "pattern_type_id": 0,
        "correction_or_impulse_type_id": 0,
    }

    # Convert categoricals to string to ensure categorical embedding
    for col in UNKNOWN_PAST_CATEGORICALS:
        if col in df.columns:
            default_val = defaults.get(col, 0)
            df[col] = df[col].fillna(default_val).astype(int).astype(str)

    # Fill NaNs in numeric features with 0.0 (e.g. BB, RSI warmup)
    all_reals = KNOWN_FUTURE_REALS + UNKNOWN_PAST_REALS
    for col in all_reals:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Target close_pct_change must be filled with 0.0 to avoid NA validation failures in PyTorch Forecasting
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].fillna(0.0)

    return df


def create_tft_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 90,
    max_prediction_length: int = 60,
) -> TimeSeriesDataSet:
    """
    Create a TimeSeriesDataSet for the TFT model.
    """
    return TimeSeriesDataSet(
        df,
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=GROUP_IDS,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["asset_timeframe"],
        time_varying_known_reals=KNOWN_FUTURE_REALS,
        time_varying_unknown_reals=UNKNOWN_PAST_REALS,
        time_varying_unknown_categoricals=UNKNOWN_PAST_CATEGORICALS,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def build_tft_model(
    dataset: TimeSeriesDataSet,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    attention_head_size: int = 2,
    dropout: float = 0.1,
) -> TemporalFusionTransformer:
    """
    Build a TemporalFusionTransformer model from a TimeSeriesDataSet config.
    """
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        loss=QuantileLoss([0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=4,
    )
