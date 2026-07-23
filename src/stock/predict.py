"""
predict.py
----------
TFT forecast inference wrapper for the stock module.

Attempts to load a trained TFT checkpoint for the given symbol and produce
quantile forecasts. If no checkpoint exists, returns None gracefully —
the Fusion Layer handles this as "AI Forecast: Unavailable".
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger("stock_predict")

# Checkpoint directory — stock models live separately from BTC models
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"

# Log file for audit trail (Fix Plan requirement: confirm via log, not code inspection)
FORECAST_LOG = Path(__file__).resolve().parent.parent.parent / "logs" / "forecast.log"


def _log_forecast_attempt(symbol: str, result: str, detail: str = ""):
    """Write a timestamped entry to logs/forecast.log on every predict_tft() call."""
    FORECAST_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=None).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    line = f"[{ts}] predict_tft({symbol}) → {result}"
    if detail:
        line += f" — {detail}"
    with open(FORECAST_LOG, "a") as f:
        f.write(line + "\n")
    logger.info(line)


class TFTResult:
    """Container for TFT forecast output."""

    def __init__(
        self,
        symbol: str,
        quantiles: Dict[str, float],
        horizons: Dict[str, Dict[str, float]],
        model_version: str = "unknown",
    ):
        self.symbol = symbol
        self.quantiles = quantiles          # e.g. {"q10": ..., "q50": ..., "q90": ...}
        self.horizons = horizons            # e.g. {"7d": {"q10":..., "q50":..., "q90":...}, ...}
        self.model_version = model_version

    @property
    def direction(self) -> str:
        """Infer forecast direction from q50 median."""
        q50 = self.quantiles.get("q50", 0.0)
        if q50 > 0.001:
            return "bullish"
        elif q50 < -0.001:
            return "bearish"
        return "neutral"

    @property
    def confidence(self) -> float:
        """Confidence score 0.0–1.0 based on spread between q10 and q90."""
        q10 = self.quantiles.get("q10", 0.0)
        q90 = self.quantiles.get("q90", 0.0)
        spread = abs(q90 - q10)
        # Narrower spread = higher confidence; clamp to [0, 1]
        return max(0.0, min(1.0, 1.0 - spread))


def _build_inference_dataset(
    window_df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 20,
):
    """
    Build a TimeSeriesDataSet from the provided feature window using
    the SAME stock feature schema as train.py (not BTC's).
    Uses NaNLabelEncoder(add_nan=True) for all categoricals to handle
    unseen categories gracefully during inference.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder
    from src.stock.features.schema import get_default_schema

    schema = get_default_schema()

    df = window_df.copy()
    df["symbol"] = "BMRI_JK"
    df["time_idx"] = range(len(df))

    # Same target definition as train.py
    df["target_pct"] = df["close"].pct_change().shift(-1)
    df = df.ffill().bfill()

    time_varying_unknown_reals = ["target_pct"]
    for feat in schema.numerical_features:
        if feat in df.columns:
            time_varying_unknown_reals.append(feat)

    time_varying_known_categoricals = []
    for feat, values in schema.categorical_features.items():
        if feat in df.columns:
            time_varying_known_categoricals.append(feat)

    # Use NaNLabelEncoder(add_nan=True) for ALL categoricals — this maps
    # unseen categories to NaN index instead of crashing with IndexError
    categorical_encoders = {}
    for feat in time_varying_known_categoricals + ["symbol"]:
        categorical_encoders[feat] = NaNLabelEncoder(add_nan=True)

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target_pct",
        group_ids=["symbol"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["symbol"],
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    return dataset


def predict_tft(
    symbol: str,
    window_df: Optional[pd.DataFrame] = None,
) -> Optional[TFTResult]:
    """
    Attempt TFT inference for the given stock symbol.

    Resolution:
      1. Check for checkpoint at models/checkpoints/{symbol}.ckpt
      2. If no checkpoint: log attempt, return None (graceful — Fusion Layer handles)
      3. If checkpoint exists: load model, run inference, return TFTResult

    Args:
        symbol: Stock ticker (e.g. "BMRI.JK")
        window_df: Optional pre-built feature window DataFrame.
                   If None, loads from feature cache.

    Returns:
        TFTResult if forecast produced, None if no checkpoint or inference failed.
    """
    # Normalize symbol for filename (e.g. "BMRI.JK" -> "BMRI_JK")
    safe_symbol = symbol.upper().replace(".", "_")
    checkpoint_path = CHECKPOINT_DIR / f"{safe_symbol}.ckpt"

    # 1. Checkpoint existence check — the critical gate
    if not checkpoint_path.exists():
        _log_forecast_attempt(
            symbol,
            "UNAVAILABLE",
            f"no trained checkpoint found at {checkpoint_path}",
        )
        return None

    # 2. Checkpoint exists — attempt actual inference
    logger.info(f"Loading TFT checkpoint for {symbol} from {checkpoint_path}")
    try:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer

        # Load Lightning-native checkpoint (saved by trainer.save_checkpoint)
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(checkpoint_path),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        model.eval()

        # Load feature data if not provided
        if window_df is None:
            from src.stock.data.storage import BMRIStorage
            storage = BMRIStorage()
            window_df = storage.load("features_daily")
            if window_df is None:
                _log_forecast_attempt(symbol, "UNAVAILABLE", "no feature cache found")
                return None

        # Clean data: fill NaN/inf before inference (prevents index errors in encoders)
        import numpy as np
        window_df = window_df.ffill().bfill()
        window_df = window_df.replace([np.inf, -np.inf], 0.0)

        # Build inference dataset using STOCK schema (not BTC's)
        dataset = _build_inference_dataset(
            window_df,
            max_encoder_length=60,
            max_prediction_length=20,
        )

        # Run prediction
        with torch.no_grad():
            raw_pred = model.predict(
                dataset,
                mode="quantiles",
                trainer_kwargs={
                    "accelerator": "cpu",
                    "logger": False,
                    "enable_checkpointing": False,
                },
            )

        # raw_pred shape: (n_samples, max_prediction_length, n_quantiles=7)
        pred_array = raw_pred.numpy() if hasattr(raw_pred, "numpy") else np.array(raw_pred)
        last_pred = pred_array[-1]  # shape: (max_prediction_length, 7)

        quantile_labels = ["q10", "q20", "q30", "q40", "q50", "q60", "q90"]
        horizons = {}
        for step, label in [(5, "5d"), (10, "10d"), (20, "20d")]:
            idx = min(step, len(last_pred)) - 1
            row = last_pred[idx]
            horizons[label] = {
                "q10": float(row[0]),
                "q50": float(row[4]),
                "q90": float(row[6]),
            }

        overall = horizons.get("20d", {"q10": 0.0, "q50": 0.0, "q90": 0.0})

        result = TFTResult(
            symbol=symbol,
            quantiles=overall,
            horizons=horizons,
            model_version="tft_v1",
        )

        _log_forecast_attempt(
            symbol,
            "SUCCESS",
            f"direction={result.direction}, confidence={result.confidence:.2f}",
        )
        return result

    except Exception as e:
        _log_forecast_attempt(symbol, "ERROR", str(e))
        logger.error(f"TFT inference failed for {symbol}: {e}")
        return None
