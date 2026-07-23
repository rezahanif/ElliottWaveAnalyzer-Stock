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
import os
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
                   If None, caller is expected to have already prepared features.

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
        # Import here so missing pytorch_forecasting doesn't crash the module
        # on systems that only run the deterministic path.
        import torch
        from pytorch_forecasting import TemporalFusionTransformer

        device = torch.device("cpu")
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
        model.eval()

        if window_df is None:
            _log_forecast_attempt(symbol, "UNAVAILABLE", "no window_df provided for inference")
            return None

        # Prepare features using the same schema as BTC TFT
        from src.btc.wave_model.model import prepare_df_for_tft
        prep_df = prepare_df_for_tft(window_df)

        with torch.no_grad():
            predictions = model.predict(
                prep_df,
                mode="quantiles",
                trainer_kwargs={
                    "accelerator": "cpu",
                    "logger": False,
                    "enable_checkpointing": False,
                },
            )

        quantiles_raw = predictions[0].numpy()  # shape: (60, 3) → (q10, q50, q90)

        # Extract per-horizon quantiles (7, 14, 30, 60 days)
        horizons = {}
        for step, label in [(7, "7d"), (14, "14d"), (30, "30d"), (60, "60d")]:
            idx = min(step, len(quantiles_raw)) - 1
            horizons[label] = {
                "q10": float(quantiles_raw[idx, 0]),
                "q50": float(quantiles_raw[idx, 1]),
                "q90": float(quantiles_raw[idx, 2]),
            }

        # Overall quantiles (use 30d as representative)
        overall = horizons.get("30d", {"q10": 0.0, "q50": 0.0, "q90": 0.0})

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
