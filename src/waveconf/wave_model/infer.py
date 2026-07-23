"""
infer.py
--------
Model Inference — loads the trained TFT model checkpoint and projects
quantile forecasts (q10, q50, q90) over forecast horizons (7, 14, 30, 60 days).
Converts the percentage change predictions back into absolute prices.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.waveconf.wave_model.model import prepare_df_for_tft


def predict_tft(
    window_df: pd.DataFrame,
    model_path: str = "models/wave_model.pt",
) -> Optional[Dict[str, float]]:
    """
    Run TFT model inference on prepared window_df.
    - Fills NaNs and pre-processes features.
    - Predicts quantile paths (q10, q50, q90) for the future 60 steps.
    - Compounds percentage changes starting from the last historical close price.
    - Returns absolute price dict for horizons 7, 14, 30, 60.
    """
    if not os.path.exists(model_path):
        print(f"  [infer] Model file not found at: {model_path}")
        return None

    try:
        from pytorch_forecasting import TemporalFusionTransformer

        # Load model using CPU to ensure low memory footprint and compatibility
        device = torch.device("cpu")
        model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=device, weights_only=False)
        model.eval()

        # Preprocess inference dataframe
        prep_df = prepare_df_for_tft(window_df)

        with torch.no_grad():
            # Run prediction to get raw quantiles
            # mode="quantiles" returns shape (samples, prediction_length, num_quantiles)
            predictions = model.predict(prep_df, mode="quantiles", trainer_kwargs={"accelerator": "cpu", "logger": False, "enable_checkpointing": False})

        # Extract predictions
        quantiles = predictions[0].numpy()  # shape: (60, 3) where quantiles are (q10, q50, q90)

        # Compounding prices starting from the last close price
        # Index 89 is the last historical row in the 150-row window (90 hist + 60 future)
        hist_rows = window_df[window_df["close"].notna()]
        if len(hist_rows) == 0:
            print("  [infer] No historical close price found in window")
            return None

        last_close = float(hist_rows.iloc[-1]["close"])

        # Accumulate log returns and estimate standard deviation to compile quantiles
        # under a log-normal random walk assumption, preventing exponential explosion.
        cumulative_mean = 0.0
        cumulative_var = 0.0

        q10_prices = [last_close]
        q50_prices = [last_close]
        q90_prices = [last_close]

        for step in range(min(60, len(quantiles))):
            y_q10 = float(quantiles[step, 0])
            y_q50 = float(quantiles[step, 1])
            y_q90 = float(quantiles[step, 2])

            # Convert to log returns. Force a minimum of -0.99 to avoid math domain errors.
            r_q10 = np.log(max(0.01, 1.0 + y_q10))
            r_q50 = np.log(max(0.01, 1.0 + y_q50))
            r_q90 = np.log(max(0.01, 1.0 + y_q90))

            # Mean is the median (q50) log return
            mean = r_q50
            # Volatility (standard deviation) estimation using normal distribution quantile factor 1.28155
            sd_90 = (r_q90 - r_q50) / 1.28155
            sd_10 = (r_q50 - r_q10) / 1.28155
            sd = max(0.0, (sd_90 + sd_10) / 2.0)

            cumulative_mean += mean
            cumulative_var += sd ** 2

            cum_sd = np.sqrt(cumulative_var)

            # Cumulative log return quantiles
            cum_r_q10 = cumulative_mean - 1.28155 * cum_sd
            cum_r_q50 = cumulative_mean
            cum_r_q90 = cumulative_mean + 1.28155 * cum_sd

            # Convert back to absolute prices
            q10_prices.append(last_close * np.exp(cum_r_q10))
            q50_prices.append(last_close * np.exp(cum_r_q50))
            q90_prices.append(last_close * np.exp(cum_r_q90))

        # Horizons are 7, 14, 30, 60
        result = {}
        for h in [7, 14, 30, 60]:
            if h < len(q50_prices):
                result[f"q10_{h}d"] = q10_prices[h]
                result[f"q50_{h}d"] = q50_prices[h]
                result[f"q90_{h}d"] = q90_prices[h]

        return result

    except Exception as e:
        print(f"  ❌ [infer] Prediction failed: {e}")
        import traceback

        traceback.print_exc()
        return None
