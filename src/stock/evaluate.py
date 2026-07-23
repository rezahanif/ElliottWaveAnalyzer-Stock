"""
evaluate.py
-----------
Offline evaluation of trained BMRI TFT model.
Compares against naive baseline, runs Fusion Layer logic offline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("stock_evaluate")

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "BMRI_JK.ckpt"


def load_trained_model(checkpoint_path: Optional[Path] = None):
    """Load trained TFT model from Lightning-native checkpoint."""
    from pytorch_forecasting import TemporalFusionTransformer

    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint_path),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.eval()
    return model


def backtest_predictions(
    model,
    test_months: int = 3,
) -> pd.DataFrame:
    """
    Run backtest on held-out test period.
    Returns DataFrame with predictions vs actuals.
    """
    from src.stock.data.storage import BMRIStorage
    from src.stock.train import prepare_time_series_dataset, BMRIModelConfig

    storage = BMRIStorage()
    df = storage.load("features_daily")
    if df is None:
        raise ValueError("No feature cache found")

    config = BMRIModelConfig()
    _, df_prepared = prepare_time_series_dataset(df, config)

    # Time-based split: test = last N months
    test_cutoff = len(df_prepared) - test_months * 21
    test_df = df_prepared.iloc[test_cutoff:].copy()

    if len(test_df) == 0:
        raise ValueError("Test set is empty")

    # Build dataset on full data, then filter for test predictions
    from pytorch_forecasting import TimeSeriesDataSet
    from src.stock.features.schema import get_default_schema

    schema = get_default_schema()
    time_varying_unknown_reals = ["target_pct"]
    for feat in schema.numerical_features:
        if feat in test_df.columns:
            time_varying_unknown_reals.append(feat)

    time_varying_known_categoricals = []
    for feat, values in schema.categorical_features.items():
        if feat in test_df.columns:
            time_varying_known_categoricals.append(feat)

    # Build dataset with ALL data (train+test) so encoder can look back
    full_dataset = TimeSeriesDataSet(
        df_prepared,
        time_idx="time_idx",
        target="target_pct",
        group_ids=["symbol"],
        min_encoder_length=config.max_encoder_length // 2,
        max_encoder_length=config.max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=config.max_prediction_length,
        static_categoricals=["symbol"],
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Filter to test period — use predict with return_x=True to get both
    # predictions and actuals aligned, no need for dataset.filter
    with torch.no_grad():
        pred_output = model.predict(
            full_dataset,
            mode="quantiles",
            return_x=True,
            trainer_kwargs={
                "accelerator": "cpu",
                "logger": False,
                "enable_checkpointing": False,
            },
        )

    # pytorch-forecasting returns (predictions, x) or (predictions, x, index, y)
    if isinstance(pred_output, tuple):
        raw_pred = pred_output[0]
        x = pred_output[1]
    else:
        raw_pred = pred_output

    pred_array = raw_pred.numpy() if hasattr(raw_pred, "numpy") else np.array(raw_pred)
    # shape: (n_samples, max_prediction_length, n_quantiles=7)
    # q50 = index 4
    median_pred = pred_array[:, -1, 4]  # last step, q50

    # Get actuals from x — decoder_target is the actual target values
    actuals = x["decoder_target"][:, -1].numpy()  # last step of decoder target
    time_idx_vals = x["decoder_time_idx"][:, -1].numpy()

    assert len(median_pred) == len(actuals), \
        f"Length mismatch: pred={len(median_pred)}, actual={len(actuals)}"

    # Filter to test period only (time_idx >= test_cutoff)
    test_mask = time_idx_vals >= test_cutoff

    results = pd.DataFrame({
        "prediction": median_pred[test_mask],
        "actual": actuals[test_mask],
    })

    return results


def compute_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """Compute evaluation metrics."""
    pred = results["prediction"].values
    actual = results["actual"].values

    # MAE
    mae = np.mean(np.abs(pred - actual))

    # RMSE
    rmse = np.sqrt(np.mean((pred - actual) ** 2))

    # Directional accuracy
    pred_direction = np.sign(pred)
    actual_direction = np.sign(actual)
    directional_accuracy = np.mean(pred_direction == actual_direction)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_accuracy": float(directional_accuracy),
    }


def naive_baseline(results: pd.DataFrame) -> Dict[str, float]:
    """
    Compute naive baseline metrics (predict no change).
    Naive prediction = 0 (no price change).
    """
    actual = results["actual"].values
    naive_pred = np.zeros_like(actual)

    mae = np.mean(np.abs(naive_pred - actual))
    rmse = np.sqrt(np.mean((naive_pred - actual) ** 2))
    directional_accuracy = 0.5  # Random chance for direction

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_accuracy": float(directional_accuracy),
    }


def evaluate(checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Full evaluation pipeline.
    Returns comparison of TFT vs naive baseline.
    """
    logger.info("Loading trained model...")
    model = load_trained_model(checkpoint_path)

    logger.info("Running backtest...")
    results = backtest_predictions(model)

    logger.info("Computing metrics...")
    tft_metrics = compute_metrics(results)
    naive_metrics = naive_baseline(results)

    # Compare
    beats_baseline = tft_metrics["directional_accuracy"] > naive_metrics["directional_accuracy"]

    report = {
        "tft_metrics": tft_metrics,
        "naive_metrics": naive_metrics,
        "beats_baseline": beats_baseline,
        "test_samples": len(results),
    }

    logger.info(f"TFT Metrics: MAE={tft_metrics['mae']:.4f}, RMSE={tft_metrics['rmse']:.4f}, Dir Acc={tft_metrics['directional_accuracy']:.2%}")
    logger.info(f"Naive Metrics: MAE={naive_metrics['mae']:.4f}, RMSE={naive_metrics['rmse']:.4f}, Dir Acc={naive_metrics['directional_accuracy']:.2%}")
    logger.info(f"Beats baseline: {beats_baseline}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = evaluate()
    print(f"\n{'='*50}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*50}")
    print(f"  Test samples: {report['test_samples']}")
    print(f"  TFT  — MAE={report['tft_metrics']['mae']:.4f}, RMSE={report['tft_metrics']['rmse']:.4f}, Dir Acc={report['tft_metrics']['directional_accuracy']:.2%}")
    print(f"  Naive— MAE={report['naive_metrics']['mae']:.4f}, RMSE={report['naive_metrics']['rmse']:.4f}, Dir Acc={report['naive_metrics']['directional_accuracy']:.2%}")
    print(f"  Beats baseline: {report['beats_baseline']}")
    print(f"{'='*50}")
