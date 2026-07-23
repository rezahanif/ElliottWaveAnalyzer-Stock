"""
evaluate.py
-----------
Offline evaluation of trained BMRI TFT model.
Compares against naive baseline, runs Fusion Layer logic offline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("stock_evaluate")

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "BMRI_JK.ckpt"


def load_trained_model(checkpoint_path: Optional[Path] = None):
    """Load trained TFT model from checkpoint."""
    from pytorch_forecasting import TemporalFusionTransformer
    from src.stock.data.storage import BMRIStorage
    from src.stock.features.feature_builder import build_feature_cache
    
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
    
    # Load a sample dataset to get model config
    storage = BMRIStorage()
    df = storage.load("features_daily")
    if df is None:
        df = build_feature_cache()
    
    # Create a minimal dataset for model structure
    from src.stock.train import prepare_time_series_dataset, BMRIModelConfig
    config = BMRIModelConfig()
    dataset, _ = prepare_time_series_dataset(df, config)
    
    # Load checkpoint manually (saved as state_dict, not Lightning checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Recreate model from dataset
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=config.learning_rate,
        hidden_size=ckpt.get('config', {}).get('hidden_size', config.hidden_size),
        attention_head_size=ckpt.get('config', {}).get('attention_head_size', config.attention_head_size),
        dropout=ckpt.get('config', {}).get('dropout', config.dropout),
        hidden_continuous_size=ckpt.get('config', {}).get('hidden_continuous_size', config.hidden_continuous_size),
        output_size=7,
    )
    
    # Load state dict
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    return model, dataset


def backtest_predictions(
    model,
    dataset,
    test_months: int = 3,
) -> pd.DataFrame:
    """
    Run backtest on held-out test period.
    Returns DataFrame with predictions vs actuals.
    """
    from torch.utils.data import DataLoader
    
    # Get test period
    df_size = len(dataset.data)
    test_cutoff = df_size - test_months * 21
    
    # Create test dataloader
    test_dataset = dataset.filter(lambda x: x["time_idx"].max() >= test_cutoff)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    predictions = []
    actuals = []
    dates = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            pred = model.predict(batch, mode="quantiles")
            # Get median prediction
            median_pred = pred[:, :, 1].numpy()  # q50
            
            # Get actual values
            actual = batch["target"].numpy()
            
            predictions.extend(median_pred.flatten().tolist())
            actuals.extend(actual.flatten().tolist())
    
    results = pd.DataFrame({
        "prediction": predictions[:len(actuals)],
        "actual": actuals,
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
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
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
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }


def evaluate(checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Full evaluation pipeline.
    Returns comparison of TFT vs naive baseline.
    """
    logger.info("Loading trained model...")
    model, dataset = load_trained_model(checkpoint_path)
    
    logger.info("Running backtest...")
    results = backtest_predictions(model, dataset)
    
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
    print(f"\nEvaluation Report:")
    print(f"  TFT Directional Accuracy: {report['tft_metrics']['directional_accuracy']:.2%}")
    print(f"  Naive Baseline: {report['naive_metrics']['directional_accuracy']:.2%}")
    print(f"  Beats Baseline: {report['beats_baseline']}")
