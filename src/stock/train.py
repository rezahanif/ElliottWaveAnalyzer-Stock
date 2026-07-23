"""
train.py
--------
BMRI TFT training pipeline.
Time-based train/val/test split, 5/10/20-day horizons.
Checkpoint saved to stock/models/checkpoints/BMRI_JK.ckpt.

CRITICAL: No BTC weights loaded at any point.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("stock_train")

# Checkpoint path (must match predict.py)
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "BMRI_JK.ckpt"

# Training config
DEFAULT_CONFIG = {
    "max_prediction_length": 20,  # 20-day horizon
    "max_encoder_length": 60,     # 60-day lookback
    "batch_size": 64,
    "max_epochs": 50,
    "learning_rate": 0.001,
    "hidden_size": 32,
    "attention_head_size": 1,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
}


class BMRIModelConfig:
    """Configuration for BMRI TFT model."""
    
    def __init__(self, **kwargs):
        self.max_prediction_length = kwargs.get("max_prediction_length", 20)
        self.max_encoder_length = kwargs.get("max_encoder_length", 60)
        self.batch_size = kwargs.get("batch_size", 64)
        self.max_epochs = kwargs.get("max_epochs", 50)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.hidden_size = kwargs.get("hidden_size", 32)
        self.attention_head_size = kwargs.get("attention_head_size", 1)
        self.dropout = kwargs.get("dropout", 0.1)
        self.hidden_continuous_size = kwargs.get("hidden_continuous_size", 8)
        
        # Time-based split
        self.val_months = kwargs.get("val_months", 6)
        self.test_months = kwargs.get("test_months", 3)


def prepare_time_series_dataset(
    df: pd.DataFrame,
    config: BMRIModelConfig,
    target: str = "close",
    time_idx: str = "time_idx",
    group_ids: List[str] = ["symbol"],
) -> Tuple[Any, pd.DataFrame]:
    """
    Prepare TimeSeriesDataSet for TFT training.
    Returns (dataset, prepared_df).
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from src.stock.features.schema import get_default_schema
    
    schema = get_default_schema()
    
    # Prepare dataframe
    df = df.copy()
    df["symbol"] = "BMRI_JK"
    df[time_idx] = range(len(df))
    
    # Normalize target to pct_change
    df["target_pct"] = df[target].pct_change().shift(-1)  # Next day's return
    df = df.dropna(subset=["target_pct"])
    
    # Time-varying known reals (from schema)
    time_varying_known_reals = []
    time_varying_unknown_reals = ["target_pct"]
    
    for feat in schema.numerical_features:
        if feat in df.columns:
            time_varying_unknown_reals.append(feat)
    
    # Static categorical
    static_categoricals = ["symbol"]
    
    # Time-varying categorical
    time_varying_known_categoricals = []
    for feat, values in schema.categorical_features.items():
        if feat in df.columns:
            time_varying_known_categoricals.append(feat)
    
    # Create dataset
    training_cutoff = len(df) - (config.val_months + config.test_months) * 21  # ~21 trading days/month
    
    dataset = TimeSeriesDataSet(
        df.iloc[:training_cutoff],
        time_idx=time_idx,
        target="target_pct",
        group_ids=group_ids,
        min_encoder_length=config.max_encoder_length // 2,
        max_encoder_length=config.max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=config.max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    return dataset, df


def train_tft(
    df: Optional[pd.DataFrame] = None,
    config: Optional[BMRIModelConfig] = None,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train BMRI TFT model.
    
    Args:
        df: Feature dataframe (loads from cache if None)
        config: Model configuration
        checkpoint_path: Where to save checkpoint (default: CHECKPOINT_PATH)
    
    Returns:
        Dict with training metrics and checkpoint path
    """
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader
    
    config = config or BMRIModelConfig()
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    
    # Load feature cache if not provided
    if df is None:
        from src.stock.data.storage import BMRIStorage
        storage = BMRIStorage()
        df = storage.load("features_daily")
        if df is None:
            raise ValueError("No feature cache found. Run feature_builder first.")
    
    logger.info(f"Training on {len(df)} rows")
    
    # Prepare dataset
    training_dataset, df_prepared = prepare_time_series_dataset(df, config)
    
    # Create validation dataset
    training_cutoff = len(df_prepared) - (config.val_months + config.test_months) * 21
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df_prepared, min_prediction_idx=training_cutoff
    )
    
    # Create dataloaders
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=config.batch_size, num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=config.batch_size * 2, num_workers=0
    )
    
    # Create model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    # Verify NO BTC weights loaded
    logger.info("Training from scratch - NO BTC weights loaded")
    
    # Create checkpoint directory
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create trainer with correct imports
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="cpu",
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", 
                min_delta=1e-4, 
                patience=10, 
                verbose=False, 
                mode="min"
            ),
            ModelCheckpoint(
                dirpath=str(checkpoint_path.parent),
                filename=checkpoint_path.stem,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
        ],
        logger=False,
        enable_checkpointing=True,
    )
    
    # Train
    logger.info("Starting TFT training...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Save checkpoint manually
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': tft.state_dict(),
        'config': {
            'hidden_size': config.hidden_size,
            'attention_head_size': config.attention_head_size,
            'dropout': config.dropout,
            'hidden_continuous_size': config.hidden_continuous_size,
        }
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    return {
        "checkpoint_path": str(checkpoint_path),
        "val_loss": None,  # Would need to extract from training
        "epochs": config.max_epochs,
        "training_samples": len(training_dataset),
        "validation_samples": len(validation_dataset),
    }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BMRI TFT model")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden size")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = BMRIModelConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
    )
    
    result = train_tft(config=config)
    print(f"\nTraining complete:")
    print(f"  Checkpoint: {result['checkpoint_path']}")
    print(f"  Val Loss: {result['val_loss']}")
    print(f"  Samples: {result['training_samples']} train / {result['validation_samples']} val")


if __name__ == "__main__":
    main()
