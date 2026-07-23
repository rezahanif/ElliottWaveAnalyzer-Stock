"""
train.py
--------
Model Trainer — compiles labeled datasets across all timeframes (1D, 4H, 1W),
partitions them into TimeSeriesDataSet structures, trains the TFT model using
PyTorch Lightning, and exports the final model checkpoint to models/wave_model.pt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.waveconf.wave_model.model import (
    prepare_df_for_tft,
    create_tft_dataset,
    build_tft_model,
    TARGET,
)


def train_tft(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    out_path: str = "models/wave_model.pt",
    hidden_size: int = 32,
    attention_head_size: int = 2,
    accelerator: str = "auto",
):
    print("\n" + "=" * 60)
    print("Initializing TFT Training")
    print("=" * 60)

    # 1. Load labeled CSV files
    data_dir = os.path.join(ROOT, "data", "labeled")
    timeframes = ["1D", "4H", "1W"]
    dfs = []

    for tf in timeframes:
        path = os.path.join(data_dir, f"BTC_{tf}_labeled.csv")
        if not os.path.exists(path):
            print(f"⚠️ Warning: Labeled file not found for {tf}: {path}. Skipping.")
            continue
        print(f"  [loader] Loading {tf} dataset: {path}")
        df_tf = pd.read_csv(path)
        dfs.append(df_tf)

    if not dfs:
        raise FileNotFoundError(
            f"No labeled CSV files found in {data_dir}. "
            "Please run scripts/build_labeled_dataset.py first."
        )

    # Combine datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  [loader] Total combined rows: {len(combined_df):,}")

    # Drop rows where target is NaN (the last row of each timeframe series) before preparing features
    combined_df = combined_df.dropna(subset=[TARGET]).reset_index(drop=True)

    # Prepare features
    combined_df = prepare_df_for_tft(combined_df)

    # 2. Build TimeSeriesDataSet
    max_encoder_length = 90
    max_prediction_length = 60

    print("  [dataset] Creating PyTorch Forecasting TimeSeriesDataSet...")
    dataset = create_tft_dataset(
        combined_df,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )

    # Create validation and training sets by partitioning per timeframe group
    train_dfs = []
    for tf, df_tf in combined_df.groupby("asset_timeframe"):
        cutoff = df_tf["time_idx"].max() - max_prediction_length
        train_dfs.append(df_tf[df_tf["time_idx"] <= cutoff])

    train_subset = pd.concat(train_dfs, ignore_index=True)

    # Create subset dataset configs
    training_data = TimeSeriesDataSet.from_dataset(dataset, train_subset)
    validation_data = TimeSeriesDataSet.from_dataset(dataset, combined_df, predict=True, stop_randomization=True)

    # Convert to dataloaders
    train_dataloader = training_data.to_dataloader(batch_size=batch_size, train=True, num_workers=0)
    val_dataloader = validation_data.to_dataloader(batch_size=batch_size * 10, train=False, num_workers=0)

    print(f"  [dataset] Train batches: {len(train_dataloader)} | Val batches: {len(val_dataloader)}")

    # 3. Create Model
    print("  [model] Building Temporal Fusion Transformer model...")
    tft = build_tft_model(
        dataset,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=0.1,
    )
    print(f"  [model] Parameters: {tft.size():,}")

    # 4. Configure Trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=3, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="src/waveconf/wave_model/checkpoints",
        filename="tft_checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Configure accelerator and devices
    if accelerator == "auto":
        if torch.cuda.is_available():
            accel, devices = "gpu", 1
            print("  [trainer] GPU (CUDA) available. Running on GPU.")
        else:
            accel, devices = "cpu", "auto"
            print("  [trainer] Defaulting to CPU. (Apple MPS has known op-coverage/failed assertions in TFT on macOS and is disabled by default).")
    else:
        accel = accelerator
        devices = 1 if accelerator in ["gpu", "mps"] else "auto"
        print(f"  [trainer] Using user-specified accelerator: {accel}")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accel,
        devices=devices,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
    )

    # 5. Fit Model
    print("  [trainer] Starting training...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Save final model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, out_path)
        print(f"  ✅ Saved best checkpoint to: {out_path}")
    else:
        # Fallback to saving final state
        trainer.save_checkpoint(out_path)
        print(f"  ✅ Saved final training checkpoint to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Temporal Fusion Transformer model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out", default="models/wave_model.pt", help="Path to save the model weights")
    parser.add_argument("--hidden-size", type=int, default=32, help="TFT hidden state size")
    parser.add_argument("--heads", type=int, default=2, help="Attention heads")
    parser.add_argument("--accelerator", default="auto", help="Accelerator (cpu, gpu, mps, auto)")

    args = parser.parse_args()

    # Move to project root
    os.chdir(ROOT)

    train_tft(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
        hidden_size=args.hidden_size,
        attention_head_size=args.heads,
        accelerator=args.accelerator,
    )


if __name__ == "__main__":
    main()
