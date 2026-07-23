"""
build_labeled_dataset.py
------------------------
Dataset Compiler — converts raw OHLCV + pivot data into the final tabular
CSV that TimeSeriesDataSet (pytorch-forecasting) expects for TFT training.

Leverages the core DatasetBuilder from src.waveconf.wave_model.dataset to
avoid code duplication and ensure absolute consistency between training
and serving/inference feature pipelines.

Usage:
    # From project root:
    python scripts/build_labeled_dataset.py --timeframe 1D
    python scripts/build_labeled_dataset.py --timeframe 1W
    python scripts/build_labeled_dataset.py --timeframe 1D 4H 1W  (all at once)

    # With custom paths:
    python scripts/build_labeled_dataset.py --timeframe 1D --layers-dir data/pivots --out-dir data/labeled
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.waveconf.wave_model.dataset import DatasetBuilder


def build_labeled_dataset(
    timeframe: str,
    layers_dir: str = "data/pivots",
    out_dir: str = "data/labeled",
) -> str:
    """
    Runs the full DatasetBuilder for the given timeframe and saves to CSV.
    """
    layers_path = os.path.join(layers_dir, f"BTC_{timeframe}_with_layers.json")

    if not os.path.exists(layers_path):
        raise FileNotFoundError(
            f"Enriched layer file not found: {layers_path}\n"
            f"Please run run_daily_analysis.py or ingestion scripts first to generate it."
        )

    print(f"\n{'='*60}")
    print(f"Building Labeled Dataset: BTC {timeframe}")
    print(f"{'='*60}")

    # Use the official tested DatasetBuilder to assemble features
    builder = DatasetBuilder(
        asset_timeframe=f"BTC_{timeframe}",
        astro_config_path="config/astro_features.yaml",
        calendar_config_path="config/economic_calender.yaml",
    )

    labeled = builder.build(layers_path)
    df_final = labeled.df

    # Ensure output directory exists and save CSV
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"BTC_{timeframe}_labeled.csv")
    df_final.to_csv(out_path, index=False)

    print(f"\n  ✅ Saved {len(df_final):,} rows × {len(df_final.columns)} columns")
    print(f"     → {out_path}")
    print(f"\n  Feature Columns Summary:")
    print(f"     Time Index   : {labeled.time_idx_column}")
    print(f"     Group ID     : {labeled.group_id_column}")
    print(f"     Known Future ({len(labeled.known_future_columns)}): {labeled.known_future_columns}")
    print(f"     Unknown Past ({len(labeled.unknown_past_columns)}): {labeled.unknown_past_columns}")
    print(f"     Target       : {labeled.target_column}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compile labeled TFT training dataset from enriched OHLCV pivot data."
    )
    parser.add_argument(
        '--timeframe', '-t',
        nargs='+',
        default=['1D'],
        choices=['1D', '4H', '1W'],
        help="Timeframe(s) to compile. Default: 1D"
    )
    parser.add_argument(
        '--layers-dir',
        default='data/pivots',
        help="Directory containing BTC_{TF}_with_layers.json files"
    )
    parser.add_argument(
        '--out-dir',
        default='data/labeled',
        help="Output directory for labeled CSV files"
    )
    args = parser.parse_args()

    # Change to project root so relative paths work
    os.chdir(ROOT)

    failed = []
    for tf in args.timeframe:
        try:
            build_labeled_dataset(
                timeframe=tf,
                layers_dir=args.layers_dir,
                out_dir=args.out_dir,
            )
        except Exception as e:
            print(f"\n  ❌ Failed for {tf}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(tf)

    if failed:
        print(f"\nFailed timeframes: {failed}")
        sys.exit(1)
    else:
        print(f"\n✅ All timeframes compiled successfully.")


if __name__ == "__main__":
    main()