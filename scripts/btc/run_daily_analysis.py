"""
run_daily_analysis.py
---------------------
Ingestion Orchestrator — runs the full live inference pipeline.

Called by APScheduler every 6 and 12 hours (Section 9, v2.0 spec).
Can also be run manually from the CLI for ad-hoc analysis.

Pipeline:
    1. Fetch latest candles from Binance (incremental — only new bars)
    2. Append to existing OHLCV history (no re-fetch of full history)
    3. Re-calculate volatility layers (calculate_layers logic)
    4. Run ZigZag pivot detection on updated data
    5. Compute all indicators + context columns via DatasetBuilder
    6. Run TFT inference (if model weights exist)
    7. Run FibonacciEngine for cluster targets
    8. Run ConfluenceChecker
    9. Write prediction record to SQLite
   10. Send Telegram alert if confluence_valid=1 or signal changed

Usage:
    # Full daily run (1D + 4H timing check):
    python scripts/run_daily_analysis.py

    # Specific timeframe:
    python scripts/run_daily_analysis.py --timeframe 1D
    python scripts/run_daily_analysis.py --timeframe 4H   # invalidation check only

    # Dry run (no Telegram, no SQLite write):
    python scripts/run_daily_analysis.py --dry-run

Environment variables:
    TELEGRAM_BOT_TOKEN   — Telegram bot token
    TELEGRAM_CHAT_ID     — Target chat ID for alerts
    MODEL_PATH           — Path to trained .pt weights (default: models/wave_model.pt)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env file if it exists to populate environment variables
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")


from src.btc.pivots.zigzag import ZigZagDetector
from src.btc.pivots.pivot_schema import SwingType
from src.btc.fib_engine.fibonacci import FibonacciEngine

# Optional heavy imports — guarded to avoid hard failures at import time
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    warnings.warn("ccxt not installed — live fetch unavailable. Run: pip install ccxt")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

TIMEFRAME_META = {
    '1D': {'ccxt_tf': '1d',  'ms_per_bar': 86_400_000,      'min_bars': 3},
    '4H': {'ccxt_tf': '4h',  'ms_per_bar': 14_400_000,      'min_bars': 6},
    '1W': {'ccxt_tf': '1w',  'ms_per_bar': 7 * 86_400_000,  'min_bars': 2},
}

DEFAULT_MODEL_PATH = "models/wave_model.pt"
SQLITE_PATH        = "data/predictions.db"
OHLCV_DIR          = "data/ohlcv"
LAYERS_DIR         = "data/pivots"


# ─────────────────────────────────────────────────────────────
# Real-time spot price (used across all timeframes)
# ─────────────────────────────────────────────────────────────

def fetch_spot_price(fallback_price: float = None) -> float:
    """
    Fetch the current BTC/USD spot price from a live ticker.
    Fallback chain: Kraken → Yahoo Finance → provided fallback.
    """
    # Try Kraken ticker
    if CCXT_AVAILABLE:
        try:
            import ccxt
            kraken = ccxt.kraken({'enableRateLimit': True, 'verify': False})
            ticker = kraken.fetch_ticker('BTC/USD')
            price = float(ticker['last'])
            print(f"  [spot] Live price from Kraken: ${price:,.2f}")
            return price
        except Exception as e:
            print(f"  [spot] Kraken ticker failed: {e}")

    # Try Yahoo Finance
    try:
        import yfinance as yf
        ticker = yf.Ticker("BTC-USD")
        info = ticker.fast_info
        price = float(info['lastPrice'])
        print(f"  [spot] Live price from Yahoo Finance: ${price:,.2f}")
        return price
    except Exception as e:
        print(f"  [spot] Yahoo Finance ticker failed: {e}")

    # Fallback to last candle close
    if fallback_price is not None:
        print(f"  [spot] Using last candle close as fallback: ${fallback_price:,.2f}")
        return fallback_price

    raise RuntimeError("Could not fetch live BTC price from any source")


# ─────────────────────────────────────────────────────────────
# Step 1 & 2: Incremental candle fetch + append
# ─────────────────────────────────────────────────────────────

def fetch_incremental(timeframe: str, full_refetch: bool = False) -> pd.DataFrame:
    """
    Fetch only new candles since the last stored bar.
    Appends to existing OHLCV JSON — no full re-download.

    Returns the full updated DataFrame.
    """
    if not CCXT_AVAILABLE:
        raise RuntimeError("ccxt not installed. Run: pip install ccxt")

    meta     = TIMEFRAME_META[timeframe]
    ohlcv_path = os.path.join(OHLCV_DIR, f"BTC_{timeframe}.json")
    columns    = ["timestamp_ms", "open", "high", "low", "close", "volume"]

    # Load existing history
    existing_df = pd.DataFrame(columns=columns)
    if os.path.exists(ohlcv_path) and not full_refetch:
        with open(ohlcv_path) as f:
            raw = json.load(f)
        existing_df = pd.DataFrame(raw['data'], columns=raw['columns'])
        last_ts = int(existing_df['timestamp_ms'].max())
        since   = last_ts + meta['ms_per_bar']
        print(f"  [fetch] {timeframe}: existing {len(existing_df):,} bars, "
              f"fetching from {datetime.utcfromtimestamp(since/1000).strftime('%Y-%m-%d %H:%M')} UTC")
    else:
        # First-time fetch or full refetch — start from 2017-08-17
        import ccxt
        exchange = ccxt.kraken({'enableRateLimit': True, 'verify': False})
        since = exchange.parse8601('2017-08-17T00:00:00Z')
        print(f"  [fetch] {timeframe}: full history fetch starting from 2017-08-17...")

    new_candles = []

    # Step A: Try Kraken (Primary source — completely unblocked and has all timeframes)
    try:
        import ccxt
        kraken = ccxt.kraken({'enableRateLimit': True, 'verify': False})
        print("  [fetch] Fetching from Kraken...")
        now_ms = kraken.milliseconds()
        since_temp = since
        while since_temp < now_ms:
            batch = kraken.fetch_ohlcv(
                'BTC/USD',
                timeframe = meta['ccxt_tf'],
                since     = since_temp,
                limit     = 1000,
            )
            if not batch:
                break
            new_candles.extend(batch)
            since_temp = batch[-1][0] + meta['ms_per_bar']
            if len(batch) < 1000:
                break
        if new_candles:
            print(f"  [fetch] Successfully fetched {len(new_candles)} candles from Kraken.")
    except Exception as e_kraken:
        print(f"  [fetch] Kraken primary fetch failed: {e_kraken}")
        new_candles = []

    # Step B: Fallback 1 — Try Yahoo Finance (only for 1D/1W)
    if not new_candles and timeframe in ('1D', '1W'):
        try:
            import yfinance as yf
            print("  [fetch] Trying Yahoo Finance fallback...")
            ticker = yf.Ticker("BTC-USD")
            start_date = datetime.utcfromtimestamp(since / 1000).strftime('%Y-%m-%d')
            yf_tf = '1d' if timeframe == '1D' else '1wk'
            yf_df = ticker.history(start=start_date, interval=yf_tf)
            for idx, row in yf_df.iterrows():
                ts_ms = int(idx.timestamp() * 1000)
                new_candles.append([
                    ts_ms,
                    round(float(row['Open']), 2),
                    round(float(row['High']), 2),
                    round(float(row['Low']), 2),
                    round(float(row['Close']), 2),
                    round(float(row['Volume']), 2)
                ])
            if new_candles:
                print(f"  [fetch] Successfully fetched {len(new_candles)} candles from Yahoo Finance fallback.")
        except Exception as e_yf:
            print(f"  [fetch] Yahoo Finance fallback failed: {e_yf}")
            new_candles = []

    # Step C: Fallback 2 — Try Binance (final fallback — might require VPN)
    if not new_candles:
        try:
            import ccxt
            binance = ccxt.binance({'enableRateLimit': True, 'verify': False})
            print("  [fetch] Trying Binance fallback (VPN might be required)...")
            now_ms = binance.milliseconds()
            since_temp = since
            while since_temp < now_ms:
                batch = binance.fetch_ohlcv(
                    'BTC/USDT',
                    timeframe = meta['ccxt_tf'],
                    since     = since_temp,
                    limit     = 1000,
                )
                if not batch:
                    break
                new_candles.extend(batch)
                since_temp = batch[-1][0] + meta['ms_per_bar']
                if len(batch) < 1000:
                    break
            if new_candles:
                print(f"  [fetch] Successfully fetched {len(new_candles)} candles from Binance fallback.")
        except Exception as e_binance:
            print(f"  [fetch] Binance fallback failed: {e_binance}")
            new_candles = []


    if not new_candles:
        print(f"  [fetch] No new candles for {timeframe} — already up to date or all sources failed")
        return existing_df



    new_df = pd.DataFrame(new_candles, columns=columns)
    print(f"  [fetch] {len(new_df)} new candles fetched")

    # Merge and deduplicate
    combined = (
        pd.concat([existing_df, new_df], ignore_index=True)
        .drop_duplicates(subset=['timestamp_ms'], keep='last')
        .sort_values('timestamp_ms')
        .reset_index(drop=True)
    )

    # Save updated history
    os.makedirs(OHLCV_DIR, exist_ok=True)
    out = {
        "asset":     "BTCUSD",
        "timeframe": timeframe,
        "columns":   columns,
        "data":      combined.values.tolist(),
    }
    with open(ohlcv_path, 'w') as f:
        json.dump(out, f)

    print(f"  [fetch] Total bars now: {len(combined):,} → saved to {ohlcv_path}")
    return combined


# ─────────────────────────────────────────────────────────────
# Step 3: Re-calculate volatility layers
# ─────────────────────────────────────────────────────────────

def recalculate_layers(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Re-run the dual ATR threshold calculation on updated OHLCV data.
    Saves enriched JSON and returns the enriched DataFrame.
    """
    cpd = {'1W': 1/7, '1D': 1, '4H': 6, '1H': 24}.get(timeframe, 1)

    atr_20_p = max(1, int(20 * cpd))
    atr_14_p = max(1, int(14 * cpd))
    date_fmt  = '%Y-%m-%d %H:%M' if timeframe == '4H' else '%Y-%m-%d'

    df = df.sort_values('timestamp_ms').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.strftime(date_fmt)

    df['prev_close'] = df['close'].shift(1)
    df['h_l']  = df['high'] - df['low']
    df['h_pc'] = (df['high'] - df['prev_close']).abs()
    df['l_pc'] = (df['low']  - df['prev_close']).abs()
    df['true_range'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)

    df['atr_20'] = df['true_range'].ewm(alpha=1/atr_20_p, adjust=False).mean()
    df['wall_street_threshold_pct'] = round((df['atr_20'] * 3 / df['close']) * 100, 2)

    df['atr_14'] = df['true_range'].ewm(alpha=1/atr_14_p, adjust=False).mean()
    df['behavioral_threshold_pct']  = round((df['atr_14'] * 1.5 / df['close']) * 100, 2)

    keep = ['timestamp_ms', 'date', 'open', 'high', 'low', 'close', 'volume',
            'wall_street_threshold_pct', 'behavioral_threshold_pct']
    df_clean = df[keep].dropna()

    os.makedirs(LAYERS_DIR, exist_ok=True)
    layers_path = os.path.join(LAYERS_DIR, f"BTC_{timeframe}_with_layers.json")
    out = {
        "asset":      "BTCUSD",
        "timeframe":  timeframe,
        "columns":    list(df_clean.columns),
        "data":       df_clean.values.tolist(),
    }
    with open(layers_path, 'w') as f:
        json.dump(out, f)

    print(f"  [layers] Recalculated {len(df_clean):,} rows → {layers_path}")
    return df_clean


# ─────────────────────────────────────────────────────────────
# Step 4: Latest pivot detection
# ─────────────────────────────────────────────────────────────

def detect_latest_pivots(df: pd.DataFrame, timeframe: str):
    """
    Run ZigZag on the enriched DataFrame.
    Returns (macro_pivots, micro_pivots) — full list, sorted ascending.
    """
    detector = ZigZagDetector(timeframe=timeframe)
    result   = detector.run(df, asset="BTCUSD")
    print(f"  [zigzag] {timeframe}: {len(result.macro)} macro | {len(result.micro)} micro pivots")
    return result.macro, result.micro


# ─────────────────────────────────────────────────────────────
# Step 5: Feature preparation for inference
# ─────────────────────────────────────────────────────────────

def prep_inference_window(
    df: pd.DataFrame,
    macro_pivots,  # kept for signature compatibility
    timeframe: str,
    lookback: int = 90,
    horizon: int = 60,
) -> pd.DataFrame:
    """
    Prepare the input DataFrame for TFT inference.
    Runs the official DatasetBuilder on history, then appends future rows
    populated with future timestamps and known-future features (astro + calendar).
    """
    from src.btc.wave_model.dataset import DatasetBuilder

    # 1. Build features for all historical candles
    builder = DatasetBuilder(
        asset_timeframe=f"BTC_{timeframe}",
        astro_config_path="config/astro_features.yaml",
        calendar_config_path="config/economic_calender.yaml",
    )
    layers_path = os.path.join(LAYERS_DIR, f"BTC_{timeframe}_with_layers.json")
    labeled = builder.build(layers_path)
    hist_df = labeled.df

    # We only need the last `lookback` rows of history
    hist_window = hist_df.tail(lookback).copy()

    # 2. Generate future rows
    meta = TIMEFRAME_META[timeframe]
    last_row = hist_window.iloc[-1]
    last_ts = int(last_row['timestamp_ms'])
    last_time_idx = int(last_row['time_idx'])

    future_rows = []
    for step in range(1, horizon + 1):
        future_ts = last_ts + step * meta['ms_per_bar']
        future_rows.append({
            'timestamp_ms': future_ts,
            'time_idx': last_time_idx + step,
            'asset_timeframe': f"BTC_{timeframe}",
        })

    future_df = pd.DataFrame(future_rows)

    # 3. Compute known-future features for the future rows
    future_df = builder._attach_known_future(future_df)

    # Fill unknown past columns with NaN
    for col in labeled.unknown_past_columns:
        future_df[col] = np.nan
    future_df['close_pct_change'] = np.nan

    # 4. Concatenate history and future
    cols = list(hist_window.columns)
    for col in cols:
        if col not in future_df.columns:
            future_df[col] = np.nan
    future_df = future_df[cols]

    inference_df = pd.concat([hist_window, future_df], ignore_index=True)
    return inference_df


# ─────────────────────────────────────────────────────────────
# Step 6: TFT inference
# ─────────────────────────────────────────────────────────────

def run_tft_inference(
    window_df:   pd.DataFrame,
    model_path:  str = DEFAULT_MODEL_PATH,
) -> Optional[Dict]:
    """
    Load frozen TFT weights and run inference on the prepared window.
    Returns quantile forecast dict {q10, q50, q90} per horizon, or None if no model.
    """
    if not TORCH_AVAILABLE:
        print("  [tft] torch not available — skipping inference")
        return None

    if not os.path.exists(model_path):
        print(f"  [tft] No model weights at {model_path} — skipping inference")
        print(f"        Train the model first: python train.py, then copy .pt here")
        return None

    try:
        from src.btc.wave_model.infer import predict_tft
        result = predict_tft(window_df, model_path)
        if result:
            print(f"  [tft] Inference complete. q50_30d=${result.get('q50_30d', 0):,.0f}")
        return result

    except Exception as e:
        print(f"  [tft] Inference failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Step 7: Fibonacci cluster targets
# ─────────────────────────────────────────────────────────────

def compute_fib_targets(macro_pivots, version: str = "v5_relaxed") -> Optional[Dict]:
    """
    Auto-detect directional context from the last confirmed macro pivot,
    then compute the dual Fibonacci cluster.

    Bearish (last pivot is HIGH):
        c_top = last HIGH | b_low = most recent LOW before it
        Projects 2.618 / 1.618 extensions BELOW c_top.

    Bullish (last pivot is LOW):
        c_bottom = last LOW | b_high = most recent HIGH before it
        Projects 2.618 / 1.618 extensions ABOVE c_bottom.
        (c_top / b_low variable names kept for FibonacciEngine API compat)
    """
    if len(macro_pivots) < 4:
        print("  [fib] Not enough macro pivots for cluster computation")
        return None

    highs = [p for p in macro_pivots if p.is_high()]
    lows  = [p for p in macro_pivots if p.is_low()]

    if not highs or not lows:
        return None

    last_pivot = macro_pivots[-1]
    engine = FibonacciEngine()

    if last_pivot.is_high():
        # BEARISH: last pivot is a HIGH - project down
        direction   = "bearish"
        c_top_pivot = last_pivot
        b_low_pivot = max(
            (p for p in lows if p.bar_index < c_top_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        if b_low_pivot is None:
            print("  [fib] No B low found before C top")
            return None
        
        a_high_pivot = max(
            (p for p in highs if p.bar_index < b_low_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        ab_range = abs(a_high_pivot.price - b_low_pivot.price) if a_high_pivot else None
        a_price = a_high_pivot.price if a_high_pivot else None
        c_top_val = c_top_pivot.price
        b_low_val = b_low_pivot.price

        invalidation = round(c_top_pivot.price * (1 + engine.invalidation_buffer), 2)
        c_date = pd.to_datetime(c_top_pivot.timestamp_ms, unit='ms').date()
        b_date = pd.to_datetime(b_low_pivot.timestamp_ms, unit='ms').date()
        print(f"  [fib] Direction: BEARISH (version: {version})")
        print(f"  [fib] C top:    ${c_top_pivot.price:>12,.2f}  ({c_date})")
        print(f"  [fib] B low:    ${b_low_pivot.price:>12,.2f}  ({b_date})")
        if a_high_pivot:
            a_date = pd.to_datetime(a_high_pivot.timestamp_ms, unit='ms').date()
            print(f"  [fib] A high:   ${a_high_pivot.price:>12,.2f} ({a_date})")
        print(f"  [fib] Invalidated above: ${invalidation:,.2f}")
    else:
        # BULLISH: last pivot is a LOW - project up
        direction   = "bullish"
        c_top_pivot = last_pivot   # c_bottom in semantic terms
        b_low_pivot = max(         # b_high in semantic terms
            (p for p in highs if p.bar_index < c_top_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        if b_low_pivot is None:
            print("  [fib] No B high found before C bottom")
            return None
            
        a_low_pivot = max(
            (p for p in lows if p.bar_index < b_low_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        ab_range = abs(a_low_pivot.price - b_low_pivot.price) if a_low_pivot else None
        a_price = a_low_pivot.price if a_low_pivot else None
        c_top_val = b_low_pivot.price
        b_low_val = c_top_pivot.price

        invalidation = round(c_top_pivot.price * (1 - engine.invalidation_buffer), 2)
        c_date = pd.to_datetime(c_top_pivot.timestamp_ms, unit='ms').date()
        b_date = pd.to_datetime(b_low_pivot.timestamp_ms, unit='ms').date()
        print(f"  [fib] Direction: BULLISH (version: {version})")
        print(f"  [fib] C bottom: ${c_top_pivot.price:>12,.2f}  ({c_date})")
        print(f"  [fib] B high:   ${b_low_pivot.price:>12,.2f}  ({b_date})")
        if a_low_pivot:
            a_date = pd.to_datetime(a_low_pivot.timestamp_ms, unit='ms').date()
            print(f"  [fib] A low:    ${a_low_pivot.price:>12,.2f} ({a_date})")
        print(f"  [fib] Invalidated below: ${invalidation:,.2f}")

    cluster = engine.dual_cluster(
        c_top     = c_top_val,
        b_low     = b_low_val,
        direction = direction,
        ab_range  = ab_range,
        a_price   = a_price,
        version   = version,
    )
    print(f"  [fib] {cluster}")

    return {
        "direction":        direction,
        "c_top":            c_top_pivot.price,
        "b_low":            b_low_pivot.price,
        "target_a":         cluster.target_a.price,
        "target_b":         cluster.target_b.price,
        "cluster_valid":    cluster.cluster_valid,
        "cluster_strength": cluster.cluster_strength,
        "cluster_upper":    cluster.cluster_upper,
        "cluster_lower":    cluster.cluster_lower,
        "scenario_a_price": cluster.scenario_a.price,
        "scenario_b_price": cluster.scenario_b.price,
        "invalidation":     invalidation,
    }



# ─────────────────────────────────────────────────────────────
# Step 8: Calendar risk adjustment
# ─────────────────────────────────────────────────────────────

def apply_calendar_risk(strength: float, window_df: pd.DataFrame, lookback: int = 90) -> Tuple[float, str]:
    """
    Penalise cluster confidence if a high-impact event is imminent.
    Returns (adjusted_strength, risk_flag_text).
    """
    if 'days_to_fomc' not in window_df.columns:
        return strength, ""

    # Current bar is the last bar of history (index lookback - 1)
    last_hist = window_df.iloc[lookback - 1]
    post_event = last_hist.get('post_event_window', 0)
    within_2d = last_hist.get('high_impact_within_2d', 0)
    within_5d = last_hist.get('high_impact_within_5d', 0)

    risk_flag = ""

    if post_event == 1:
        strength  = min(1.0, strength * 1.10)
        risk_flag = "Post-FOMC (+10% boost)"
    elif within_2d == 1:
        strength  = strength * 0.60
        risk_flag = "High-impact event in 2 days or less — HIGH RISK (×0.60)"
    elif within_5d == 1:
        strength  = strength * 0.80
        risk_flag = "High-impact event in 5 days or less — moderate risk (×0.80)"


    return round(strength, 4), risk_flag


# ─────────────────────────────────────────────────────────────
# Step 9: SQLite write
# ─────────────────────────────────────────────────────────────

def init_db(db_path: str):
    """Create predictions table if not exists and run migrations if needed."""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp             DATETIME DEFAULT CURRENT_TIMESTAMP,
            timeframe             TEXT,
            direction             TEXT,
            btc_close_at_signal   REAL,
            cluster_valid         INTEGER,
            cluster_upper         REAL,
            cluster_lower         REAL,
            cluster_strength      REAL,
            cluster_strength_adj  REAL,
            target_a              REAL,
            target_b              REAL,
            scenario_a_price      REAL,
            scenario_b_price      REAL,
            invalidation_level    REAL,
            c_top                 REAL,
            b_low                 REAL,
            q10_7d REAL, q50_7d REAL, q90_7d REAL,
            q10_14d REAL, q50_14d REAL, q90_14d REAL,
            q10_30d REAL, q50_30d REAL, q90_30d REAL,
            q10_60d REAL, q50_60d REAL, q90_60d REAL,
            calendar_risk_flag    TEXT,
            macro_pivot_count     INTEGER,
            micro_pivot_count     INTEGER,
            actual_outcome        TEXT,
            prediction_correct    INTEGER
        )
    """)

    # Run migration to add direction if it's missing in existing database
    cursor = conn.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cursor.fetchall()]
    if "direction" not in columns:
        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN direction TEXT")
            print("  [db] Migration: Added 'direction' column to predictions table.")
        except Exception as e:
            print(f"  [db] Migration failed: {e}")

    conn.commit()
    conn.close()



def write_prediction(db_path: str, record: Dict):
    """Insert one prediction record into SQLite."""
    conn   = sqlite3.connect(db_path)
    cols   = ', '.join(record.keys())
    placeholders = ', '.join(['?'] * len(record))
    conn.execute(
        f"INSERT INTO predictions ({cols}) VALUES ({placeholders})",
        list(record.values())
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Step 10: Telegram alert
# ─────────────────────────────────────────────────────────────

from src.shared.telegram.client import send_telegram



def get_economic_calendar_section(timeframe: str) -> Tuple[str, str]:
    """Fetch next 24h events and format the Economic Risk Calendar and risk warnings."""
    # ponytail: Return empty strings to completely omit the 24h calendar section
    # and elevated macro risk warnings from the main 1D/4H Elliott Wave forecasts.
    # The dedicated daily/weekly reminder and pre-event alert services handle this.
    return "", ""



def build_telegram_message(
    timeframe:    str,
    fib_result:   Optional[Dict],
    tft_result:   Optional[Dict],
    adj_strength: float,
    risk_flag:    str,
    current_price: float,
    version:      str = "v5_relaxed",
) -> str:
    """Format the Telegram alert per Section 11 spec."""
    direction = fib_result.get('direction', 'bearish') if fib_result else 'bearish'
    is_bullish = direction == 'bullish'
    direction_emoji = "📈" if is_bullish else "📉"
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    valid = fib_result and fib_result.get("cluster_valid")

    lines = [
        f"{direction_emoji} <b>BTC ELLIOTT WAVE FORECAST [{timeframe}] ({version})</b>",
        f"<code>{now}</code>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"💰 Current price: <b>${current_price:,.2f}</b>",
        f"📐 Bias: <b>{'BULLISH — projecting UP from confirmed low' if is_bullish else 'BEARISH — projecting DOWN from confirmed high'}</b>",
        "",
    ]

    if valid:
        lines += [
            f"✅ <b>CONFLUENCE {'CONFIRMED' if adj_strength >= 0.6 else 'WEAK'}</b>"
            f" (raw: {fib_result['cluster_strength']:.2f} → adj: {adj_strength:.2f})",
        ]
        if risk_flag:
            lines.append(f"⚠️  {risk_flag}")
        lines += [
            "",
            f"SCENARIO A   ${fib_result['scenario_a_price']:>10,.2f}   → 10–20% entry",
            f"SCENARIO B   ${fib_result['scenario_b_price']:>10,.2f}   → full entry",
            "",
        ]
        if is_bullish:
            lines.append(f"❌ INVALIDATED below: ${fib_result['invalidation']:,.2f}")
        else:
            lines.append(f"❌ INVALIDATED above: ${fib_result['invalidation']:,.2f}")
    else:
        lines.append("⏳ No confluence zone confirmed at this time.")
        if fib_result:
            lines.append(
                f"   Target A: ${fib_result['target_a']:,.2f} | "
                f"Target B: ${fib_result['target_b']:,.2f} (not clustered)"
            )

    if tft_result:
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 <b>TFT QUANTILE FORECAST</b>",
        ]
        for h in [7, 14, 30, 60]:
            q10 = tft_result.get(f'q10_{h}d')
            q50 = tft_result.get(f'q50_{h}d')
            q90 = tft_result.get(f'q90_{h}d')
            if q50 is not None:
                match = ""
                if fib_result and valid:
                    lo = fib_result['cluster_lower']
                    hi = fib_result['cluster_upper']
                    if lo <= q50 <= hi:
                        match = " ✅ inside cluster"
                lines.append(f"  t+{h:2d}d  q10=${q10:>9,.0f}  q50=${q50:>9,.0f}  q90=${q90:>9,.0f}{match}")

    calendar_section, warning_section = get_economic_calendar_section(timeframe)
    if warning_section:
        lines.append(warning_section)
    if calendar_section:
        lines.append(calendar_section)

    lines += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"⏰ Next update in {'6h' if timeframe == '4H' else '12h'}",
    ]

    return "\n".join(lines)



# ─────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    timeframe:    str,
    dry_run:      bool = False,
    full_refetch: bool = False,
    model_path:   str  = DEFAULT_MODEL_PATH,
    db_path:      str  = SQLITE_PATH,
    version:      str  = "v5_relaxed",
) -> Dict:
    """
    Run the full 10-step inference pipeline for one timeframe.
    Returns the assembled result dict.
    """
    print(f"\n{'='*60}")
    print(f"  run_daily_analysis | BTC {timeframe} | "
      f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    # Step 1+2: Fetch + append
    df_raw = fetch_incremental(timeframe, full_refetch=full_refetch)

    # Step 3: Recalculate layers
    df_layers = recalculate_layers(df_raw, timeframe)

    # Step 4: Pivot detection
    macro_pivots, micro_pivots = detect_latest_pivots(df_layers, timeframe)

    # Step 5: Inference window
    window_df = prep_inference_window(df_layers, macro_pivots, timeframe)

    # Step 6: TFT inference
    tft_result = run_tft_inference(window_df, model_path)

    # Step 7: Fibonacci cluster
    fib_result = compute_fib_targets(macro_pivots, version=version)

    # Step 8: Calendar risk adjustment
    raw_strength = fib_result['cluster_strength'] if fib_result else 0.0

    adj_strength, risk_flag = apply_calendar_risk(raw_strength, window_df)

    candle_close = float(df_layers['close'].iloc[-1])
    current_price = fetch_spot_price(fallback_price=candle_close)

    # Step 9: SQLite write
    record = {
        "timeframe":             timeframe,
        "direction":             fib_result['direction'] if fib_result else "neutral",
        "btc_close_at_signal":   current_price,
        "cluster_valid":         int(fib_result['cluster_valid']) if fib_result else 0,
        "cluster_upper":         fib_result['cluster_upper']    if fib_result else None,
        "cluster_lower":         fib_result['cluster_lower']    if fib_result else None,
        "cluster_strength":      raw_strength,
        "cluster_strength_adj":  adj_strength,
        "target_a":              fib_result['target_a']         if fib_result else None,
        "target_b":              fib_result['target_b']         if fib_result else None,
        "scenario_a_price":      fib_result['scenario_a_price'] if fib_result else None,
        "scenario_b_price":      fib_result['scenario_b_price'] if fib_result else None,
        "invalidation_level":    fib_result['invalidation']     if fib_result else None,
        "c_top":                 fib_result['c_top']            if fib_result else None,
        "b_low":                 fib_result['b_low']            if fib_result else None,
        "calendar_risk_flag":    risk_flag,
        "macro_pivot_count":     len(macro_pivots),
        "micro_pivot_count":     len(micro_pivots),
    }
    # Attach TFT quantiles if available
    if tft_result:
        record.update(tft_result)

    if not dry_run:
        init_db(db_path)
        write_prediction(db_path, record)
        print(f"  [db] Prediction saved to {db_path}")
    else:
        print("  [db] DRY RUN — SQLite write skipped")

    message = build_telegram_message(
        timeframe     = timeframe,
        fib_result    = fib_result,
        tft_result    = tft_result,
        adj_strength  = adj_strength,
        risk_flag     = risk_flag,
        current_price = current_price,
        version       = version,
    )

    # Alert if confluence confirmed or significant structure found
    should_alert = fib_result is not None
    if should_alert:
        send_telegram(message, dry_run=dry_run)

    print(f"\n  Pipeline complete for {timeframe} ✅")
    return record


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Elliott Wave daily ingestion and inference orchestrator."
    )
    parser.add_argument(
        '--timeframe', '-t',
        nargs='+',
        default=['1D'],
        choices=['1D', '4H', '1W'],
        help="Timeframe(s) to run. Default: 1D"
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help="Run pipeline without writing to SQLite or sending Telegram alerts"
    )
    parser.add_argument(
        '--full-refetch',
        action='store_true',
        help="Re-fetch full history from Binance (slow — use rarely)"
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained .pt model weights (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        '--db',
        default=SQLITE_PATH,
        help=f"Path to SQLite predictions database (default: {SQLITE_PATH})"
    )
    parser.add_argument(
        '--version',
        default='v5_relaxed',
        choices=['v1_buggy', 'v4_log', 'v5_relaxed'],
        help="Fibonacci math version. Default: v5_relaxed"
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    tf_results = {}
    for tf in args.timeframe:
        try:
            res = run_pipeline(
                timeframe    = tf,
                dry_run      = args.dry_run,
                full_refetch = args.full_refetch,
                model_path   = args.model,
                db_path      = args.db,
                version      = args.version,
            )
            tf_results[tf] = res
        except Exception as e:
            print(f"\n❌ Pipeline failed for {tf}: {e}")
            import traceback
            traceback.print_exc()

    if len(tf_results) >= 2:
        print("\n" + "=" * 60)
        print("Running Multi-Timeframe Confluence")
        print("=" * 60)
        from src.btc.confluence.multi_tf import compute_multi_tf_confluence
        report = compute_multi_tf_confluence(tf_results)
        print(f"  [multi-tf] Dominant Bias: {report.dominant_bias.upper()}")
        print(f"  [multi-tf] Agreement: {report.agreement_count}/{len(tf_results)}")
        print(f"  [multi-tf] Confluence Score: {report.confluence_score}")
        if report.confluent_zone:
            print(f"  [multi-tf] Confluent Zone: ${report.confluent_zone[0]:,.2f} - ${report.confluent_zone[1]:,.2f}")
        print(f"  [multi-tf] Notes: {report.notes}")

        if report.confluence_score >= 0.4:
            msg = (
                f"🌐 <b>MULTI-TIMEFRAME CONFLUENCE REPORT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f" Bias: <b>{report.dominant_bias.upper()}</b> ({report.agreement_count}/3 timeframes agree)\n"
                f" Score: <b>{report.confluence_score:.2f}</b>\n"
                f" Notes: {report.notes}\n"
            )
            send_telegram(msg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()