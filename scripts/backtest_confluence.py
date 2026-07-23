import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
ROOT = Path("/Users/reza/ElliottWaveAnalyzer")
sys.path.insert(0, str(ROOT))

from src.waveconf.pivots.zigzag import ZigZagDetector
from src.waveconf.fib_engine.fibonacci import FibonacciEngine

def load_layers(timeframe: str) -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "labeled", f"BTC_{timeframe}_labeled.csv")
    df = pd.read_csv(path)
    df["timestamp_ms"] = df["timestamp_ms"].astype(float)
    return df

def detect_macro_regime(macro_pivots) -> str:
    """
    Returns 'BEARISH', 'BULLISH', or 'NEUTRAL' based on
    the last 4 macro pivots forming LH+LL or HH+HL sequence.
    """
    if len(macro_pivots) < 4:
        return 'NEUTRAL'
    
    last_4 = macro_pivots[-4:]
    highs = [p for p in last_4 if p.is_high()]
    lows  = [p for p in last_4 if p.is_low()]
    
    if len(highs) >= 2 and len(lows) >= 1:
        # LH sequence = bearish
        if highs[-1].price < highs[-2].price:
            return 'BEARISH'
        # HH sequence = bullish  
        if highs[-1].price > highs[-2].price:
            return 'BULLISH'
            
    return 'NEUTRAL'

def compute_fib_targets(macro_pivots, engine: FibonacciEngine, row_c: pd.Series, timeframe: str, config_name: str, use_log_scale: bool = True, use_regime_gate: bool = False) -> Optional[Dict]:
    if len(macro_pivots) < 4:
        return None

    highs = [p for p in macro_pivots if p.is_high()]
    lows  = [p for p in macro_pivots if p.is_low()]

    if not highs or not lows:
        return None

    last_pivot = macro_pivots[-1]

    if use_regime_gate:
        regime = detect_macro_regime(macro_pivots)
        if regime == 'BEARISH':
            direction = 'bearish'
        elif regime == 'BULLISH':
            direction = 'bullish'
        else:
            return None

        # Last pivot type must match the regime direction
        if direction == 'bearish' and not last_pivot.is_high():
            return None
        if direction == 'bullish' and not last_pivot.is_low():
            return None
    else:
        # Default: last pivot type determines direction
        if last_pivot.is_high():
            direction = 'bearish'
        else:
            direction = 'bullish'

    if direction == 'bearish':
        c_top_pivot = last_pivot
        b_low_pivot = max(
            (p for p in lows if p.bar_index < c_top_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        if b_low_pivot is None:
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

    else:
        c_top_pivot = last_pivot
        b_low_pivot = max(
            (p for p in highs if p.bar_index < c_top_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        if b_low_pivot is None:
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

    cluster = engine.dual_cluster(
        c_top         = c_top_val,
        b_low         = b_low_val,
        direction     = direction,
        ab_range      = ab_range,
        a_price       = a_price,
        version       = "v4_log" if use_log_scale else "v2_linear",
    )

    # Ignore negative targets
    if cluster.target_a.price <= 0 or cluster.target_b.price <= 0:
        return None

    if config_name == "baseline":
        stop_buffer = 0.005
        invalidation_level = round(c_top_pivot.price * (1 + stop_buffer), 2) if direction == "bearish" else round(c_top_pivot.price * (1 - stop_buffer), 2)
    else:
        # Hybrid Configuration
        # Filter 1: Wave Match Confidence (>=98% for 4H, >=80% for 1D)
        wmc = float(row_c["wave_match_confidence"])
        min_wmc = 0.98 if "4H" in timeframe else 0.80
        if wmc < min_wmc:
            return None

        # Filter 2: Target Distance (cap at 25% for 4H, 180% for 1D)
        target_lower = cluster.cluster_lower
        target_upper = cluster.cluster_upper
        entry_price = c_top_pivot.price
        target_dist_pct = abs(entry_price - target_lower if direction == "bullish" else entry_price - target_upper) / entry_price * 100
        max_target_dist = 25.0 if "4H" in timeframe else 180.0
        if target_dist_pct > max_target_dist:
            return None

        # Filter 3: Trend Momentum (MACD >= -250 on Daily Bullish)
        macd = float(row_c["macd_line"])
        if direction == "bullish" and "1D" in timeframe:
            if macd < -250.0:
                return None

        # Filter 4: Volatility-adjusted stop-loss
        atr_norm = float(row_c["atr_14_norm"])
        stop_buffer = max(0.005, atr_norm * 0.15)
        invalidation_level = round(c_top_pivot.price * (1 + stop_buffer), 2) if direction == "bearish" else round(c_top_pivot.price * (1 - stop_buffer), 2)

    return {
        "cluster_valid":      cluster.cluster_valid,
        "cluster_lower":      cluster.cluster_lower,
        "cluster_upper":      cluster.cluster_upper,
        "direction":          direction,
        "entry_price":        c_top_pivot.price,
        "invalidation_level": invalidation_level,
        "c_bar_index":        c_top_pivot.bar_index,
    }

def run_backtest(timeframe: str, config_name: str) -> Dict:
    df = load_layers(timeframe)
    detector = ZigZagDetector(timeframe=timeframe)
    zigzag_result = detector.run(df)
    macro_pivots = zigzag_result.macro

    engine = FibonacciEngine()
    signals = []

    # 1. Identify all valid confluence signals historically
    for i in range(3, len(macro_pivots)):
        sub_pivots = macro_pivots[:i+1]
        c_pivot = sub_pivots[-1]
        row_c = df.iloc[c_pivot.bar_index]
        res = compute_fib_targets(sub_pivots, engine, row_c, timeframe, config_name)
        if res is not None and res["cluster_valid"]:
            # Only add signal if it's not already in list (avoid duplicate analysis of same C pivot)
            if not any(s["c_bar_index"] == res["c_bar_index"] for s in signals):
                signals.append(res)

    if len(signals) == 0:
        return {
            "total": 0, "resolved": 0, "wins": 0, "losses": 0, "pending": 0, "win_rate": 0.0,
            "avg_bars_win": 0.0, "avg_bars_loss": 0.0, "avg_mfe": 0.0, "avg_mae": 0.0
        }

    results = []

    # 2. Evaluate price outcomes for each signal
    for signal in signals:
        c_bar = signal["c_bar_index"]
        entry_price = signal["entry_price"]
        invalidation = signal["invalidation_level"]
        cluster_lower = signal["cluster_lower"]
        cluster_upper = signal["cluster_upper"]
        direction = signal["direction"]

        outcome = "Pending"
        resolution_bar = len(df) - 1
        bars_to_res = len(df) - 1 - c_bar

        # Loop through future bars to find resolution
        for t in range(c_bar + 1, len(df)):
            row = df.iloc[t]
            low_val = float(row["low"])
            high_val = float(row["high"])

            if direction == "bullish":
                # Check invalidation breach first (conservative)
                is_invalid = low_val <= invalidation
                is_hit = high_val >= cluster_lower
                
                if is_invalid and is_hit:
                    # If both hit in same candle, treat as Loss (conservative)
                    outcome = "Loss"
                    resolution_bar = t
                    bars_to_res = t - c_bar
                    break
                elif is_invalid:
                    outcome = "Loss"
                    resolution_bar = t
                    bars_to_res = t - c_bar
                    break
                elif is_hit:
                    outcome = "Win"
                    resolution_bar = t
                    bars_to_res = t - c_bar
                    break

            elif direction == "bearish":
                is_invalid = high_val >= invalidation
                is_hit = low_val <= cluster_upper

                if is_invalid and is_hit:
                    outcome = "Loss"
                    resolution_bar = t
                    bars_to_res = t - c_bar
                    break
                elif is_invalid:
                    outcome = "Loss"
                    resolution_bar = t
                    bars_to_res = t - c_bar
                    break
                elif is_hit:
                    outcome = "Win"
                    resolution_bar = t
                    bars_to_res = t - c_bar
                    break

        # Calculate excursions over the trade's duration
        trade_bars = df.iloc[c_bar + 1 : resolution_bar + 1]
        
        if len(trade_bars) > 0:
            highs = trade_bars["high"].values
            lows = trade_bars["low"].values
            
            if direction == "bullish":
                max_fav = np.max(highs)
                min_adv = np.min(lows)
                mfe = (max_fav - entry_price) / entry_price * 100
                mae = (entry_price - min_adv) / entry_price * 100
            else: # bearish
                min_fav = np.min(lows)
                max_adv = np.max(highs)
                mfe = (entry_price - min_fav) / entry_price * 100
                mae = (max_adv - entry_price) / entry_price * 100
        else:
            mfe = 0.0
            mae = 0.0

        results.append({
            "c_bar": c_bar,
            "direction": direction,
            "outcome": outcome,
            "bars_to_resolution": bars_to_res,
            "mfe": mfe,
            "mae": mae,
        })

    # 3. Print Results Summary
    results_df = pd.DataFrame(results)
    resolved_df = results_df[results_df["outcome"] != "Pending"]
    wins = sum(results_df["outcome"] == "Win")
    losses = sum(results_df["outcome"] == "Loss")
    pending = sum(results_df["outcome"] == "Pending")
    
    total_resolved = wins + losses
    win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0.0
    
    avg_bars_win = results_df[results_df["outcome"] == "Win"]["bars_to_resolution"].mean()
    avg_bars_loss = results_df[results_df["outcome"] == "Loss"]["bars_to_resolution"].mean()
    avg_mfe = results_df["mfe"].mean()
    avg_mae = results_df["mae"].mean()

    # Time span in months
    timespan_ms = df.iloc[-1]["timestamp_ms"] - df.iloc[0]["timestamp_ms"]
    months = timespan_ms / (1000 * 60 * 60 * 24 * 30.4375)
    density = len(signals) / months if months > 0 else 0.0

    return {
        "total": len(signals),
        "density": density,
        "resolved": total_resolved,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "win_rate": win_rate,
        "avg_bars_win": avg_bars_win,
        "avg_bars_loss": avg_bars_loss,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae
    }

def main():
    timeframes = ["1D", "4H"]
    for tf in timeframes:
        try:
            b_metrics = run_backtest(tf, "baseline")
            h_metrics = run_backtest(tf, "hybrid")
            
            print(f"\n============================================================")
            print(f"  BACKTEST COMPARISON | BTC {tf}")
            print(f"============================================================")
            print(f"METRIC                    | BASELINE (Unfiltered) | HYBRID OPTIMIZED")
            print(f"------------------------------------------------------------")
            print(f"Total Signals             | {b_metrics['total']:<21} | {h_metrics['total']:<16}")
            print(f"Signal Density (/Month)   | {b_metrics['density']:.3f}/mo              | {h_metrics['density']:.3f}/mo")
            print(f"Resolved Trades           | {b_metrics['resolved']:<21} | {h_metrics['resolved']:<16}")
            print(f"  - Wins (Target Hit)     | {b_metrics['wins']:<21} | {h_metrics['wins']:<16}")
            print(f"  - Losses (Invalidated)  | {b_metrics['losses']:<21} | {h_metrics['losses']:<16}")
            print(f"Pending Trades            | {b_metrics['pending']:<21} | {h_metrics['pending']:<16}")
            print(f"Win Rate (Resolved)       | {b_metrics['win_rate']:.2f}%               | {h_metrics['win_rate']:.2f}%")
            
            avg_win_str_b = f"{b_metrics['avg_bars_win']:.1f} bars" if b_metrics['wins'] > 0 else "N/A"
            avg_win_str_h = f"{h_metrics['avg_bars_win']:.1f} bars" if h_metrics['wins'] > 0 else "N/A"
            print(f"Avg Bars to Win           | {avg_win_str_b:<21} | {avg_win_str_h}")
            
            avg_loss_str_b = f"{b_metrics['avg_bars_loss']:.1f} bars" if b_metrics['losses'] > 0 else "N/A"
            avg_loss_str_h = f"{h_metrics['avg_bars_loss']:.1f} bars" if h_metrics['losses'] > 0 else "N/A"
            print(f"Avg Bars to Loss          | {avg_loss_str_b:<21} | {avg_loss_str_h}")
            
            print(f"Avg MFE (Favorable %)     | {b_metrics['avg_mfe']:.2f}%              | {h_metrics['avg_mfe']:.2f}%")
            print(f"Avg MAE (Adverse %)       | {b_metrics['avg_mae']:.2f}%              | {h_metrics['avg_mae']:.2f}%")
            print(f"------------------------------------------------------------")
        except Exception as e:
            print(f"Error backtesting {tf}: {e}")

if __name__ == "__main__":
    main()
