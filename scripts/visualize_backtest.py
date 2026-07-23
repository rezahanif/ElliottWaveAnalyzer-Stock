import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
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

def compute_fib_targets_with_pivots(
    macro_pivots,
    engine: FibonacciEngine,
    row_c: pd.Series,
    timeframe: str,
    config_name: str,
    version: str = "v4_log"
) -> Optional[Dict]:
    if len(macro_pivots) < 4:
        return None

    highs = [p for p in macro_pivots if p.is_high()]
    lows  = [p for p in macro_pivots if p.is_low()]

    if not highs or not lows:
        return None

    last_pivot = macro_pivots[-1]

    use_regime_gate = (version == "v3_gated")
    use_log_scale = (version in ("v4_log", "v5_relaxed"))

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
        # Default: last pivot determines direction
        if last_pivot.is_high():
            direction = 'bearish'
        else:
            direction = 'bullish'

    if direction == 'bearish':
        c_pivot = last_pivot
        b_pivot = max(
            (p for p in lows if p.bar_index < c_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        if b_pivot is None:
            return None
            
        a_pivot = max(
            (p for p in highs if p.bar_index < b_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        ab_range = abs(a_pivot.price - b_pivot.price) if a_pivot else None
        a_price = a_pivot.price if a_pivot else None
        c_top_val = c_pivot.price
        b_low_val = b_pivot.price

    else:
        c_pivot = last_pivot
        b_pivot = max(
            (p for p in highs if p.bar_index < c_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        if b_pivot is None:
            return None
            
        a_pivot = max(
            (p for p in lows if p.bar_index < b_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        ab_range = abs(a_pivot.price - b_pivot.price) if a_pivot else None
        a_price = a_pivot.price if a_pivot else None
        c_top_val = b_pivot.price
        b_low_val = c_pivot.price

    cluster_version = version if version in ("v1_buggy", "v4_log", "v5_relaxed") else "v2_linear"

    cluster = engine.dual_cluster(
        c_top         = c_top_val,
        b_low         = b_low_val,
        direction     = direction,
        ab_range      = ab_range,
        a_price       = a_price,
        version       = cluster_version,
    )

    # Ignore negative targets
    if cluster.target_a.price <= 0 or cluster.target_b.price <= 0:
        return None

    if config_name == "baseline":
        stop_buffer = 0.005
        invalidation_level = round(c_pivot.price * (1 + stop_buffer), 2) if direction == "bearish" else round(c_pivot.price * (1 - stop_buffer), 2)
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
        entry_price = c_pivot.price
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
        invalidation_level = round(c_pivot.price * (1 + stop_buffer), 2) if direction == "bearish" else round(c_pivot.price * (1 - stop_buffer), 2)

    return {
        "cluster_valid":      cluster.cluster_valid,
        "cluster_lower":      cluster.cluster_lower,
        "cluster_upper":      cluster.cluster_upper,
        "direction":          direction,
        "entry_price":        c_pivot.price,
        "invalidation_level": invalidation_level,
        "c_bar_index":        c_pivot.bar_index,
        "timestamp_ms":       c_pivot.timestamp_ms,
        "pivot_A": {
            "bar_index": a_pivot.bar_index if a_pivot else -1,
            "price": a_pivot.price if a_pivot else 0.0,
            "timestamp_ms": a_pivot.timestamp_ms if a_pivot else 0,
            "type": "Low" if direction == "bullish" else "High"
        },
        "pivot_B": {
            "bar_index": b_pivot.bar_index,
            "price": b_pivot.price,
            "timestamp_ms": b_pivot.timestamp_ms,
            "type": "High" if direction == "bullish" else "Low"
        },
        "pivot_C": {
            "bar_index": c_pivot.bar_index,
            "price": c_pivot.price,
            "timestamp_ms": c_pivot.timestamp_ms,
            "type": "Low" if direction == "bullish" else "High"
        }
    }


def classify_signal_tier(
    entry_price: float,
    cluster_lower: float,
    cluster_upper: float,
    invalidation_level: float,
    direction: str,
    timeframe: str,
) -> str:
    """
    Classify a backtest signal as 'selective' or 'aggressive' based on
    structural quality metrics that mirror the live ConfluenceChecker tiers.

    SELECTIVE criteria (ALL must pass):
      1. Cluster width  < 10% of entry price  (tight zone)
      2. Target distance < 30% for 1D / 15% for 4H  (reasonable reach)
      3. Risk:Reward ratio > 1.5  (stop < 66% of target distance)
      4. Entry is inside or very near the cluster zone

    Everything else is AGGRESSIVE.
    """
    # Cluster width as % of entry
    cluster_width_pct = abs(cluster_upper - cluster_lower) / entry_price * 100

    # Target distance: entry -> nearest cluster edge
    if direction == "bullish":
        target_dist_pct = abs(cluster_lower - entry_price) / entry_price * 100
        risk_dist = abs(entry_price - invalidation_level) / entry_price * 100
    else:
        target_dist_pct = abs(entry_price - cluster_upper) / entry_price * 100
        risk_dist = abs(invalidation_level - entry_price) / entry_price * 100

    # Risk:reward ratio
    rr_ratio = target_dist_pct / risk_dist if risk_dist > 0 else 0.0

    # --- Tier checks ---
    max_cluster_width = 10.0
    max_target_dist = 30.0 if "1D" in timeframe else 15.0
    min_rr = 1.5

    is_tight_cluster = cluster_width_pct <= max_cluster_width
    is_reachable = target_dist_pct <= max_target_dist
    is_good_rr = rr_ratio >= min_rr

    if is_tight_cluster and is_reachable and is_good_rr:
        return "selective"
    return "aggressive"


def get_trades_for_timeframe(timeframe: str, config_name: str, version: str) -> List[Dict]:
    df = load_layers(timeframe)
    detector = ZigZagDetector(timeframe=timeframe)
    zigzag_result = detector.run(df)
    macro_pivots = zigzag_result.macro

    engine = FibonacciEngine()
    signals = []

    # Get signals using 0.5% baseline
    for i in range(3, len(macro_pivots)):
        sub_pivots = macro_pivots[:i+1]
        c_pivot = sub_pivots[-1]
        row_c = df.iloc[c_pivot.bar_index]
        res = compute_fib_targets_with_pivots(sub_pivots, engine, row_c, timeframe, config_name, version=version)
        if res is not None and res["cluster_valid"]:
            if not any(s["c_bar_index"] == res["c_bar_index"] for s in signals):
                signals.append(res)

    results = []

    for idx, signal in enumerate(signals):
        c_bar = signal["c_bar_index"]
        entry_price = signal["entry_price"]
        invalidation = signal["invalidation_level"]
        cluster_lower = signal["cluster_lower"]
        cluster_upper = signal["cluster_upper"]
        direction = signal["direction"]
        pivot_A = signal["pivot_A"]
        pivot_B = signal["pivot_B"]
        pivot_C = signal["pivot_C"]

        outcome = "Pending"
        resolution_bar = len(df) - 1
        bars_to_res = len(df) - 1 - c_bar

        # Loop through future bars to find resolution
        for t in range(c_bar + 1, len(df)):
            row = df.iloc[t]
            low_val = float(row["low"])
            high_val = float(row["high"])

            if direction == "bullish":
                is_invalid = low_val <= invalidation
                is_hit = high_val >= cluster_lower
                
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

        # Define window of candles to slice for visualization
        start_candle_idx = max(0, min(pivot_A["bar_index"], pivot_B["bar_index"], pivot_C["bar_index"]) - 15)
        end_candle_idx = min(len(df) - 1, resolution_bar + 15)

        candles_slice = []
        for index in range(start_candle_idx, end_candle_idx + 1):
            row = df.iloc[index]
            dt = pd.to_datetime(row["timestamp_ms"], unit="ms").strftime('%Y-%m-%d %H:%M')
            candles_slice.append({
                "bar_index": int(index),
                "date": dt,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            })

        start_date = pd.to_datetime(df.iloc[c_bar]["timestamp_ms"], unit="ms").strftime('%Y-%m-%d %H:%M')
        end_date = pd.to_datetime(df.iloc[resolution_bar]["timestamp_ms"], unit="ms").strftime('%Y-%m-%d %H:%M')

        # Classify signal tier
        tier = classify_signal_tier(
            entry_price=entry_price,
            cluster_lower=cluster_lower,
            cluster_upper=cluster_upper,
            invalidation_level=invalidation,
            direction=direction,
            timeframe=timeframe,
        )

        results.append({
            "id": idx + 1,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date if outcome != "Pending" else "N/A",
            "direction": direction.upper(),
            "outcome": outcome,
            "signal_tier": tier,
            "entry_price": float(entry_price),
            "invalidation_level": float(invalidation),
            "target_lower": float(cluster_lower),
            "target_upper": float(cluster_upper),
            "bars_held": int(bars_to_res),
            "mfe": round(float(mfe), 2),
            "mae": round(float(mae), 2),
            "pivot_A": {
                "bar_index": int(pivot_A["bar_index"]),
                "price": float(pivot_A["price"]),
                "date": pd.to_datetime(pivot_A["timestamp_ms"], unit="ms").strftime('%Y-%m-%d %H:%M'),
                "type": pivot_A["type"]
            },
            "pivot_B": {
                "bar_index": int(pivot_B["bar_index"]),
                "price": float(pivot_B["price"]),
                "date": pd.to_datetime(pivot_B["timestamp_ms"], unit="ms").strftime('%Y-%m-%d %H:%M'),
                "type": pivot_B["type"]
            },
            "pivot_C": {
                "bar_index": int(pivot_C["bar_index"]),
                "price": float(pivot_C["price"]),
                "date": pd.to_datetime(pivot_C["timestamp_ms"], unit="ms").strftime('%Y-%m-%d %H:%M'),
                "type": pivot_C["type"]
            },
            "resolution_bar": int(resolution_bar),
            "candles": candles_slice
        })

    return results

def get_dashboard_template() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elliott Wave Confluence Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.24.0.min.js"></script>
    <style>
        :root {
            --bg-deep: #090d16;
            --bg-card: #131926;
            --bg-card-hover: #1b2336;
            --border-color: #222e45;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --color-long: #10b981;
            --color-long-dim: rgba(16, 185, 129, 0.15);
            --color-long-border: rgba(16, 185, 129, 0.4);
            --color-short: #ef4444;
            --color-short-dim: rgba(239, 68, 68, 0.15);
            --color-short-border: rgba(239, 68, 68, 0.4);
            --color-accent: #06b6d4;
            --color-accent-dim: rgba(6, 182, 212, 0.1);
            --color-pending: #f59e0b;
            --color-selective: #a78bfa;
            --color-selective-dim: rgba(167, 139, 250, 0.15);
            --color-aggressive: #fb923c;
            --color-aggressive-dim: rgba(251, 146, 60, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-deep);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        header {
            background-color: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }

        .header-title h1 {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-title p {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
        }

        .stats-summary {
            display: flex;
            gap: 1.5rem;
        }

        .stat-badge {
            background-color: var(--bg-deep);
            border: 1px solid var(--border-color);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            text-align: center;
        }

        .stat-badge .value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-badge .label {
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.1rem;
        }

        .main-container {
            flex: 1;
            display: flex;
            overflow: hidden;
        }

        .sidebar {
            width: 320px;
            background-color: var(--bg-card);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .sidebar-controls {
            padding: 1.25rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .timeframe-toggle {
            display: flex;
            background-color: var(--bg-deep);
            border-radius: 8px;
            padding: 0.25rem;
            border: 1px solid var(--border-color);
        }

        .toggle-btn {
            flex: 1;
            background: none;
            border: none;
            color: var(--text-secondary);
            padding: 0.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
            font-family: inherit;
            transition: all 0.2s ease;
        }

        .toggle-btn.active {
            background-color: var(--border-color);
            color: var(--text-primary);
        }

        .filter-controls {
            display: flex;
            gap: 0.5rem;
        }

        .filter-select {
            flex: 1;
            background-color: var(--bg-deep);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.5rem;
            font-size: 0.8rem;
            font-family: inherit;
            outline: none;
        }

        .trade-list {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        /* Scrollbar styles */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-deep);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }

        .trade-card {
            background-color: var(--bg-deep);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }

        .trade-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 4px;
        }

        .trade-card.long::before {
            background-color: var(--color-long);
        }

        .trade-card.short::before {
            background-color: var(--color-short);
        }

        .trade-card:hover {
            background-color: var(--bg-card-hover);
            transform: translateY(-2px);
        }

        .trade-card.active {
            border-color: var(--color-accent);
            background-color: var(--bg-card-hover);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .card-title {
            font-size: 0.9rem;
            font-weight: 700;
        }

        .card-date {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .card-badge {
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 100px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .badge-win {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--color-long);
        }

        .badge-loss {
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--color-short);
        }

        .badge-pending {
            background-color: rgba(245, 158, 11, 0.15);
            color: var(--color-pending);
        }

        .badge-selective {
            background-color: var(--color-selective-dim);
            color: var(--color-selective);
            border: 1px solid rgba(167, 139, 250, 0.3);
        }

        .badge-aggressive {
            background-color: var(--color-aggressive-dim);
            color: var(--color-aggressive);
            border: 1px solid rgba(251, 146, 60, 0.3);
        }

        .tier-badge {
            font-size: 0.6rem;
            padding: 0.15rem 0.4rem;
            border-radius: 100px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .card-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
            border-top: 1px dashed var(--border-color);
            padding-top: 0.5rem;
        }

        .card-details div span {
            font-weight: 600;
            color: var(--text-primary);
        }

        .content-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background-color: var(--bg-deep);
        }

        .chart-wrapper {
            flex: 1;
            padding: 1.5rem;
            position: relative;
            min-height: 0;
        }

        #chart-container {
            width: 100%;
            height: 100%;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .inspection-panel {
            height: 240px;
            background-color: var(--bg-card);
            border-top: 1px solid var(--border-color);
            padding: 1.5rem 2rem;
            display: grid;
            grid-template-columns: 1.5fr 2fr 1fr;
            gap: 2rem;
            overflow-y: auto;
        }

        .panel-section {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .panel-section h3 {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
            margin-bottom: 0.25rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }

        .metric-item {
            display: flex;
            flex-direction: column;
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .metric-value {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 0.1rem;
        }

        .metric-value.win {
            color: var(--color-long);
        }

        .metric-value.loss {
            color: var(--color-short);
        }

        .metric-value.pending {
            color: var(--color-pending);
        }

        .wave-structure-flow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: var(--bg-deep);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.25rem;
        }

        .wave-node {
            text-align: center;
        }

        .wave-node .node-name {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--color-accent);
        }

        .wave-node .node-price {
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.1rem;
        }

        .wave-node .node-date {
            font-size: 0.65rem;
            color: var(--text-secondary);
            margin-top: 0.1rem;
        }

        .wave-arrow {
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 300;
        }

        .outcome-banner {
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
        }

        .outcome-banner.win {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--color-long-border);
        }

        .outcome-banner.loss {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--color-short-border);
        }

        .outcome-banner.pending {
            background-color: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.4);
        }

        .outcome-banner .status-title {
            font-size: 1.25rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .outcome-banner.win .status-title { color: var(--color-long); }
        .outcome-banner.loss .status-title { color: var(--color-short); }
        .outcome-banner.pending .status-title { color: var(--color-pending); }

        .outcome-banner .status-subtitle {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>

    <header>
        <div class="header-title">
            <h1>Elliott Wave Confluence Dashboard</h1>
            <p>Interactive Backtest & Signal Confluence Analyzer for BTC/USD</p>
        </div>
        <div class="stats-summary" id="stats-container">
            <!-- Dynamically populated stats -->
        </div>
    </header>

    <div class="main-container">
        <div class="sidebar">
            <div class="sidebar-controls">
                <div class="timeframe-toggle">
                    <button class="toggle-btn active" id="btn-1d" onclick="switchTimeframe('1D')">Daily (1D)</button>
                    <button class="toggle-btn" id="btn-4h" onclick="switchTimeframe('4H')">4-Hour (4H)</button>
                </div>
                <div class="timeframe-toggle" style="margin-top: 0.25rem;">
                    <button class="toggle-btn" id="btn-baseline" onclick="switchMode('baseline')">Baseline (Unfiltered)</button>
                    <button class="toggle-btn active" id="btn-hybrid" onclick="switchMode('hybrid')">Hybrid Optimized</button>
                </div>
                <div class="filter-controls" style="margin-top: 0.25rem;">
                    <select class="filter-select" id="filter-version" onchange="switchVersion(this.value)">
                        <option value="v5_relaxed">v5_relaxed (Relaxed log-scale)</option>
                        <option value="v4_log">v4_log (Log-scale default)</option>
                        <option value="v1_buggy">v1_buggy (Buggy original)</option>
                    </select>
                </div>
                <div class="filter-controls">
                    <select class="filter-select" id="filter-outcome" onchange="applyFilters()">
                        <option value="ALL">All Outcomes</option>
                        <option value="Win">Wins</option>
                        <option value="Loss">Losses</option>
                        <option value="Pending">Pending</option>
                    </select>
                    <select class="filter-select" id="filter-direction" onchange="applyFilters()">
                        <option value="ALL">All Directions</option>
                        <option value="BULLISH">Bullish (Long)</option>
                        <option value="BEARISH">Bearish (Short)</option>
                    </select>
                </div>
                <div class="filter-controls">
                    <select class="filter-select" id="filter-tier" onchange="applyFilters()">
                        <option value="ALL">All Tiers</option>
                        <option value="selective">★ Selective Only</option>
                        <option value="aggressive">⚡ Aggressive Only</option>
                    </select>
                </div>
            </div>
            <div class="trade-list" id="trade-list-container">
                <!-- Dynamically populated trade list -->
            </div>
        </div>

        <div class="content-panel">
            <div class="chart-wrapper">
                <div id="chart-container"></div>
            </div>

            <div class="inspection-panel">
                <div class="panel-section">
                    <h3>Trade Metrics</h3>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <span class="metric-label">Signal ID</span>
                            <span class="metric-value" id="detail-id">-</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Timeframe</span>
                            <span class="metric-value" id="detail-timeframe">-</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Direction</span>
                            <span class="metric-value" id="detail-direction">-</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Bars Held</span>
                            <span class="metric-value" id="detail-bars">-</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">MFE (Favorable)</span>
                            <span class="metric-value win" id="detail-mfe">-</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">MAE (Adverse)</span>
                            <span class="metric-value loss" id="detail-mae">-</span>
                        </div>
                    </div>
                </div>

                <div class="panel-section">
                    <h3>Elliott Wave Structure</h3>
                    <div class="wave-structure-flow">
                        <div class="wave-node">
                            <div class="node-name" id="node-a-title">Pivot A</div>
                            <div class="node-price" id="node-a-price">-</div>
                            <div class="node-date" id="node-a-date">-</div>
                        </div>
                        <div class="wave-arrow">&rarr;</div>
                        <div class="wave-node">
                            <div class="node-name" id="node-b-title">Pivot B</div>
                            <div class="node-price" id="node-b-price">-</div>
                            <div class="node-date" id="node-b-date">-</div>
                        </div>
                        <div class="wave-arrow">&rarr;</div>
                        <div class="wave-node">
                            <div class="node-name" id="node-c-title">Pivot C (Entry)</div>
                            <div class="node-price" id="node-c-price">-</div>
                            <div class="node-date" id="node-c-date">-</div>
                        </div>
                    </div>
                    <div class="metrics-grid" style="margin-top: 0.5rem;">
                        <div class="metric-item">
                            <span class="metric-label">Stop Level (Invalidation)</span>
                            <span class="metric-value loss" id="detail-stop">-</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Target Zone (Scenario A/B)</span>
                            <span class="metric-value win" id="detail-target">-</span>
                        </div>
                    </div>
                </div>

                <div class="panel-section">
                    <div class="outcome-banner" id="outcome-banner-box">
                        <div class="status-title" id="outcome-status">-</div>
                        <div class="status-subtitle" id="outcome-desc">-</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Inject data from python
        const dbData = __DATA_JSON__;

        let currentTF = '1D';
        let currentMode = 'hybrid';
        let currentVersion = 'v5_relaxed';
        let currentFilterOutcome = 'ALL';
        let currentFilterDirection = 'ALL';
        let currentFilterTier = 'ALL';
        let selectedTradeId = null;

        function init() {
            // Render initial view
            switchTimeframe('1D');
        }

        function switchTimeframe(tf) {
            currentTF = tf;
            document.getElementById('btn-1d').classList.toggle('active', tf === '1D');
            document.getElementById('btn-4h').classList.toggle('active', tf === '4H');
            
            // Recalculate summary stats for header
            renderStatsSummary();
            
            // Re-render sidebar
            renderTradeList();
            
            // Select first trade in list
            const trades = getFilteredTrades();
            if (trades.length > 0) {
                selectTrade(trades[0].id);
            } else {
                clearDetails();
            }
        }

        function switchMode(mode) {
            currentMode = mode;
            document.getElementById('btn-baseline').classList.toggle('active', mode === 'baseline');
            document.getElementById('btn-hybrid').classList.toggle('active', mode === 'hybrid');
            
            // Recalculate summary stats for header
            renderStatsSummary();
            
            // Re-render sidebar
            renderTradeList();
            
            // Select first trade in list
            const trades = getFilteredTrades();
            if (trades.length > 0) {
                selectTrade(trades[0].id);
            } else {
                clearDetails();
            }
        }

        function switchVersion(version) {
            currentVersion = version;
            
            // Recalculate summary stats for header
            renderStatsSummary();
            
            // Re-render sidebar
            renderTradeList();
            
            // Select first trade in list
            const trades = getFilteredTrades();
            if (trades.length > 0) {
                selectTrade(trades[0].id);
            } else {
                clearDetails();
            }
        }

        function getFilteredTrades() {
            const list = (((dbData[currentTF] || {})[currentMode] || {})[currentVersion]) || [];
            return list.filter(t => {
                const matchOutcome = currentFilterOutcome === 'ALL' || t.outcome === currentFilterOutcome;
                const matchDirection = currentFilterDirection === 'ALL' || t.direction === currentFilterDirection;
                const matchTier = currentFilterTier === 'ALL' || t.signal_tier === currentFilterTier;
                return matchOutcome && matchDirection && matchTier;
            });
        }

        function applyFilters() {
            currentFilterOutcome = document.getElementById('filter-outcome').value;
            currentFilterDirection = document.getElementById('filter-direction').value;
            currentFilterTier = document.getElementById('filter-tier').value;
            
            renderStatsSummary();
            renderTradeList();
            
            const trades = getFilteredTrades();
            if (trades.length > 0) {
                selectTrade(trades[0].id);
            } else {
                clearDetails();
            }
        }

        function renderStatsSummary() {
            const allTfTrades = (((dbData[currentTF] || {})[currentMode] || {})[currentVersion]) || [];
            const filteredTrades = getFilteredTrades();
            const wins = filteredTrades.filter(t => t.outcome === 'Win').length;
            const losses = filteredTrades.filter(t => t.outcome === 'Loss').length;
            const pending = filteredTrades.filter(t => t.outcome === 'Pending').length;
            const resolved = wins + losses;
            const winRate = resolved > 0 ? (wins / resolved * 100).toFixed(1) : '0.0';
            
            const selectiveCount = allTfTrades.filter(t => t.signal_tier === 'selective').length;
            const aggressiveCount = allTfTrades.filter(t => t.signal_tier === 'aggressive').length;
            
            // Selective win rate
            const selWins = allTfTrades.filter(t => t.signal_tier === 'selective' && t.outcome === 'Win').length;
            const selLosses = allTfTrades.filter(t => t.signal_tier === 'selective' && t.outcome === 'Loss').length;
            const selResolved = selWins + selLosses;
            const selWinRate = selResolved > 0 ? (selWins / selResolved * 100).toFixed(1) : '-';
            
            // MFE comparison
            const selMfe = allTfTrades.filter(t => t.signal_tier === 'selective' && t.outcome !== 'Pending');
            const aggMfe = allTfTrades.filter(t => t.signal_tier === 'aggressive' && t.outcome !== 'Pending');
            const avgSelMfe = selMfe.length > 0 ? (selMfe.reduce((s, t) => s + t.mfe, 0) / selMfe.length).toFixed(1) : '-';
            const avgAggMfe = aggMfe.length > 0 ? (aggMfe.reduce((s, t) => s + t.mfe, 0) / aggMfe.length).toFixed(1) : '-';
            
            const container = document.getElementById('stats-container');
            container.innerHTML = `
                <div class="stat-badge">
                    <div class="value">${filteredTrades.length}</div>
                    <div class="label">Total Signals</div>
                </div>
                <div class="stat-badge">
                    <div class="value" style="color: var(--color-long);">${wins}</div>
                    <div class="label">Wins</div>
                </div>
                <div class="stat-badge">
                    <div class="value" style="color: var(--color-short);">${losses}</div>
                    <div class="label">Losses</div>
                </div>
                <div class="stat-badge">
                    <div class="value" style="color: var(--color-pending);">${pending}</div>
                    <div class="label">Pending</div>
                </div>
                <div class="stat-badge">
                    <div class="value" style="color: ${winRate >= 40 ? 'var(--color-long)' : 'var(--text-primary)'};">${winRate}%</div>
                    <div class="label">Win Rate</div>
                </div>
                <div class="stat-badge">
                    <div class="value" style="color: var(--color-selective);">${selectiveCount}</div>
                    <div class="label">★ Selective (${selWinRate}% WR)</div>
                </div>
                <div class="stat-badge">
                    <div class="value" style="color: var(--color-aggressive);">${aggressiveCount}</div>
                    <div class="label">⚡ Aggressive (MFE ${avgAggMfe}%)</div>
                </div>
            `;
        }

        function renderTradeList() {
            const container = document.getElementById('trade-list-container');
            container.innerHTML = '';
            
            const trades = getFilteredTrades();
            
            if (trades.length === 0) {
                container.innerHTML = '<div style="color: var(--text-secondary); text-align: center; margin-top: 2rem; font-size: 0.9rem;">No signals match filters</div>';
                return;
            }
            
            trades.forEach(t => {
                const card = document.createElement('div');
                card.className = `trade-card ${t.direction.toLowerCase()} ${selectedTradeId === t.id ? 'active' : ''}`;
                card.id = `card-${t.id}`;
                card.onclick = () => selectTrade(t.id);
                
                const badgeClass = t.outcome === 'Win' ? 'badge-win' : (t.outcome === 'Loss' ? 'badge-loss' : 'badge-pending');
                const sign = t.direction === 'BULLISH' ? '&uarr;' : '&darr;';
                const tierBadge = t.signal_tier === 'selective' 
                    ? '<span class="tier-badge badge-selective">★ SEL</span>' 
                    : '<span class="tier-badge badge-aggressive">⚡ AGG</span>';
                
                card.innerHTML = `
                    <div class="card-header">
                        <span class="card-title" style="color: ${t.direction === 'BULLISH' ? 'var(--color-long)' : 'var(--color-short)'}">
                            ${sign} ${t.direction} #${t.id}
                        </span>
                        <div style="display: flex; gap: 0.3rem; align-items: center;">
                            ${tierBadge}
                            <span class="card-badge ${badgeClass}">${t.outcome}</span>
                        </div>
                    </div>
                    <div class="card-date">Entry: ${t.start_date}</div>
                    <div class="card-details">
                        <div>Entry: <span>$${t.entry_price.toLocaleString(undefined, {minimumFractionDigits:2})}</span></div>
                        <div>Target: <span>$${t.target_lower.toLocaleString(undefined, {minimumFractionDigits:2})}</span></div>
                    </div>
                `;
                container.appendChild(card);
            });
        }

        function selectTrade(id) {
            // Deselect old
            if (selectedTradeId !== null) {
                const oldCard = document.getElementById(`card-${selectedTradeId}`);
                if (oldCard) oldCard.classList.remove('active');
            }
            
            selectedTradeId = id;
            const card = document.getElementById(`card-${id}`);
            if (card) card.classList.add('active');
            
            const trades = (((dbData[currentTF] || {})[currentMode] || {})[currentVersion]) || [];
            const trade = trades.find(t => t.id === id);
            if (!trade) return;
            
            updateDetailsPanel(trade);
            renderPlotlyChart(trade);
        }

        function clearDetails() {
            selectedTradeId = null;
            document.getElementById('detail-id').innerText = '-';
            document.getElementById('detail-timeframe').innerText = '-';
            document.getElementById('detail-direction').innerText = '-';
            document.getElementById('detail-bars').innerText = '-';
            document.getElementById('detail-mfe').innerText = '-';
            document.getElementById('detail-mae').innerText = '-';
            
            document.getElementById('node-a-price').innerText = '-';
            document.getElementById('node-a-date').innerText = '-';
            document.getElementById('node-b-price').innerText = '-';
            document.getElementById('node-b-date').innerText = '-';
            document.getElementById('node-c-price').innerText = '-';
            document.getElementById('node-c-date').innerText = '-';
            
            document.getElementById('detail-stop').innerText = '-';
            document.getElementById('detail-target').innerText = '-';
            
            const banner = document.getElementById('outcome-banner-box');
            banner.className = 'outcome-banner';
            document.getElementById('outcome-status').innerText = '-';
            document.getElementById('outcome-desc').innerText = '-';
            
            document.getElementById('chart-container').innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding-top: 15%; font-size: 1.1rem;">Select a trade to display the chart</div>';
        }

        function updateDetailsPanel(trade) {
            document.getElementById('detail-id').innerText = `#${trade.id}`;
            document.getElementById('detail-timeframe').innerText = trade.timeframe;
            document.getElementById('detail-direction').innerText = trade.direction;
            document.getElementById('detail-bars').innerText = `${trade.bars_held} bars`;
            document.getElementById('detail-mfe').innerText = `+${trade.mfe.toFixed(2)}%`;
            document.getElementById('detail-mae').innerText = `-${trade.mae.toFixed(2)}%`;
            
            // Pivots
            document.getElementById('node-a-price').innerText = `$${trade.pivot_A.price.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById('node-a-date').innerText = trade.pivot_A.date.split(' ')[0];
            document.getElementById('node-a-title').innerText = `${trade.pivot_A.type} A`;

            document.getElementById('node-b-price').innerText = `$${trade.pivot_B.price.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById('node-b-date').innerText = trade.pivot_B.date.split(' ')[0];
            document.getElementById('node-b-title').innerText = `${trade.pivot_B.type} B`;

            document.getElementById('node-c-price').innerText = `$${trade.pivot_C.price.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById('node-c-date').innerText = trade.pivot_C.date.split(' ')[0];
            document.getElementById('node-c-title').innerText = `${trade.pivot_C.type} C`;
            
            // Levels
            document.getElementById('detail-stop').innerText = `$${trade.invalidation_level.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById('detail-target').innerText = `$${trade.target_lower.toLocaleString(undefined, {minimumFractionDigits:2})} - $${trade.target_upper.toLocaleString(undefined, {minimumFractionDigits:2})}`;
            
            // Banner
            const banner = document.getElementById('outcome-banner-box');
            banner.className = `outcome-banner ${trade.outcome.toLowerCase()}`;
            
            const statusEl = document.getElementById('outcome-status');
            const descEl = document.getElementById('outcome-desc');
            
            if (trade.outcome === 'Win') {
                statusEl.innerText = 'Win (Target Hit)';
                descEl.innerText = `Target zone hit after ${trade.bars_held} bars. Peak move: +${trade.mfe.toFixed(1)}%.`;
            } else if (trade.outcome === 'Loss') {
                statusEl.innerText = 'Loss (Stopped Out)';
                descEl.innerText = `Invalidation level breached. Loss contained within 0.5% stop range.`;
            } else {
                statusEl.innerText = 'Trade Pending';
                descEl.innerText = `Currently active. Current MFE: +${trade.mfe.toFixed(1)}% | MAE: -${trade.mae.toFixed(1)}%.`;
            }
        }

        function renderPlotlyChart(trade) {
            const candles = trade.candles;
            
            // Candlesticks
            const traceCandles = {
                x: candles.map(c => c.date),
                open: candles.map(c => c.open),
                high: candles.map(c => c.high),
                low: candles.map(c => c.low),
                close: candles.map(c => c.close),
                type: 'candlestick',
                name: 'Price',
                increasing: { line: { color: '#10b981', width: 1.5 }, fillcolor: 'rgba(16, 185, 129, 0.4)' },
                decreasing: { line: { color: '#ef4444', width: 1.5 }, fillcolor: 'rgba(239, 68, 68, 0.4)' }
            };

            // Zigzag paths
            const pA = candles.find(c => c.bar_index === trade.pivot_A.bar_index);
            const pB = candles.find(c => c.bar_index === trade.pivot_B.bar_index);
            const pC = candles.find(c => c.bar_index === trade.pivot_C.bar_index);
            
            const zigzagX = [];
            const zigzagY = [];
            const textLabels = [];
            
            if (pA) { zigzagX.push(pA.date); zigzagY.push(trade.pivot_A.price); textLabels.push('Pivot A'); }
            if (pB) { zigzagX.push(pB.date); zigzagY.push(trade.pivot_B.price); textLabels.push('Pivot B'); }
            if (pC) { zigzagX.push(pC.date); zigzagY.push(trade.pivot_C.price); textLabels.push('Pivot C (Entry)'); }
            
            const traceZigzag = {
                x: zigzagX,
                y: zigzagY,
                type: 'scatter',
                mode: 'lines+markers+text',
                name: 'Elliott Wave Swing',
                line: { color: '#f59e0b', width: 2, dash: 'dashdot' },
                marker: { size: 9, color: '#f59e0b', symbol: 'diamond' },
                text: textLabels,
                textposition: 'top center',
                textfont: { color: '#f8fafc', size: 11, weight: 'bold' }
            };

            // Entry line (dashed cyan) starting from Pivot C date to end
            const entryRange = candles.filter(c => c.bar_index >= trade.pivot_C.bar_index && c.bar_index <= trade.resolution_bar);
            const entryX = entryRange.map(c => c.date);
            const entryY = entryX.map(() => trade.entry_price);
            
            const traceEntry = {
                x: entryX,
                y: entryY,
                type: 'scatter',
                mode: 'lines',
                name: 'Entry Level',
                line: { color: '#06b6d4', width: 1.5, dash: 'dot' }
            };

            // Invalidation line (dashed red)
            const invalidationY = entryX.map(() => trade.invalidation_level);
            const traceInvalidation = {
                x: entryX,
                y: invalidationY,
                type: 'scatter',
                mode: 'lines',
                name: 'Invalidation Level (Stop)',
                line: { color: '#ef4444', width: 1.5, dash: 'dash' }
            };

            // Outcome marker at resolution bar
            const resCandle = candles.find(c => c.bar_index === trade.resolution_bar) || candles[candles.length - 1];
            let resPrice = trade.entry_price;
            if (trade.outcome === 'Win') {
                resPrice = trade.direction === 'BULLISH' ? trade.target_lower : trade.target_upper;
            } else if (trade.outcome === 'Loss') {
                resPrice = trade.invalidation_level;
            } else {
                resPrice = resCandle.close;
            }

            const traceOutcome = {
                x: [resCandle.date],
                y: [resPrice],
                type: 'scatter',
                mode: 'markers+text',
                name: 'Trade Outcome',
                marker: {
                    size: 14,
                    symbol: trade.outcome === 'Win' ? 'star' : (trade.outcome === 'Loss' ? 'x-thin' : 'circle-open'),
                    color: trade.outcome === 'Win' ? '#10b981' : (trade.outcome === 'Loss' ? '#ef4444' : '#f59e0b'),
                    line: { width: 3 }
                },
                text: [trade.outcome],
                textposition: 'top right',
                textfont: { color: '#f8fafc', size: 12, weight: 'bold' }
            };

            // Setup shapes for target cluster band
            const shapes = [];
            if (pC && resCandle) {
                shapes.push({
                    type: 'rect',
                    xref: 'x',
                    yref: 'y',
                    x0: pC.date,
                    y0: trade.target_lower,
                    x1: resCandle.date,
                    y1: trade.target_upper,
                    fillcolor: trade.direction === 'BULLISH' ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)',
                    line: { width: 1, color: trade.direction === 'BULLISH' ? 'rgba(16, 185, 129, 0.25)' : 'rgba(239, 68, 68, 0.25)', dash: 'dot' }
                });
            }

            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: '#0d131f',
                xaxis: {
                    gridcolor: '#1e293b',
                    linecolor: '#222e45',
                    tickfont: { color: '#94a3b8', family: 'Outfit, sans-serif' },
                    rangeslider: { visible: false },
                    spikethickness: 1,
                    spikecolor: '#94a3b8',
                    spikemode: 'across'
                },
                yaxis: {
                    gridcolor: '#1e293b',
                    linecolor: '#222e45',
                    tickfont: { color: '#94a3b8', family: 'Outfit, sans-serif' },
                    title: { text: 'Price (USD)', font: { color: '#94a3b8', family: 'Outfit, sans-serif', size: 12 } },
                    spikethickness: 1,
                    spikecolor: '#94a3b8',
                    spikemode: 'across'
                },
                legend: {
                    font: { color: '#94a3b8', family: 'Outfit, sans-serif', size: 11 },
                    orientation: 'h',
                    x: 0,
                    y: 1.15
                },
                shapes: shapes,
                margin: { t: 40, r: 20, b: 30, l: 60 },
                hovermode: 'x unified',
                dragmode: 'pan'
            };

            Plotly.newPlot('chart-container', [traceCandles, traceZigzag, traceEntry, traceInvalidation, traceOutcome], layout, {responsive: true});
        }

        window.onload = init;
    </script>
</body>
</html>
"""

def main():
    print("Generating trade visualization data...")
    versions = ["v1_buggy", "v4_log", "v5_relaxed"]
    db_data = {
        "1D": {
            "baseline": {v: get_trades_for_timeframe("1D", "baseline", v) for v in versions},
            "hybrid": {v: get_trades_for_timeframe("1D", "hybrid", v) for v in versions},
        },
        "4H": {
            "baseline": {v: get_trades_for_timeframe("4H", "baseline", v) for v in versions},
            "hybrid": {v: get_trades_for_timeframe("4H", "hybrid", v) for v in versions},
        }
    }
    print(f"Loaded {len(db_data['1D']['baseline']['v5_relaxed'])} baseline / {len(db_data['1D']['hybrid']['v5_relaxed'])} hybrid signals for 1D (v5_relaxed)")
    print(f"Loaded {len(db_data['4H']['baseline']['v5_relaxed'])} baseline / {len(db_data['4H']['hybrid']['v5_relaxed'])} hybrid signals for 4H (v5_relaxed)")

    # Compile the final dashboard HTML
    template = get_dashboard_template()
    data_json = json.dumps(db_data, indent=2)
    html_content = template.replace("__DATA_JSON__", data_json)

    # Save html page in the diagnostics folder
    output_dir = ROOT / "data" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "confluence_visualization.html"
    
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"\nDashboard successfully generated at:\n  {output_path}")

    # Copy to artifacts directory as well (only if it exists on local Mac)
    artifact_path = Path("/Users/reza/.gemini/antigravity-ide/brain/4d7e645b-a713-4573-8ccd-7c0f0d773b29/confluence_visualization.html")
    if artifact_path.parent.exists():
        try:
            with open(artifact_path, "w") as f:
                f.write(html_content)
            print(f"Also copied to artifacts folder:\n  {artifact_path}")
        except Exception as e:
            print(f"Failed to copy to artifacts: {e}")


if __name__ == "__main__":
    main()
