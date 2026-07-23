"""
diagnose_short_blindness.py
---------------------------
Diagnostic script to expose WHY the 1D timeframe never generates short signals.
Collects three datasets that directly measure the three root causes.

Run BEFORE making any fixes. Output gives you the baseline numbers to
validate whether fixes actually worked.

Usage:
    PYTHONPATH=. python scripts/diagnose_short_blindness.py

Output files:
    data/diagnostics/pivot_threshold_audit_1D.csv   — Issue 1: threshold calibration
    data/diagnostics/running_extreme_audit_1D.csv   — Issue 2: unconfirmed swing loss
    data/diagnostics/wave_structure_audit_1D.csv    — Issue 3: missed internal pivots
    data/diagnostics/summary.txt                    — Plain-language summary
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.waveconf.pivots.zigzag import ZigZagDetector, _ZigZagState
from src.waveconf.pivots.pivot_schema import SwingType, PivotLayer, WaveDegree


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────

def load_layers(timeframe: str) -> pd.DataFrame:
    path = f"data/pivots/BTC_{timeframe}_with_layers.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\nRun calculate_layers.py first."
        )
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw['data'], columns=raw['columns'])
    return df.sort_values('timestamp_ms').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# ISSUE 1: Threshold audit
# How large does a move need to be to confirm a pivot on 1D?
# How many real corrective highs (potential short entries) were
# missed because they didn't beat the threshold?
# ─────────────────────────────────────────────────────────────

def audit_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every local high (rolling 10-bar peak), measure:
    - What % move would be needed to confirm it as a macro pivot
    - What % the price actually reversed from it
    - Whether it was confirmed or missed

    If missed_highs >> confirmed_highs, the threshold is too coarse
    and is filtering out legitimate short setups.
    """
    records = []
    closes  = df['close'].values
    highs   = df['high'].values
    dates   = df['date'].values
    wall_st = df['wall_street_threshold_pct'].values
    behav   = df['behavioral_threshold_pct'].values

    window = 10  # local peak detection window

    for i in range(window, len(df) - window):
        # Is this bar a local high?
        if highs[i] != max(highs[i-window:i+window+1]):
            continue

        # How far did price actually drop from this high?
        future_lows = closes[i+1:i+51]  # next 50 bars
        if len(future_lows) == 0:
            continue
        min_future  = future_lows.min()
        actual_drop = (highs[i] - min_future) / highs[i] * 100

        # What threshold was required to confirm?
        required_macro = wall_st[i]
        required_micro = behav[i]

        confirmed_macro = actual_drop >= required_macro
        confirmed_micro = actual_drop >= required_micro

        records.append({
            'date':             dates[i],
            'high_price':       round(highs[i], 2),
            'min_next_50_bars': round(min_future, 2),
            'actual_drop_pct':  round(actual_drop, 4),
            'required_macro_pct': required_macro,
            'required_micro_pct': required_micro,
            'confirmed_as_macro': confirmed_macro,
            'confirmed_as_micro': confirmed_micro,
            'missed_by_macro_pct': round(required_macro - actual_drop, 4),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# ISSUE 2: Running extreme audit
# What unconfirmed swing extremes exist at the 6-hour check points?
# These are the short entries the system is blind to.
# ─────────────────────────────────────────────────────────────

def audit_running_extremes(df: pd.DataFrame, timeframe: str = '1D') -> pd.DataFrame:
    """
    Replay the ZigZag state machine and log the running extreme at
    every bar — even when no pivot is confirmed.

    The running extreme is the candidate top/bottom that the state
    machine is tracking but hasn't confirmed yet. For short setups,
    this is an unconfirmed high that price has been declining from
    but hasn't reversed enough to lock in.

    Captures: how many bars the running extreme sits there before
    either being confirmed or abandoned (never confirmed = missed pivot).
    """
    records = []

    # Manually replay the macro state machine
    from src.waveconf.pivots.pivot_schema import PivotLayer, WaveDegree
    sm = _ZigZagState(PivotLayer.MACRO, min_bars=3, degree=WaveDegree.INTERMEDIATE)

    confirmed_bars = set()

    for i, row in df.iterrows():
        prev_pivot_count = len(sm.pivots)
        sm.process_bar(row, i, 'wall_street_threshold_pct')
        new_confirmed = len(sm.pivots) > prev_pivot_count

        if new_confirmed:
            confirmed_bars.add(sm.pivots[-1].bar_index)

        # Log the running extreme state at every bar
        if sm.state is not None:
            pct_from_extreme = 0.0
            if sm.extreme_price > 0:
                pct_from_extreme = (float(row['close']) - sm.extreme_price) / sm.extreme_price * 100

            records.append({
                'bar_index':          i,
                'date':               row['date'],
                'close':              row['close'],
                'state':              sm.state,
                'running_extreme_price': sm.extreme_price,
                'running_extreme_bar':   sm.extreme_bar,
                'bars_tracking':      i - sm.extreme_bar,
                'pct_from_extreme':   round(pct_from_extreme, 4),
                'locked_threshold':   sm.locked_threshold * 100,
                'pct_needed_to_confirm': round(
                    sm.locked_threshold * 100 + pct_from_extreme
                    if sm.state == 'SEEKING_HIGH'
                    else sm.locked_threshold * 100 - pct_from_extreme, 4
                ),
                'just_confirmed_pivot': new_confirmed,
            })

    result_df = pd.DataFrame(records)

    # Flag cases where state was SEEKING_LOW (tracking a high) for many
    # bars without confirming — these are potential missed short setups
    if len(result_df) > 0:
        seeking_low = result_df[result_df['state'] == 'SEEKING_LOW'].copy()
        # Group consecutive SEEKING_LOW runs
        seeking_low['run_id'] = (seeking_low['state'] != seeking_low['state'].shift()).cumsum()
        run_lengths = seeking_low.groupby('run_id')['bars_tracking'].max()
        result_df['long_unconfirmed_run'] = False
        long_runs = run_lengths[run_lengths > 20].index
        long_run_bars = seeking_low[seeking_low['run_id'].isin(long_runs)]['bar_index']
        result_df.loc[result_df['bar_index'].isin(long_run_bars), 'long_unconfirmed_run'] = True

    return result_df


# ─────────────────────────────────────────────────────────────
# ISSUE 3: Internal wave structure audit
# Within each macro A→B→C swing, how many internal pivots
# (micro pivots) exist that represent tradeable sub-waves?
# ─────────────────────────────────────────────────────────────

def audit_internal_structure(df: pd.DataFrame, timeframe: str = '1D') -> pd.DataFrame:
    """
    Compare macro vs micro pivot lists.
    For each macro swing (A→B), count how many micro pivots fall inside.
    Micro pivots inside a bearish macro swing = internal short opportunities
    the system is currently ignoring.
    """
    detector = ZigZagDetector(timeframe=timeframe)
    result   = detector.run(df, asset='BTCUSD')

    macro_highs = [p for p in result.macro if p.is_high()]
    macro_lows  = [p for p in result.macro if p.is_low()]

    records = []

    for i in range(len(macro_highs) - 1):
        high_pivot = macro_highs[i]
        # Find the next macro low after this high
        next_lows = [p for p in macro_lows if p.bar_index > high_pivot.bar_index]
        if not next_lows:
            continue
        next_low = next_lows[0]

        swing_drop_pct = (high_pivot.price - next_low.price) / high_pivot.price * 100

        # Count micro pivots within this bearish swing
        micro_in_swing = [
            p for p in result.micro
            if high_pivot.bar_index <= p.bar_index <= next_low.bar_index
        ]
        micro_highs_in_swing = [p for p in micro_in_swing if p.is_high()]
        micro_lows_in_swing  = [p for p in micro_in_swing if p.is_low()]

        # Each micro high within a bearish macro swing = potential short entry
        short_opportunities = []
        for mh in micro_highs_in_swing:
            # How far did price drop from this micro high?
            post_bars = df[df.index > mh.bar_index]
            if len(post_bars) == 0:
                continue
            future_low = post_bars['close'].iloc[:30].min() if len(post_bars) >= 5 else None
            drop = (mh.price - future_low) / mh.price * 100 if future_low else None
            short_opportunities.append({
                'micro_high_price': mh.price,
                'drop_pct': round(drop, 2) if drop else None,
            })

        records.append({
            'macro_high_date':     datetime.fromtimestamp(
                high_pivot.timestamp_ms/1000, tz=timezone.utc
            ).strftime('%Y-%m-%d'),
            'macro_high_price':    round(high_pivot.price, 2),
            'macro_low_date':      datetime.fromtimestamp(
                next_low.timestamp_ms/1000, tz=timezone.utc
            ).strftime('%Y-%m-%d'),
            'macro_low_price':     round(next_low.price, 2),
            'swing_drop_pct':      round(swing_drop_pct, 2),
            'macro_bars':          next_low.bar_index - high_pivot.bar_index,
            'micro_pivots_inside': len(micro_in_swing),
            'micro_highs_inside':  len(micro_highs_in_swing),   # = short opportunities
            'micro_lows_inside':   len(micro_lows_in_swing),
            'short_opps_detail':   str(short_opportunities),
            'system_saw_any_short': len(micro_highs_in_swing) > 0,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# Threshold comparison: what if we used 0.5× or 0.3× multiplier?
# ─────────────────────────────────────────────────────────────

def audit_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run ZigZag with different ATR multipliers and compare pivot counts.
    Shows at what multiplier short pivots start appearing on 1D.
    """
    results = []
    close = df['close']
    tr_df = df.copy()

    tr_df['prev_close'] = close.shift(1)
    tr_df['tr'] = pd.concat([
        tr_df['high'] - tr_df['low'],
        (tr_df['high'] - tr_df['prev_close']).abs(),
        (tr_df['low']  - tr_df['prev_close']).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr_df['tr'].ewm(alpha=1/14, adjust=False).mean()

    for multiplier in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        test_df = df.copy()
        test_df['wall_street_threshold_pct'] = round((atr14 * multiplier / close) * 100, 2)
        test_df['behavioral_threshold_pct']  = test_df['wall_street_threshold_pct']

        try:
            det = ZigZagDetector(timeframe='1D')
            res = det.run(test_df.dropna(), asset='BTCUSD')
            macro_highs = sum(1 for p in res.macro if p.is_high())
            macro_lows  = sum(1 for p in res.macro if p.is_low())

            # Check if any macro high followed by lower close = bearish signal candidate
            bearish_candidates = 0
            for p in res.macro:
                if p.is_high():
                    post = test_df[test_df.index > p.bar_index]['close']
                    if len(post) > 5 and post.iloc[5] < p.price:
                        bearish_candidates += 1

            results.append({
                'atr_multiplier':      multiplier,
                'macro_pivot_count':   len(res.macro),
                'macro_highs':         macro_highs,
                'macro_lows':          macro_lows,
                'bearish_candidates':  bearish_candidates,
                'avg_threshold_pct':   round(test_df['wall_street_threshold_pct'].mean(), 2),
            })
        except Exception as e:
            results.append({'atr_multiplier': multiplier, 'error': str(e)})

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs('data/diagnostics', exist_ok=True)

    print("Loading 1D data...")
    try:
        df = load_layers('1D')
        print(f"  {len(df):,} bars | {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return

    lines = []

    # ── Issue 1: Threshold audit ─────────────────────────────
    print("\n[1/4] Threshold audit — how many highs were missed by macro threshold...")
    thresh_df = audit_thresholds(df)
    thresh_df.to_csv('data/diagnostics/pivot_threshold_audit_1D.csv', index=False)

    total_local_highs  = len(thresh_df)
    confirmed_macro    = thresh_df['confirmed_as_macro'].sum()
    confirmed_micro    = thresh_df['confirmed_as_micro'].sum()
    missed_macro       = total_local_highs - confirmed_macro
    avg_required       = thresh_df['required_macro_pct'].mean()
    avg_actual_drop    = thresh_df['actual_drop_pct'].mean()

    lines += [
        "=" * 60,
        "ISSUE 1 — THRESHOLD CALIBRATION",
        "=" * 60,
        f"Local highs detected (rolling 10-bar):  {total_local_highs}",
        f"Confirmed as MACRO pivot:               {confirmed_macro}  ({confirmed_macro/total_local_highs*100:.1f}%)",
        f"Confirmed as MICRO pivot:               {confirmed_micro}  ({confirmed_micro/total_local_highs*100:.1f}%)",
        f"MISSED by macro (potential short setup):{missed_macro}  ({missed_macro/total_local_highs*100:.1f}%)",
        f"Avg required macro threshold:           {avg_required:.2f}%",
        f"Avg actual drop from local high:        {avg_actual_drop:.2f}%",
        "",
        "→ If avg_actual_drop < avg_required_threshold by a large margin,",
        "  the macro threshold is too coarse for 1D short detection.",
        "  These missed highs are the short setups the system cannot see.",
    ]
    print(f"  Local highs: {total_local_highs} | Confirmed macro: {confirmed_macro} | Missed: {missed_macro}")

    # ── Issue 2: Running extreme audit ──────────────────────
    print("\n[2/4] Running extreme audit — unconfirmed swing highs at 6h checkpoints...")
    running_df = audit_running_extremes(df, '1D')
    running_df.to_csv('data/diagnostics/running_extreme_audit_1D.csv', index=False)

    seeking_low_df  = running_df[running_df['state'] == 'SEEKING_LOW']
    long_runs       = running_df['long_unconfirmed_run'].sum() if 'long_unconfirmed_run' in running_df.columns else 0
    avg_bars_before_confirm = seeking_low_df['bars_tracking'].mean() if len(seeking_low_df) > 0 else 0

    lines += [
        "",
        "=" * 60,
        "ISSUE 2 — RUNNING EXTREME (UNCONFIRMED PIVOT LOSS)",
        "=" * 60,
        f"Bars spent in SEEKING_LOW state:     {len(seeking_low_df)}",
        f"Avg bars tracking unconfirmed high:  {avg_bars_before_confirm:.1f}",
        f"Long unconfirmed runs (>20 bars):    {long_runs}",
        "",
        "→ Each SEEKING_LOW bar is a bar where the system IS tracking a",
        "  potential short entry but cannot act because the pivot is",
        "  unconfirmed. Long runs = the system tracked a real high for",
        "  weeks before either confirming or abandoning it.",
        "  Fix: expose running_extreme_price and pct_from_extreme in",
        "  the 6-hour check as a 'provisional' signal.",
    ]
    print(f"  Bars in SEEKING_LOW: {len(seeking_low_df)} | Long unconfirmed runs: {long_runs}")

    # ── Issue 3: Internal structure audit ───────────────────
    print("\n[3/4] Internal wave structure — micro short opportunities inside macro swings...")
    internal_df = audit_internal_structure(df, '1D')
    internal_df.to_csv('data/diagnostics/wave_structure_audit_1D.csv', index=False)

    if len(internal_df) > 0:
        total_swings    = len(internal_df)
        swings_with_micro_shorts = internal_df['system_saw_any_short'].sum()
        avg_micro_highs = internal_df['micro_highs_inside'].mean()
        best_swing = internal_df.loc[internal_df['swing_drop_pct'].idxmax()]

        lines += [
            "",
            "=" * 60,
            "ISSUE 3 — INTERNAL WAVE STRUCTURE (MISSED SUB-WAVE SHORTS)",
            "=" * 60,
            f"Macro bearish swings analyzed:          {total_swings}",
            f"Swings with micro short opportunities:  {swings_with_micro_shorts}",
            f"Avg micro highs per bearish macro swing:{avg_micro_highs:.1f}",
            f"Largest swing: {best_swing['macro_high_date']} ${best_swing['macro_high_price']:,.0f} → "
            f"${best_swing['macro_low_price']:,.0f} ({best_swing['swing_drop_pct']:.1f}% drop)",
            f"  Micro highs inside: {best_swing['micro_highs_inside']}  (each = 1 missed short entry)",
            "",
            "→ The system sees the macro high and low but not the",
            "  internal corrective bounces. Each micro high inside a",
            "  bearish macro swing is a missed short-entry opportunity.",
            "  Fix: run dual-layer (macro + micro) pivot detection and",
            "  flag micro highs within macro bearish swings as SHORT signals.",
        ]
        print(f"  Macro swings: {total_swings} | With micro short opps: {swings_with_micro_shorts} | "
              f"Avg micro highs/swing: {avg_micro_highs:.1f}")
    else:
        lines.append("  No macro swings found — check data availability")

    # ── Issue 4: Threshold sensitivity ──────────────────────
    print("\n[4/4] Threshold sensitivity — what multiplier unlocks short detection on 1D...")
    sens_df = audit_threshold_sensitivity(df)
    sens_df.to_csv('data/diagnostics/threshold_sensitivity_1D.csv', index=False)

    lines += [
        "",
        "=" * 60,
        "THRESHOLD SENSITIVITY — ATR MULTIPLIER vs PIVOT COUNT",
        "=" * 60,
    ]
    for _, row in sens_df.iterrows():
        if 'error' in row and pd.notna(row.get('error')):
            lines.append(f"  ×{row['atr_multiplier']:.1f}  ERROR: {row['error']}")
        else:
            lines.append(
                f"  ×{row['atr_multiplier']:.1f}  pivots={int(row['macro_pivot_count']):>4}  "
                f"highs={int(row['macro_highs']):>3}  "
                f"bearish_candidates={int(row['bearish_candidates']):>3}  "
                f"avg_threshold={row['avg_threshold_pct']:.2f}%"
            )
    lines += [
        "",
        "→ Find the multiplier where bearish_candidates first becomes",
        "  meaningful (>5-10) — that is your short-detection threshold.",
    ]
    print(sens_df[['atr_multiplier', 'macro_pivot_count', 'bearish_candidates',
                    'avg_threshold_pct']].to_string(index=False))

    # ── Write summary ────────────────────────────────────────
    lines += [
        "",
        "=" * 60,
        "RECOMMENDED FIXES (in priority order)",
        "=" * 60,
        "1. Add 'provisional signal' output from running_extreme_price",
        "   in the 6-hour check — do not wait for confirmation.",
        "2. Add micro-layer pivot detection to the signal pipeline.",
        "   Micro highs inside macro bearish swings = short signals.",
        "3. Add a BEARISH threshold layer (lower multiplier) for 1D",
        "   specifically for short detection. Use the sensitivity table",
        "   above to find the right multiplier.",
        "4. Log-scale Fibonacci: apply log transform before computing",
        "   extensions, back-transform the result. Lower priority —",
        "   fixes target precision, not directional blindness.",
    ]

    summary_text = "\n".join(lines)
    with open('data/diagnostics/summary.txt', 'w') as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"\n✅ Diagnostics saved to data/diagnostics/")


if __name__ == "__main__":
    main()