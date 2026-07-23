"""
indicators.py
-------------
Technical indicator enrichment. Pure pandas/numpy, no external TA library
dependency (keeps the 8GB home-server footprint small).

This was an empty stub referenced by structure_tokenizer.py's DIV_H/DIV_L
divergence detection (see that module's docstring) and required by
wave_model/dataset.py for the observed-past feature columns. Filling it
in is what makes both of those actually functional rather than silently
degraded.

Smoothing convention matches the existing ATR implementation in
ingestion/calculate_layers.py: Wilder's smoothing via
`.ewm(alpha=1/period, adjust=False)`, NOT a plain SMA-based EMA. Kept
consistent here so RSI/ATR don't disagree with the rest of the repo on
what "14-period smoothed" means.

Usage:
    from src.waveconf.ingestion.indicators import add_indicators

    df = add_indicators(df)   # adds rsi_14, macd_line, macd_signal,
                               # macd_hist, atr_14, atr_14_norm, bb_width,
                               # open_norm, high_norm, low_norm, close_norm,
                               # volume_norm

    # For StructureTokenizer divergence detection:
    rsi_series = dict(zip(df.index, df['rsi_14']))
    macd_hist_series = dict(zip(df.index, df['macd_hist']))
"""

from __future__ import annotations

import pandas as pd


def add_rsi(df: pd.DataFrame, period: int = 14, close_col: str = "close") -> pd.DataFrame:
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is exactly 0 (pure uptrend run), RS is undefined/inf -> RSI = 100
    rsi = rsi.fillna(100.0)

    df[f"rsi_{period}"] = rsi
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    close_col: str = "close",
) -> pd.DataFrame:
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    return df


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    prev_close = df[close_col].shift(1)
    h_l = df[high_col] - df[low_col]
    h_pc = (df[high_col] - prev_close).abs()
    l_pc = (df[low_col] - prev_close).abs()
    true_range = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()

    df[f"atr_{period}"] = atr
    df[f"atr_{period}_norm"] = atr / df[close_col]
    return df


def add_bollinger_width(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    close_col: str = "close",
) -> pd.DataFrame:
    sma = df[close_col].rolling(period).mean()
    std = df[close_col].rolling(period).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    # Normalized by close, per AGENT_BRIEFING Section 7: "bb_width = width / close"
    df["bb_width"] = (upper - lower) / df[close_col]
    return df


def add_price_normalization(df: pd.DataFrame, sma_period: int = 20) -> pd.DataFrame:
    """
    open_norm/high_norm/low_norm/close_norm: % deviation from the rolling
    SMA, per AGENT_BRIEFING Section 7. volume_norm: ratio to its own
    rolling mean (volume has no natural "price level" to normalize against).
    """
    sma = df["close"].rolling(sma_period).mean()

    for col in ["open", "high", "low", "close"]:
        df[f"{col}_norm"] = (df[col] - sma) / sma

    vol_sma = df["volume"].rolling(sma_period).mean()
    df["volume_norm"] = df["volume"] / vol_sma.replace(0, float("nan"))

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper: runs all of the above in the correct order.
    Returns a NEW DataFrame (works on a copy) so callers passing in a
    shared DataFrame aren't surprised by in-place mutation.
    """
    out = df.copy()
    out = add_rsi(out)
    out = add_macd(out)
    out = add_atr(out)
    out = add_bollinger_width(out)
    out = add_price_normalization(out)
    return out