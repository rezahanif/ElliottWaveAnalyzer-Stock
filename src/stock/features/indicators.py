"""
indicators.py
-------------
Stock-specific technical indicators.
Computes ALL features defined in the stock schema, using pandas/numpy only.

Does NOT touch BTC's src/btc/ingestion/indicators.py — that module serves
BTC's schema. This one serves the stock schema's full feature set:
  macd, macd_signal, macd_hist, atr_14, atr_20,
  ema_20, ema_50, ema_200, bb_upper, bb_lower, bb_width,
  obv, adx_14, rsi_14
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def add_rsi(df: pd.DataFrame, period: int = 14, close_col: str = "close") -> pd.DataFrame:
    """Wilder's RSI."""
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100.0)  # Pure uptrend → RSI = 100
    df[f"rsi_{period}"] = rsi
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    close_col: str = "close",
) -> pd.DataFrame:
    """MACD line, signal, histogram. Schema expects 'macd' not 'macd_line'."""
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # Schema uses 'macd' not 'macd_line'
    df["macd"] = macd_line
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
    """Average True Range (Wilder's smoothing)."""
    prev_close = df[close_col].shift(1)
    h_l = df[high_col] - df[low_col]
    h_pc = (df[high_col] - prev_close).abs()
    l_pc = (df[low_col] - prev_close).abs()
    true_range = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    df[f"atr_{period}"] = atr
    return df


def add_atr_20(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """ATR with period 20 for the schema."""
    prev_close = df["close"].shift(1)
    h_l = df["high"] - df["low"]
    h_pc = (df["high"] - prev_close).abs()
    l_pc = (df["low"] - prev_close).abs()
    true_range = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    df[f"atr_{period}"] = atr
    return df


def add_ema(df: pd.DataFrame, period: int, close_col: str = "close") -> pd.DataFrame:
    """Exponential Moving Average."""
    df[f"ema_{period}"] = df[close_col].ewm(span=period, adjust=False).mean()
    return df


def add_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    close_col: str = "close",
) -> pd.DataFrame:
    """Bollinger Bands: upper, lower, width."""
    sma = df[close_col].rolling(period).mean()
    std = df[close_col].rolling(period).std()

    df["bb_upper"] = sma + num_std * std
    df["bb_lower"] = sma - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df[close_col]
    return df


def add_obv(
    df: pd.DataFrame,
    close_col: str = "close",
    vol_col: str = "volume",
) -> pd.DataFrame:
    """On-Balance Volume."""
    direction = df[close_col].diff().apply(np.sign)
    df["obv"] = (direction * df[vol_col]).cumsum()
    return df


def add_adx(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """Average Directional Index (Wilder's method)."""
    # Plus/Minus Directional Movement
    up_move = df[high_col].diff()
    down_move = -df[low_col].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True Range
    prev_close = df[close_col].shift(1)
    tr = pd.concat(
        [df[high_col] - df[low_col],
         (df[high_col] - prev_close).abs(),
         (df[low_col] - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df[f"adx_{period}"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    return df


def add_stock_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ALL stock schema technical indicators.
    Returns a new DataFrame (copy).
    """
    out = df.copy()

    # RSI
    out = add_rsi(out, period=14)

    # MACD (schema expects 'macd', not 'macd_line')
    out = add_macd(out)

    # ATR (both 14 and 20)
    out = add_atr(out, period=14)
    out = add_atr_20(out, period=20)

    # EMAs
    out = add_ema(out, period=20)
    out = add_ema(out, period=50)
    out = add_ema(out, period=200)

    # Bollinger Bands
    out = add_bollinger(out, period=20)

    # OBV
    out = add_obv(out)

    # ADX
    out = add_adx(out, period=14)

    return out
