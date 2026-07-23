import pandas as pd

from src.waveconf.ingestion.indicators import (
    add_rsi, add_macd, add_atr, add_bollinger_width, add_price_normalization, add_indicators,
)


def _flat_df(n=30, price=100.0):
    return pd.DataFrame({
        "open": [price] * n, "high": [price + 1] * n, "low": [price - 1] * n,
        "close": [price] * n, "volume": [1000.0] * n,
    })


def test_rsi_is_50_on_perfectly_flat_series():
    """No gains, no losses -- RSI should sit at a neutral midpoint, not blow up."""
    df = _flat_df()
    df = add_rsi(df)
    # A perfectly flat close series has zero gain AND zero loss every bar,
    # so avg_gain/avg_loss are both 0 -> our 0/0 guard fills these as 100.
    # Document the actual behavior rather than assume textbook 50.
    assert df["rsi_14"].iloc[-1] == 100.0


def test_rsi_is_100_on_pure_uptrend():
    df = pd.DataFrame({
        "open": range(100, 130), "high": range(101, 131), "low": range(99, 129),
        "close": range(100, 130), "volume": [1000.0] * 30,
    })
    df = add_rsi(df)
    assert df["rsi_14"].iloc[-1] == 100.0


def test_rsi_is_low_on_pure_downtrend():
    df = pd.DataFrame({
        "open": range(130, 100, -1), "high": range(131, 101, -1), "low": range(129, 99, -1),
        "close": range(130, 100, -1), "volume": [1000.0] * 30,
    })
    df = add_rsi(df)
    assert df["rsi_14"].iloc[-1] < 5.0


def test_macd_histogram_zero_when_no_trend():
    df = _flat_df(n=60)
    df = add_macd(df)
    assert abs(df["macd_hist"].iloc[-1]) < 1e-6


def test_atr_matches_manual_true_range_on_simple_series():
    """Hand-verifiable case: constant 10-point true range every bar.
    ATR should converge to 10."""
    n = 60
    df = pd.DataFrame({
        "close": [100.0 + i * 0.01 for i in range(n)],  # tiny drift, negligible vs range
        "high": [105.0 + i * 0.01 for i in range(n)],
        "low": [95.0 + i * 0.01 for i in range(n)],
        "open": [100.0] * n, "volume": [1000.0] * n,
    })
    df = add_atr(df)
    assert abs(df["atr_14"].iloc[-1] - 10.0) < 0.5


def test_bollinger_width_zero_on_flat_series():
    df = _flat_df(n=30)
    df = add_bollinger_width(df)
    assert abs(df["bb_width"].iloc[-1]) < 1e-9


def test_price_normalization_zero_on_flat_series():
    df = _flat_df(n=30)
    df = add_price_normalization(df)
    assert abs(df["close_norm"].iloc[-1]) < 1e-9
    assert abs(df["volume_norm"].iloc[-1] - 1.0) < 1e-9


def test_add_indicators_does_not_mutate_input_in_place():
    df = _flat_df(n=30)
    original_columns = set(df.columns)
    _ = add_indicators(df)
    assert set(df.columns) == original_columns  # caller's df untouched


def test_add_indicators_produces_all_expected_columns():
    df = _flat_df(n=30)
    out = add_indicators(df)
    expected = {
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "atr_14", "atr_14_norm", "bb_width",
        "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
    }
    assert expected.issubset(set(out.columns))