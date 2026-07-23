import json
import math
import random

import pytest

from src.waveconf.wave_model.dataset import DatasetBuilder


def _write_enriched_json(path, data, asset="BTCUSD", timeframe="1D"):
    out = {
        "asset": asset,
        "timeframe": timeframe,
        "columns": ["timestamp_ms", "open", "high", "low", "close", "volume",
                    "wall_street_threshold_pct", "behavioral_threshold_pct"],
        "data": data,
    }
    with open(path, "w") as f:
        json.dump(out, f)


def _sine_swing_series(n=220, macro_thr=5.0, micro_thr=2.0):
    """Generic oscillating series -- enough for pivot detection, not
    engineered for any particular pattern match."""
    random.seed(42)
    data = []
    ts0 = 1577836800000
    price = 20000.0
    for i in range(n):
        drift = i * 25
        wave = 6000 * math.sin(i / 18.0)
        target = 20000 + drift + wave
        price += (target - price) * 0.25 + random.uniform(-150, 150)
        o = price
        h = price * 1.01
        l = price * 0.99
        c = price + random.uniform(-100, 100)
        data.append([ts0 + i * 86400000, round(o, 2), round(h, 2), round(l, 2),
                     round(c, 2), 1500.0, macro_thr, micro_thr])
        price = c
    return data


def _clean_channel_series(n_cycles=9, rally_step=400, rally_bars=8,
                           pullback_step=280, pullback_bars=7, macro_thr=6.0):
    """Engineered ascending zigzag with pullbacks large enough to clear
    the macro threshold, so it reliably produces enough macro pivots to
    exercise pattern + wave classification with real (non-ambiguous)
    confidence -- verified manually to hit pattern_confidence > 0.8 and
    wave_match_confidence == 1.0 before this was written as a test."""
    data = []
    ts0 = 1577836800000
    price = 20000.0
    bar = 0
    for _ in range(n_cycles):
        for _ in range(rally_bars):
            price += rally_step
            data.append([ts0 + bar * 86400000, price - 50, price + 80, price - 100,
                         price, 1500.0, macro_thr, 2.5])
            bar += 1
        for _ in range(pullback_bars):
            price -= pullback_step
            data.append([ts0 + bar * 86400000, price + 50, price + 100, price - 80,
                         price, 1500.0, macro_thr, 2.5])
            bar += 1
    return data


@pytest.fixture
def sine_dataset_path(tmp_path):
    path = tmp_path / "BTC_1D_with_layers.json"
    _write_enriched_json(path, _sine_swing_series())
    return str(path)


@pytest.fixture
def clean_channel_dataset_path(tmp_path):
    path = tmp_path / "BTC_1D_with_layers.json"
    _write_enriched_json(path, _clean_channel_series())
    return str(path)


def test_build_runs_end_to_end(sine_dataset_path):
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(sine_dataset_path)
    assert len(labeled.df) == 220


def test_known_future_and_unknown_past_do_not_overlap(sine_dataset_path):
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(sine_dataset_path)
    overlap = set(labeled.known_future_columns) & set(labeled.unknown_past_columns)
    assert overlap == set()


def test_all_declared_columns_actually_exist(sine_dataset_path):
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(sine_dataset_path)
    for col in labeled.known_future_columns + labeled.unknown_past_columns:
        assert col in labeled.df.columns, f"declared column missing from df: {col}"


def test_target_is_nan_only_on_last_row(sine_dataset_path):
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(sine_dataset_path)
    nan_mask = labeled.df["close_pct_change"].isna()
    assert nan_mask.sum() == 1
    assert nan_mask.iloc[-1] == True  # noqa: E712 -- explicit bool check is clearer here


def test_target_matches_manual_pct_change(sine_dataset_path):
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(sine_dataset_path)
    df = labeled.df
    for i in [0, 50, 150]:
        expected = (df["close"].iloc[i + 1] - df["close"].iloc[i]) / df["close"].iloc[i]
        assert abs(df["close_pct_change"].iloc[i] - expected) < 1e-9


def test_known_future_is_computable_for_dates_with_no_price_data():
    """
    The defining property of the known-future channel: it must be
    computable for ANY date, not just dates where we happen to have a
    price candle. This is what makes it usable by the TFT as a forward
    input. Verify directly against the engines, independent of the
    OHLCV pipeline.
    """
    from datetime import date, timedelta
    from src.waveconf.wave_model.astro_features import AstroFeaturesEngine
    from src.waveconf.ingestion.economic_calender import EconomicCalendarEngine

    far_future = date.today() + timedelta(days=365 * 2)
    astro = AstroFeaturesEngine().get_daily_features(far_future)
    cal = EconomicCalendarEngine().get_context(far_future)
    assert -1.0 <= astro.lunar_phase_sin <= 1.0
    assert cal.as_of == far_future  # didn't raise, didn't need price data


def test_pattern_detection_produces_real_confidence_given_enough_pivots(clean_channel_dataset_path):
    """
    Track 1 wiring check: with a properly formed multi-swing series,
    pattern_type_id must move off 'none' (0) and pattern_confidence
    must exceed 0 somewhere in the data -- proving fit_trendline() +
    PatternDetector are actually being exercised, not silently no-op'd.
    """
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(clean_channel_dataset_path)
    assert labeled.df["pattern_confidence"].max() > 0.5
    assert len(builder.pattern_type_map) > 1  # more than just 'none'


def test_wave_classification_produces_real_confidence_given_enough_pivots(clean_channel_dataset_path):
    """Track 2 wiring check, same logic as the Track 1 test above."""
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(clean_channel_dataset_path)
    assert labeled.df["wave_match_confidence"].max() > 0.5
    assert len(builder.wave_type_map) > 1


def test_sparse_pivots_degrade_gracefully_not_with_an_exception(tmp_path):
    """Too few pivots to classify anything -- must NOT raise, must
    just leave pattern/wave columns at their 'none' defaults."""
    short_data = _sine_swing_series(n=15)
    path = tmp_path / "BTC_1D_with_layers.json"
    _write_enriched_json(path, short_data)

    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(str(path))  # must not raise
    assert (labeled.df["pattern_type_id"] == 0).all()


def test_missing_file_raises_clear_error():
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    with pytest.raises(FileNotFoundError):
        builder.build("data/pivots/this_file_does_not_exist.json")


def test_structure_token_id_forward_fills_not_left_blank(sine_dataset_path):
    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build(sine_dataset_path)
    # Once the first token fires, no row after it should still be the
    # "no structure yet" sentinel of -1 (forward-fill must actually work).
    df = labeled.df
    first_token_bar = df[df["structure_token_id"] != -1]["bar_index"].min()
    if first_token_bar is not None and not (df["structure_token_id"] == -1).all():
        after = df[df["bar_index"] > first_token_bar]
        assert (after["structure_token_id"] != -1).all()