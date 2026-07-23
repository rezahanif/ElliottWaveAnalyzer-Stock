from src.waveconf.fib_engine.trendline import fit_trendline
from src.waveconf.pivots.pivot_schema import PivotPoint, SwingType, PivotLayer, StructureLabel, WaveDegree


def _mk(bar, price):
    return PivotPoint(
        timestamp_ms=bar * 1000, price=price, swing_type=SwingType.HIGH, bar_index=bar,
        layer=PivotLayer.MACRO, candle_high=price, candle_low=price, candle_close=price,
        volume=0, threshold_used=0.05, structure_label=StructureLabel.UNKNOWN,
        degree=WaveDegree.PRIMARY, swing_magnitude_pct=0, bars_from_prior=0,
    )


def test_perfect_line_gives_r_squared_one():
    pivots = [_mk(b, 100 + 2 * b) for b in [0, 5, 10, 15]]
    tl = fit_trendline(pivots)
    assert tl.r_squared > 0.999
    assert tl.slope_pct_per_bar > 0


def test_falling_line_has_negative_slope():
    pivots = [_mk(b, 200 - 3 * b) for b in [0, 5, 10, 15]]
    tl = fit_trendline(pivots)
    assert tl.slope_pct_per_bar < 0


def test_flat_line_has_near_zero_slope():
    pivots = [_mk(b, 100.0) for b in [0, 5, 10, 15]]
    tl = fit_trendline(pivots)
    assert abs(tl.slope_pct_per_bar) < 0.01


def test_noisy_line_has_lower_r_squared_than_perfect_line():
    perfect = fit_trendline([_mk(b, 100 + 2 * b) for b in [0, 5, 10, 15]])
    noisy = fit_trendline([_mk(0, 100), _mk(5, 95), _mk(10, 140), _mk(15, 90)])
    assert noisy.r_squared < perfect.r_squared


def test_single_pivot_returns_none():
    assert fit_trendline([_mk(0, 100)]) is None


def test_empty_list_returns_none():
    assert fit_trendline([]) is None


def test_unordered_input_is_sorted_before_fitting():
    """Pivots passed out of bar_index order must still fit correctly."""
    ordered = fit_trendline([_mk(0, 100), _mk(5, 110), _mk(10, 120)])
    shuffled = fit_trendline([_mk(10, 120), _mk(0, 100), _mk(5, 110)])
    assert abs(ordered.slope_pct_per_bar - shuffled.slope_pct_per_bar) < 1e-9