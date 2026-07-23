import pytest
from src.waveconf.pivots.pivot_schema import PivotPoint, SwingType, PivotLayer
from src.waveconf.pivots.classifiers import ImpulseClassifier, CorrectionClassifier, FailSeverity

def make_pivot(bar_index: int, price: float, swing_type: SwingType, layer: PivotLayer = PivotLayer.MACRO) -> PivotPoint:
    return PivotPoint(
        timestamp_ms=1718880000000 + bar_index * 86400000,
        price=price,
        swing_type=swing_type,
        bar_index=bar_index,
        layer=layer
    )

def test_standard_impulse():
    # standard_impulse: bullish
    pivots = [
        make_pivot(0, 100.0, SwingType.LOW),   # origin
        make_pivot(2, 150.0, SwingType.HIGH),  # Wave 1
        make_pivot(4, 120.0, SwingType.LOW),   # Wave 2 (retraces 60% of W1)
        make_pivot(6, 200.0, SwingType.HIGH),  # Wave 3 (r3=80, 1.6x of W1)
        make_pivot(8, 170.0, SwingType.LOW),   # Wave 4 (retraces 37.5% of W3, no overlap w1=150)
        make_pivot(10, 220.0, SwingType.HIGH), # Wave 5 (r5=50, 1.0x of W1)
    ]
    clf = ImpulseClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "standard_impulse"
    assert res.confidence > 0.8
    assert not res.truncation_flag

def test_impulse_w3_extension():
    pivots = [
        make_pivot(0, 100.0, SwingType.LOW),
        make_pivot(2, 120.0, SwingType.HIGH), # r1 = 20
        make_pivot(4, 110.0, SwingType.LOW),  # r2 = 10
        make_pivot(6, 200.0, SwingType.HIGH), # r3 = 90 (w3_ext = 4.5 >= 1.618)
        make_pivot(8, 160.0, SwingType.LOW),  # r4 = 40 (no overlap with w1=120)
        make_pivot(10, 190.0, SwingType.HIGH),# r5 = 30
    ]
    clf = ImpulseClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "impulse_w3_extension"

def test_impulse_w5_extension():
    pivots = [
        make_pivot(0, 100.0, SwingType.LOW),
        make_pivot(2, 120.0, SwingType.HIGH), # r1 = 20
        make_pivot(4, 110.0, SwingType.LOW),  # r2 = 10
        make_pivot(6, 130.0, SwingType.HIGH), # r3 = 20 (w3_vs_w1 = 1.0)
        make_pivot(8, 125.0, SwingType.LOW),  # r4 = 5
        make_pivot(10, 180.0, SwingType.HIGH),# r5 = 55 (w5_ext = 2.75 >= 1.618)
    ]
    clf = ImpulseClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "impulse_w5_extension"

def test_impulse_w1_extension():
    pivots = [
        make_pivot(0, 100.0, SwingType.LOW),
        make_pivot(2, 180.0, SwingType.HIGH), # r1 = 80
        make_pivot(4, 150.0, SwingType.LOW),  # r2 = 30
        make_pivot(6, 190.0, SwingType.HIGH), # r3 = 40
        make_pivot(8, 185.0, SwingType.LOW),  # r4 = 5
        make_pivot(10, 200.0, SwingType.HIGH),# r5 = 15
    ]
    clf = ImpulseClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "impulse_w1_extension"

def test_leading_diagonal():
    pivots = [
        make_pivot(0, 100.0, SwingType.LOW),
        make_pivot(2, 120.0, SwingType.HIGH), # r1 = 20
        make_pivot(4, 105.0, SwingType.LOW),  # r2 = 15
        make_pivot(6, 135.0, SwingType.HIGH), # r3 = 30
        make_pivot(8, 115.0, SwingType.LOW),  # r4 = 20 (overlaps w1=120)
        make_pivot(10, 142.0, SwingType.HIGH),# r5 = 27
    ]
    # slope_13 = (135-120)/4 = 3.75
    # slope_24 = (115-105)/4 = 2.5
    clf = ImpulseClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "leading_diagonal"
    assert res.diagonal_overlap_ok

def test_ending_diagonal():
    pivots = [
        make_pivot(0, 100.0, SwingType.LOW),
        make_pivot(2, 120.0, SwingType.HIGH), # r1 = 20
        make_pivot(4, 105.0, SwingType.LOW),  # r2 = 15
        make_pivot(6, 124.0, SwingType.HIGH), # r3 = 19 (< r1)
        make_pivot(8, 108.0, SwingType.LOW),  # r4 = 16 (overlaps w1=120)
        make_pivot(10, 125.0, SwingType.HIGH),# r5 = 17 (< r3)
    ]
    # slope_13 = (124-120)/4 = 1.0
    # slope_24 = (108-105)/4 = 0.75
    # r3 < r1 (19 < 20), r5 < r3 (17 < 19)
    clf = ImpulseClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "ending_diagonal"

def test_regular_flat():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 80.0, SwingType.LOW),   # A (rA = 20)
        make_pivot(4, 98.0, SwingType.HIGH),  # B (rB = 18, b_ret = 90%, no exceed)
        make_pivot(6, 79.0, SwingType.LOW),   # C (rC = 19, c_vs_a = 95%)
    ]
    clf = CorrectionClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "regular_flat"

def test_expanded_flat():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 80.0, SwingType.LOW),   # A (rA = 20)
        make_pivot(4, 105.0, SwingType.HIGH), # B (rB = 25, b_ret = 125%, B exceeds 100)
        make_pivot(6, 73.0, SwingType.LOW),   # C (rC = 32, c_vs_a = 160%)
    ]
    clf = CorrectionClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "expanded_flat"
    assert res.b_breach_expected

def test_running_flat():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 80.0, SwingType.LOW),   # A (rA = 20)
        make_pivot(4, 103.0, SwingType.HIGH), # B (rB = 23, b_ret = 115% > 110%)
        make_pivot(6, 85.0, SwingType.LOW),   # C (rC = 18, c_vs_a = 90% < 100% truncated)
    ]
    clf = CorrectionClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "running_flat"

def test_single_zigzag():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 80.0, SwingType.LOW),   # A (rA = 20)
        make_pivot(4, 90.0, SwingType.HIGH),  # B (rB = 10, b_ret = 50% <= 78.6%)
        make_pivot(6, 70.0, SwingType.LOW),   # C (rC = 20, c_vs_a = 100%)
    ]
    clf = CorrectionClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "single_zigzag"

def test_double_zigzag():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 80.0, SwingType.LOW),   # W (rW = 20)
        make_pivot(4, 90.0, SwingType.HIGH),  # X (rX = 10, x_ret = 50%)
        make_pivot(6, 72.0, SwingType.LOW),   # Y (rY = 18, y_vs_w = 90%)
    ]
    clf = CorrectionClassifier()
    res = clf._try_double_zigzag(pivots)
    assert res.matched
    assert res.pattern_type == "double_zigzag"

def test_contracting_symmetrical_triangle():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 60.0, SwingType.LOW),   # A (rA = 40)
        make_pivot(4, 90.0, SwingType.HIGH),  # B (rB = 30)
        make_pivot(6, 68.0, SwingType.LOW),   # C (rC = 22)
        make_pivot(8, 84.0, SwingType.HIGH),  # D (rD = 16)
        make_pivot(10, 72.0, SwingType.LOW),  # E (rE = 12)
    ]
    clf = CorrectionClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "contracting_symmetrical"

def test_contracting_ascending_triangle():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 70.0, SwingType.LOW),   # A (rA = 30)
        make_pivot(4, 95.0, SwingType.HIGH),  # B (rB = 25)
        make_pivot(6, 75.0, SwingType.LOW),   # C (rC = 20) - rising support
        make_pivot(8, 95.0, SwingType.HIGH),  # D (rD = 20) - flat top top top
        make_pivot(10, 80.0, SwingType.LOW),  # E (rE = 15)
    ]
    clf = CorrectionClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "contracting_ascending"

def test_contracting_descending_triangle():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 70.0, SwingType.LOW),   # A (rA = 30)
        make_pivot(4, 95.0, SwingType.HIGH),  # B (rB = 25)
        make_pivot(6, 70.0, SwingType.LOW),   # C (rC = 25) - flat bottom
        make_pivot(8, 90.0, SwingType.HIGH),  # D (rD = 20) - declining resistance
        make_pivot(10, 75.0, SwingType.LOW),  # E (rE = 15)
    ]
    clf = CorrectionClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "contracting_descending"

def test_expanding_triangle():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 90.0, SwingType.LOW),   # A (rA = 10)
        make_pivot(4, 105.0, SwingType.HIGH), # B (rB = 15)
        make_pivot(6, 83.0, SwingType.LOW),   # C (rC = 22)
        make_pivot(8, 113.0, SwingType.HIGH), # D (rD = 30)
        make_pivot(10, 73.0, SwingType.LOW),  # E (rE = 40)
    ]
    clf = CorrectionClassifier()
    res = clf.classify(pivots)
    assert res.matched
    assert res.pattern_type == "expanding_triangle"

def test_double_three():
    pivots = [
        make_pivot(0, 100.0, SwingType.HIGH), # origin
        make_pivot(2, 80.0, SwingType.LOW),   # W (rW = 20)
        make_pivot(4, 90.0, SwingType.HIGH),  # X (rX = 10, x_ret = 50%)
        make_pivot(6, 75.0, SwingType.LOW),   # Y (rY = 15)
    ]
    clf = CorrectionClassifier()
    res = clf._try_double_three(pivots)
    assert res.matched
    assert res.pattern_type == "double_three"

def test_invalid_input_count():
    clf_imp = ImpulseClassifier()
    res_imp = clf_imp.classify([])
    assert not res_imp.matched
    assert "invalid_input" in res_imp.pattern_type

    clf_corr = CorrectionClassifier()
    res_corr = clf_corr.classify([])
    assert not res_corr.matched
    assert "invalid_input" in res_corr.pattern_type
