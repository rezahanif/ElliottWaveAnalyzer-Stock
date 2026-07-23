import pytest
from src.waveconf.fib_engine.trendline import Trendline
from src.waveconf.pivots.pattern_detector import PatternDetector


def test_channel_ascending():
    # Both rising, near-parallel, constant width
    r = Trendline(start_price=100.0, end_price=110.0, slope_pct_per_bar=0.10, r_squared=0.95, pivot_count=4)
    s = Trendline(start_price=80.0, end_price=90.0, slope_pct_per_bar=0.10, r_squared=0.90, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is not None
    assert match.pattern_type == "channel_ascending"
    assert match.family == "channel"
    assert match.confidence > 0.75


def test_rising_wedge():
    # Both rising, range narrowing (converging)
    r = Trendline(start_price=100.0, end_price=108.0, slope_pct_per_bar=0.08, r_squared=0.95, pivot_count=4)
    s = Trendline(start_price=80.0, end_price=93.0, slope_pct_per_bar=0.13, r_squared=0.90, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is not None
    assert match.pattern_type == "rising_wedge"
    assert match.family == "wedge"
    assert match.width_change_pct < -3.5


def test_falling_wedge():
    # Both falling, range narrowing (converging)
    r = Trendline(start_price=100.0, end_price=87.0, slope_pct_per_bar=-0.13, r_squared=0.95, pivot_count=4)
    s = Trendline(start_price=80.0, end_price=72.0, slope_pct_per_bar=-0.08, r_squared=0.90, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is not None
    assert match.pattern_type == "falling_wedge"
    assert match.family == "wedge"


def test_symmetrical_triangle():
    # Resistance falling, support rising, converging without meeting/crossing inside window
    r = Trendline(start_price=100.0, end_price=92.0, slope_pct_per_bar=-0.10, r_squared=0.95, pivot_count=4)
    s = Trendline(start_price=80.0, end_price=88.0, slope_pct_per_bar=0.10, r_squared=0.90, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is not None
    assert match.pattern_type == "symmetrical_triangle"
    assert match.family == "triangle"


def test_ascending_triangle():
    # Flat resistance, rising support, converging
    r = Trendline(start_price=100.0, end_price=100.0, slope_pct_per_bar=0.0, r_squared=0.99, pivot_count=4)
    s = Trendline(start_price=80.0, end_price=95.0, slope_pct_per_bar=0.15, r_squared=0.90, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is not None
    assert match.pattern_type == "ascending_triangle"
    assert match.family == "triangle"


def test_descending_triangle():
    # Falling resistance, flat support, converging
    r = Trendline(start_price=100.0, end_price=85.0, slope_pct_per_bar=-0.15, r_squared=0.95, pivot_count=4)
    s = Trendline(start_price=80.0, end_price=80.0, slope_pct_per_bar=0.0, r_squared=0.99, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is not None
    assert match.pattern_type == "descending_triangle"
    assert match.family == "triangle"


def test_ascending_broadening_wedge():
    # Both rising, resistance steeper (diverging)
    r = Trendline(start_price=100.0, end_price=120.0, slope_pct_per_bar=0.20, r_squared=0.95, pivot_count=4)
    s = Trendline(start_price=80.0, end_price=90.0, slope_pct_per_bar=0.10, r_squared=0.90, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is not None
    assert match.pattern_type == "ascending_broadening_wedge"
    assert match.family == "broadening"


def test_invalid_crossed_lines():
    # Support crosses above resistance inside window
    r = Trendline(start_price=100.0, end_price=80.0, slope_pct_per_bar=-0.20, r_squared=0.95, pivot_count=4)
    s = Trendline(start_price=90.0, end_price=95.0, slope_pct_per_bar=0.05, r_squared=0.90, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is not None
    assert match.pattern_type == "invalid_crossed_lines"
    assert match.confidence == 0.0


def test_insufficient_pivots():
    r = Trendline(start_price=100.0, end_price=110.0, slope_pct_per_bar=0.10, r_squared=0.95, pivot_count=1)
    s = Trendline(start_price=80.0, end_price=90.0, slope_pct_per_bar=0.10, r_squared=0.90, pivot_count=4)
    
    detector = PatternDetector()
    match = detector.detect(r, s)
    
    assert match is None
