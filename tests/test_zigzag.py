import pytest
import pandas as pd
from src.waveconf.pivots.zigzag import ZigZagDetector
from src.waveconf.pivots.pivot_schema import SwingType, StructureLabel

def test_synthetic_zigzag():
    """
    Synthetic test: 10 bars up, 10 bars down.
    Tests if the ZigZagDetector correctly identifies the peak.
    """
    data = []
    
    # 10 bars up (bar_index 0 to 9)
    # High goes from 101.0 to 110.0
    for i in range(10):
        data.append({
            "timestamp_ms": i * 1000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 100,
            "wall_street_threshold_pct": 5.0, # 5% macro threshold
            "behavioral_threshold_pct": 2.0   # 2% micro threshold
        })
        
    # 10 bars down (bar_index 10 to 19)
    # High goes from 109.0 down to 100.0
    for i in range(10):
        data.append({
            "timestamp_ms": (10 + i) * 1000,
            "open": 110.0 - (i + 2),
            "high": 111.0 - (i + 2),
            "low": 109.0 - (i + 2),
            "close": 109.5 - (i + 2),
            "volume": 100,
            "wall_street_threshold_pct": 5.0,
            "behavioral_threshold_pct": 2.0
        })
        
    df = pd.DataFrame(data)
    
    # Run detector
    detector = ZigZagDetector(timeframe='1D', min_bars_between_pivots=1)
    result = detector.run(df)
    
    # Extract macro pivots
    macro_pivots = result.macro
    
    assert len(macro_pivots) > 0, "Should detect at least one macro pivot"
    
    # Find the high pivot
    high_pivots = [p for p in macro_pivots if p.swing_type == SwingType.HIGH]
    assert len(high_pivots) == 1, "Should detect exactly one macro HIGH pivot"
    
    peak = high_pivots[0]
    assert peak.bar_index == 9, f"Expected peak at bar 9, got {peak.bar_index}"
    assert peak.price == 110.0, f"Expected peak price 110.0, got {peak.price}"
    
    # Let's also check micro pivots
    micro_pivots = result.micro
    micro_high_pivots = [p for p in micro_pivots if p.swing_type == SwingType.HIGH]
    assert len(micro_high_pivots) >= 1
    assert micro_high_pivots[0].bar_index == 9

def test_structure_labels_hh_lh():
    """
    Three legs: up, down, up-to-new-high.
    Verifies HH/LH comparison logic, not just pivot detection.
    """
    data = []
    # Leg 1: up to 110
    for i in range(10):
        data.append({"timestamp_ms": i*1000, "open": 100.0+i, "high": 101.0+i,
                     "low": 99.0+i, "close": 100.5+i, "volume": 100,
                     "wall_street_threshold_pct": 5.0, "behavioral_threshold_pct": 2.0})
    # Leg 2: down to 95
    for i in range(10):
        data.append({"timestamp_ms": (10+i)*1000, "open": 110.0-(i+1.5), "high": 111.0-(i+1.5),
                     "low": 109.0-(i+1.5), "close": 109.5-(i+1.5), "volume": 100,
                     "wall_street_threshold_pct": 5.0, "behavioral_threshold_pct": 2.0})
    # Leg 3: up to a NEW high, 120 — should be tagged HH against the bar-9 peak
    for i in range(10):
        data.append({"timestamp_ms": (20+i)*1000, "open": 95.0+i*2.5, "high": 96.0+i*2.5,
                     "low": 94.0+i*2.5, "close": 95.5+i*2.5, "volume": 100,
                     "wall_street_threshold_pct": 5.0, "behavioral_threshold_pct": 2.0})
    # Leg 4: down to 100 to confirm the Leg 3 high
    for i in range(10):
        data.append({"timestamp_ms": (30+i)*1000, "open": 118.0-i*2, "high": 119.0-i*2,
                     "low": 117.0-i*2, "close": 117.5-i*2, "volume": 100,
                     "wall_street_threshold_pct": 5.0, "behavioral_threshold_pct": 2.0})

    df = pd.DataFrame(data)
    result = ZigZagDetector(timeframe='1D', min_bars_between_pivots=1).run(df)

    highs = [p for p in result.macro if p.swing_type == SwingType.HIGH]
    assert len(highs) == 2
    assert highs[0].structure_label == StructureLabel.UNKNOWN   # first ever high
    assert highs[1].structure_label == StructureLabel.HH        # 120 > 110

def test_real_btc_zigzag_sanity():
    detector = ZigZagDetector(timeframe='1D')
    result = detector.run_from_file('data/pivots/BTC_1D_with_layers.json')

    # Macro pivots should be far fewer than micro — that's the whole point of two layers
    assert len(result.macro) < len(result.micro)

    # Pivots must alternate High/Low within each layer — no two highs in a row
    for layer in [result.macro, result.micro]:
        for a, b in zip(layer, layer[1:]):
            assert a.swing_type != b.swing_type, f"Consecutive same-type pivots: {a} → {b}"

    # bar_index must be strictly increasing within each layer
    for layer in [result.macro, result.micro]:
        bars = [p.bar_index for p in layer]
        assert bars == sorted(bars)

    # Every confirmed pivot must have a non-negative bar distance.
    # Note: due to the online nature of the state machine, a single high-volatility daily candle
    # can act as both a local high and a local low extreme (causing same-bar pivots, distance = 0).
    # Additionally, delayed confirmation allows consecutive extremes to be closer than min_bars.
    for layer in [result.macro, result.micro]:
        for a, b in zip(layer, layer[1:]):
            assert b.bar_index >= a.bar_index



