import pytest
from src.waveconf.pivots.pivot_schema import PivotPoint, PivotLayer, SwingType, StructureLabel, WaveDegree
from src.waveconf.structure.structure_tokenizer import StructureTokenizer, STRUCTURE_TOKENS


def _pivot(bar, price, swing_type, label, bars_from_prior=5, layer=PivotLayer.MACRO):
    return PivotPoint(
        timestamp_ms=bar * 1000,
        price=price,
        swing_type=swing_type,
        bar_index=bar,
        layer=layer,
        structure_label=label,
        degree=WaveDegree.INTERMEDIATE,
        bars_from_prior=bars_from_prior,
    )


def test_bos_on_continuation():
    # Up: HH, HH -> both BOS once trend established
    pivots = [
        _pivot(0, 100, SwingType.LOW, StructureLabel.UNKNOWN),
        _pivot(5, 110, SwingType.HIGH, StructureLabel.UNKNOWN),
        _pivot(10, 105, SwingType.LOW, StructureLabel.HL),
        _pivot(15, 120, SwingType.HIGH, StructureLabel.HH),
        _pivot(20, 112, SwingType.LOW, StructureLabel.HL),
        _pivot(25, 130, SwingType.HIGH, StructureLabel.HH),
    ]
    tokens = StructureTokenizer().run(pivots)
    bos = [t for t in tokens if t.token == "BOS"]
    assert len(bos) >= 1
    assert all(t.token != "CHOCH" for t in tokens)  # pure continuation, no CHOCH expected


def test_choch_on_reversal():
    # Uptrend established (HH/HL), then a LH should flip to CHOCH
    pivots = [
        _pivot(0, 100, SwingType.LOW, StructureLabel.UNKNOWN),
        _pivot(5, 110, SwingType.HIGH, StructureLabel.UNKNOWN),
        _pivot(10, 105, SwingType.LOW, StructureLabel.HL),
        _pivot(15, 120, SwingType.HIGH, StructureLabel.HH),   # establishes 'up' trend_state
        _pivot(20, 108, SwingType.LOW, StructureLabel.HL),
        _pivot(25, 115, SwingType.HIGH, StructureLabel.LH),   # opposes uptrend -> CHOCH
    ]
    tokens = StructureTokenizer().run(pivots)
    choch = [t for t in tokens if t.token == "CHOCH"]
    assert len(choch) == 1
    assert choch[0].bar_index == 25


def test_fib_tag_detects_618_retracement():
    # Leg: 100 -> 200 (length 100). Retracement to 138.2 is a 0.618 retrace.
    pivots = [
        _pivot(0, 100, SwingType.LOW, StructureLabel.UNKNOWN),
        _pivot(5, 200, SwingType.HIGH, StructureLabel.UNKNOWN),
        _pivot(10, 138.2, SwingType.LOW, StructureLabel.HL),
    ]
    tokens = StructureTokenizer().run(pivots)
    fib_tokens = [t for t in tokens if t.token == "FIB_T"]
    assert len(fib_tokens) == 1
    assert abs(fib_tokens[0].fib_context - 0.618) < 0.01


def test_sweep_detects_fast_breach_and_reversal():
    # Prior high at 110 (bar 5). New high at 112 (bar 22), only 2 bars after
    # its own preceding pivot (fast), then sharply reversed by next pivot.
    pivots = [
        _pivot(0, 100, SwingType.LOW, StructureLabel.UNKNOWN),
        _pivot(5, 110, SwingType.HIGH, StructureLabel.UNKNOWN),
        _pivot(10, 102, SwingType.LOW, StructureLabel.HL),
        _pivot(20, 108, SwingType.HIGH, StructureLabel.LH),
        _pivot(22, 112, SwingType.HIGH, StructureLabel.HH, bars_from_prior=2),  # fast breach of 110
        _pivot(24, 95, SwingType.LOW, StructureLabel.LL),  # sharp reversal
    ]
    tokens = StructureTokenizer().run(pivots)
    sweeps = [t for t in tokens if t.token == "SWEEP"]
    assert len(sweeps) == 1
    assert sweeps[0].bar_index == 22


def test_no_sweep_on_slow_clean_continuation():
    # Same shape but the breach is slow (bars_from_prior large) and the
    # follow-through doesn't reverse sharply -> should NOT be tagged SWEEP.
    pivots = [
        _pivot(0, 100, SwingType.LOW, StructureLabel.UNKNOWN),
        _pivot(5, 110, SwingType.HIGH, StructureLabel.UNKNOWN),
        _pivot(10, 102, SwingType.LOW, StructureLabel.HL),
        _pivot(40, 112, SwingType.HIGH, StructureLabel.HH, bars_from_prior=30),
        _pivot(45, 109, SwingType.LOW, StructureLabel.HL),  # shallow pullback only
    ]
    tokens = StructureTokenizer().run(pivots)
    sweeps = [t for t in tokens if t.token == "SWEEP"]
    assert len(sweeps) == 0


def test_divergence_requires_indicator_series():
    pivots = [
        _pivot(0, 100, SwingType.LOW, StructureLabel.UNKNOWN),
        _pivot(5, 110, SwingType.HIGH, StructureLabel.UNKNOWN),
        _pivot(10, 102, SwingType.LOW, StructureLabel.HL),
        _pivot(15, 120, SwingType.HIGH, StructureLabel.HH),
    ]
    # Without indicator series -> no DIV tokens at all
    tokens = StructureTokenizer().run(pivots)
    assert not any(t.token in ("DIV_H", "DIV_L") for t in tokens)

    # With indicator series showing bearish divergence on the HH at bar 15
    rsi = {5: 70.0, 15: 60.0}
    tokens_with_rsi = StructureTokenizer().run(pivots, rsi_series=rsi)
    div = [t for t in tokens_with_rsi if t.token == "DIV_H"]
    assert len(div) == 1
    assert div[0].bar_index == 15


def test_deferred_tokens_never_emitted():
    """W3_EXT, W4_REJ, DIAG, TRUNC, ABW_T require wave labels this module
    doesn't have access to. They must never appear in tokenizer output."""
    pivots = [
        _pivot(0, 100, SwingType.LOW, StructureLabel.UNKNOWN),
        _pivot(5, 110, SwingType.HIGH, StructureLabel.UNKNOWN),
        _pivot(10, 95, SwingType.LOW, StructureLabel.LL),
        _pivot(15, 130, SwingType.HIGH, StructureLabel.HH),
    ]
    tokens = StructureTokenizer().run(pivots)
    deferred = {"W3_EXT", "W4_REJ", "DIAG", "TRUNC", "ABW_T"}
    assert all(t.token not in deferred for t in tokens)


def test_vocabulary_ids_match_spec():
    assert STRUCTURE_TOKENS["HH"] == 0
    assert STRUCTURE_TOKENS["SWEEP"] == 12
    assert STRUCTURE_TOKENS["TRUNC"] == 14


def test_empty_pivot_list_returns_empty():
    assert StructureTokenizer().run([]) == []


def test_tokens_sorted_by_bar_index():
    pivots = [
        _pivot(0, 100, SwingType.LOW, StructureLabel.UNKNOWN),
        _pivot(5, 200, SwingType.HIGH, StructureLabel.UNKNOWN),
        _pivot(10, 138.2, SwingType.LOW, StructureLabel.HL),
        _pivot(15, 250, SwingType.HIGH, StructureLabel.HH),
    ]
    tokens = StructureTokenizer().run(pivots)
    bars = [t.bar_index for t in tokens]
    assert bars == sorted(bars)
