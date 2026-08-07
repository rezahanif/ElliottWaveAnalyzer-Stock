"""
test_pivot_sanitize.py
----------------------
Guards the NaN/Inf sanitizer for dashboard pivot serialization (Gap 3 lesson
from Phase 2 Task 6: the live-data bug was caught by a curl, not by tests,
because the JS fixture had no NaN in it).

The dashboard /api/pivots route reads data/pivots/*_pivots.json in Node,
whose JSON.parse rejects bare NaN/Infinity literals. PivotPoint.to_dict()
defaults rsi_at_pivot / macd_hist_at_pivot / fib_context to float('nan'),
so raw output would break the route. sanitize_pivot_dict must map those to
None and the result must survive a strict (Node-equivalent) parse.

Only stdlib (sqlite3-style) — no torch/pandas needed.
"""

import json
import math

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.btc.pivots.pivot_schema import PivotPoint, PivotLayer, SwingType, sanitize_pivot_dict  # noqa: E402


def _pivot_with_nans() -> PivotPoint:
    return PivotPoint(
        timestamp_ms=1609545600000,
        price=100.0,
        swing_type=SwingType.LOW,
        bar_index=1,
        layer=PivotLayer.MACRO,
        # Defaults rsi_at_pivot / macd_hist_at_pivot / fib_context = NaN
    )


def _strict_roundtrip(d: dict) -> dict:
    """Parse like Node's JSON.parse — reject NaN/Infinity literals."""
    return json.loads(
        json.dumps(d),
        parse_constant=lambda c: pytest.fail(f"bare {c} literal in JSON"),
    )


def test_sanitize_maps_nan_and_inf_to_none():
    d = _pivot_with_nans().to_dict()
    assert math.isnan(d["rsi_at_pivot"]), "precondition: raw dict has NaN"

    clean = sanitize_pivot_dict(d)
    assert clean["rsi_at_pivot"] is None
    assert clean["macd_hist_at_pivot"] is None
    assert clean["fib_context"] is None
    # Regular fields survive untouched.
    assert clean["price"] == 100.0
    assert clean["swing_type"] == "Low"
    assert clean["layer"] == "macro"


def test_sanitized_output_survives_strict_node_parse():
    raw = _pivot_with_nans().to_dict()
    raw["macd_hist_at_pivot"] = math.inf  # Inf must be sanitized too
    clean = sanitize_pivot_dict(raw)

    _strict_roundtrip(clean)  # raises via pytest.fail if NaN/Inf leaks


def test_serialized_file_contains_no_nan_literal(tmp_path):
    """The exact failure mode: JSON written to disk must parse in Node."""
    pivots = [_pivot_with_nans().to_dict() for _ in range(3)]
    out = {
        "asset": "BTCUSD",
        "timeframe": "1D",
        "macro": [sanitize_pivot_dict(p) for p in pivots],
        "micro": [sanitize_pivot_dict(p) for p in pivots],
    }
    path = tmp_path / "BTC_1D_pivots.json"
    path.write_text(json.dumps(out))

    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    loaded = json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(c))
    assert len(loaded["macro"]) == 3
