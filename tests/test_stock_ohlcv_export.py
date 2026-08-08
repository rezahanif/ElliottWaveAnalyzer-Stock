import json
from pathlib import Path

import pandas as pd

from src.stock.data.storage import BMRIStorage


def test_export_ohlcv_json_uses_dashboard_contract(tmp_path):
    root = tmp_path / "BMRI"
    storage = BMRIStorage(root)
    storage.save(pd.DataFrame([
        {"timestamp_ms": 1700000000000, "date": "2023-11-14", "open": 100,
         "high": 110, "low": 90, "close": 105, "volume": 1000},
    ]), "daily")
    out = storage.export_ohlcv_json("daily", tmp_path / "ohlcv" / "BMRI.JK_1D.json")
    payload = json.loads(Path(out).read_text())
    assert payload["asset"] == "BMRI.JK"
    assert payload["columns"] == ["timestamp_ms", "open", "high", "low", "close", "volume"]
    assert payload["data"] == [[1700000000000, 100, 110, 90, 105, 1000]]
