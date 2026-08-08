import sqlite3
from types import SimpleNamespace

from scripts.migrate_dashboard_schema import migrate
from src.stock.persistence import persist_prediction


def test_persist_prediction_keeps_quantiles_null_without_ai(tmp_path, monkeypatch):
    db = tmp_path / "predictions.db"
    migrate(db)
    monkeypatch.setenv("PREDICTIONS_DB", str(db))
    zigzag = SimpleNamespace(macro=[], micro=[])
    analysis = {"direction": "bullish", "zigzag": zigzag, "fusion": None}
    forecast = {
        "signal": "BUY", "cluster_upper": 105.0, "cluster_lower": 95.0,
        "invalidation": 80.0,
    }
    confidence = {"overall": 0.71}
    row_id = persist_prediction(
        current_price=100.0,
        analysis_res=analysis,
        forecast=forecast,
        confidence=confidence,
    )
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT asset, direction, cluster_upper, cluster_lower, q10_7d, q50_7d, q90_7d "
        "FROM predictions WHERE id=?", (row_id,)
    ).fetchone()
    conn.close()
    assert row == ("BMRI.JK", "bullish", 105.0, 95.0, None, None, None)
