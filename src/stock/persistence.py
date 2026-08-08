"""Shared BMRI prediction persistence; dependency-free except stdlib."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.migrate_dashboard_schema import (
    ensure_predictions_columns,
    ensure_predictions_index,
    ensure_predictions_table,
)

ROOT = Path(__file__).resolve().parents[2]


def persist_prediction(*, current_price: float, analysis_res: Dict[str, Any],
                       forecast: Dict[str, Any], confidence: Dict[str, Any],
                       timeframe: str = "1D") -> int:
    """Persist BMRI rule signal; missing TFT quantiles remain NULL."""
    db_path = Path(os.environ.get("PREDICTIONS_DB", ROOT / "data" / "predictions.db"))
    conn = sqlite3.connect(db_path)
    try:
        ensure_predictions_table(conn)
        ensure_predictions_columns(conn)
        ensure_predictions_index(conn)
        fusion = analysis_res.get("fusion")
        ai_data = getattr(fusion, "ai_forecast_data", None) if fusion else None
        quantiles = (ai_data or {}).get("quantiles") or {}
        horizons = (ai_data or {}).get("horizons") or {}

        def q(horizon: str, key: str) -> Optional[float]:
            source = horizons.get(horizon) or (quantiles if horizon == "overall" else {})
            value = source.get(key)
            return None if value is None else float(current_price) * (1.0 + float(value))

        zigzag = analysis_res.get("zigzag")
        macro = zigzag.macro if zigzag else []
        micro = zigzag.micro if zigzag else []
        degree = macro[-1].degree.value if macro and hasattr(macro[-1].degree, "value") else None
        direction = analysis_res.get("direction") or {"BUY": "bullish", "SELL": "bearish"}.get(forecast.get("signal"), "neutral")
        record = {
            "asset": "BMRI.JK", "timeframe": timeframe, "direction": direction, "wave_degree": degree,
            "btc_close_at_signal": float(current_price),
            "cluster_valid": int(forecast.get("cluster_upper") is not None),
            "cluster_upper": forecast.get("cluster_upper"), "cluster_lower": forecast.get("cluster_lower"),
            "cluster_strength_adj": confidence.get("overall"), "invalidation_level": forecast.get("invalidation"),
            "macro_pivot_count": len(macro), "micro_pivot_count": len(micro),
            "q10_7d": q("5d", "q10"), "q50_7d": q("5d", "q50"), "q90_7d": q("5d", "q90"),
            "q10_14d": q("10d", "q10"), "q50_14d": q("10d", "q50"), "q90_14d": q("10d", "q90"),
            "q10_30d": q("20d", "q10"), "q50_30d": q("20d", "q50"), "q90_30d": q("20d", "q90"),
        }
        cols = ", ".join(record)
        marks = ", ".join("?" for _ in record)
        cur = conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({marks})", list(record.values()))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()
