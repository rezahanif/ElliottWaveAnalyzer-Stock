"""
migrate_dashboard_schema.py
----------------------------
Idempotent schema migration for the dashboard interface (see
docs/dashboard-architecture.md). Safe to run multiple times.

What it does:
  1. Creates the `predictions` table if it doesn't exist yet (matches the
     schema written by scripts/btc/run_daily_analysis.py), so this script
     also works to bootstrap a fresh dev DB.
  2. Adds an `asset` column to `predictions` if missing, backfilling
     existing rows to 'BTC' (the only asset that table has ever held).
  3. Creates `assets`, `asset_timeframes`, `jobs`, and `recipes` tables.
  4. Seeds the registry with BTC and BMRI.JK based on what's actually in
     the repo today (models/wave_model.pt, src/models/checkpoints/*.ckpt).

Usage:
    python scripts/migrate_dashboard_schema.py [--db data/predictions.db]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "predictions.db"


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(c[1] == column for c in cols)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ─────────────────────────────────────────────────────────────
# Predictions schema — SINGLE SOURCE OF TRUTH.
# scripts/btc/run_daily_analysis.py::init_db() calls the ensure_* helpers
# below instead of owning its own CREATE TABLE, so the writer and the
# dashboard can never drift apart again (Gap 2 fix).
# ─────────────────────────────────────────────────────────────

PREDICTIONS_COLUMNS: list[tuple[str, str]] = [
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("asset", "TEXT"),
    ("timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ("timeframe", "TEXT"),
    ("direction", "TEXT"),
    ("btc_close_at_signal", "REAL"),
    ("cluster_valid", "INTEGER"),
    ("cluster_upper", "REAL"),
    ("cluster_lower", "REAL"),
    ("cluster_strength", "REAL"),
    ("cluster_strength_adj", "REAL"),
    ("target_a", "REAL"),
    ("target_b", "REAL"),
    ("scenario_a_price", "REAL"),
    ("scenario_b_price", "REAL"),
    ("invalidation_level", "REAL"),
    ("c_top", "REAL"),
    ("b_low", "REAL"),
    ("q10_7d", "REAL"), ("q50_7d", "REAL"), ("q90_7d", "REAL"),
    ("q10_14d", "REAL"), ("q50_14d", "REAL"), ("q90_14d", "REAL"),
    ("q10_30d", "REAL"), ("q50_30d", "REAL"), ("q90_30d", "REAL"),
    ("q10_60d", "REAL"), ("q50_60d", "REAL"), ("q90_60d", "REAL"),
    ("calendar_risk_flag", "TEXT"),
    ("macro_pivot_count", "INTEGER"),
    ("micro_pivot_count", "INTEGER"),
    ("actual_outcome", "TEXT"),
    ("prediction_correct", "INTEGER"),
    # Order-book conviction columns (written by run_daily_analysis.py).
    ("ob_conviction", "REAL"),          # multiplier applied (0.5 - 1.10)
    ("ob_bid_ask_imbalance", "REAL"),   # weighted avg imbalance [-1, +1]
    ("ob_dominant_exchange", "TEXT"),   # exchange with largest wall
    ("ob_flag", "TEXT"),                # human-readable summary
]


def ensure_predictions_table(conn: sqlite3.Connection) -> None:
    """Create predictions with the full shared schema if absent."""
    cols = ",\n            ".join(f"{n} {t}" for n, t in PREDICTIONS_COLUMNS)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS predictions (\n            {cols}\n        )"
    )


def ensure_predictions_columns(conn: sqlite3.Connection) -> None:
    """Add any schema columns missing from a legacy predictions table."""
    existing = {c[1] for c in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    for name, typ in PREDICTIONS_COLUMNS:
        if name == "id" or name in existing:
            continue
        conn.execute(f"ALTER TABLE predictions ADD COLUMN {name} {typ}")
        print(f"  [migrate] Added '{name}' column to predictions.")


def ensure_predictions_index(conn: sqlite3.Connection) -> None:
    """Index used by the dashboard's asset+timeframe lookups."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_asset_tf "
        "ON predictions(asset, timeframe)"
    )


def add_asset_column(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "predictions", "asset"):
        print("  [migrate] Adding `asset` column to predictions...")
        conn.execute("ALTER TABLE predictions ADD COLUMN asset TEXT")
        conn.execute("UPDATE predictions SET asset = 'BTC' WHERE asset IS NULL")
    else:
        print("  [migrate] `asset` column already present, skipping.")
    # Run unconditionally — fresh bootstraps bake `asset` into CREATE TABLE and
    # would otherwise short-circuit past index creation (Gap 1).
    ensure_predictions_index(conn)


def create_registry_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT UNIQUE NOT NULL,
            display_name    TEXT NOT NULL,
            class           TEXT NOT NULL CHECK (class IN ('crypto', 'stock')),
            currency        TEXT,
            status          TEXT NOT NULL DEFAULT 'planned'
                              CHECK (status IN ('active', 'planned')),
            checkpoint_path TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_timeframes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id     INTEGER NOT NULL REFERENCES assets(id),
            timeframe    TEXT NOT NULL,
            trained      INTEGER NOT NULL DEFAULT 0,
            script_path  TEXT,
            job_action   TEXT,
            UNIQUE(asset_id, timeframe)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id     INTEGER NOT NULL REFERENCES assets(id),
            timeframe    TEXT NOT NULL,
            action       TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'queued'
                           CHECK (status IN ('queued', 'running', 'done', 'failed')),
            log_tail     TEXT,
            started_at   DATETIME,
            finished_at  DATETIME,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recipes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id     INTEGER NOT NULL REFERENCES assets(id),
            timeframe    TEXT NOT NULL,
            name         TEXT,
            recipe_json  TEXT NOT NULL,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def seed_registry(conn: sqlite3.Connection) -> None:
    """Seed assets/asset_timeframes from what actually exists in the repo today."""
    btc_checkpoint = ROOT / "models" / "wave_model.pt"
    bmri_checkpoint = ROOT / "src" / "models" / "checkpoints" / "BMRI_JK.ckpt"

    assets = [
        {
            "symbol": "BTC",
            "display_name": "Bitcoin",
            "class": "crypto",
            "currency": "USD",
            "status": "active",
            "checkpoint_path": "models/wave_model.pt" if btc_checkpoint.exists() else None,
            "timeframes": ["1D", "4H", "1W"],  # jointly trained, see src/btc/wave_model/train.py
        },
        {
            "symbol": "BMRI.JK",
            "display_name": "Bank Mandiri (BMRI.JK)",
            "class": "stock",
            "currency": "IDR",
            "status": "active",
            "checkpoint_path": str(bmri_checkpoint.relative_to(ROOT)) if bmri_checkpoint.exists() else None,
            "timeframes": ["1D"],  # src/stock/train.py: 5/10/20-day horizons off daily bars
        },
    ]

    for a in assets:
        cur = conn.execute(
            """
            INSERT INTO assets (symbol, display_name, class, currency, status, checkpoint_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                display_name=excluded.display_name,
                class=excluded.class,
                currency=excluded.currency,
                checkpoint_path=excluded.checkpoint_path
            """,
            (a["symbol"], a["display_name"], a["class"], a["currency"], a["status"], a["checkpoint_path"]),
        )
        asset_id = conn.execute(
            "SELECT id FROM assets WHERE symbol = ?", (a["symbol"],)
        ).fetchone()[0]

        for tf in a["timeframes"]:
            script_path, job_action = _script_for(a["symbol"], tf)
            trained = 1 if a["checkpoint_path"] else 0
            conn.execute(
                """
                INSERT INTO asset_timeframes (asset_id, timeframe, trained, script_path, job_action)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(asset_id, timeframe) DO UPDATE SET
                    trained=excluded.trained,
                    script_path=excluded.script_path,
                    job_action=excluded.job_action
                """,
                (asset_id, tf, trained, script_path, job_action),
            )


def _script_for(symbol: str, timeframe: str) -> tuple[str, str]:
    """Map an asset+timeframe to the exact script/action the job runner is allowed to invoke.

    This is the allow-list source of truth for the Nitro job runner (see
    dashboard/server/server/utils/jobRunner.ts) — the API never accepts a
    free-form command, only (asset_id, timeframe) which it resolves through
    this table.
    """
    if symbol == "BTC":
        return "scripts/btc/run_daily_analysis.py", f"--timeframe={timeframe}"
    if symbol == "BMRI.JK":
        return "scripts/stock_orchestrator.py", "--run-now"
    raise ValueError(f"No known script mapping for asset {symbol}")


def migrate(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[migrate] Using DB: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_predictions_table(conn)
        add_asset_column(conn)
        ensure_predictions_columns(conn)
        create_registry_tables(conn)
        seed_registry(conn)
        conn.commit()
        print("[migrate] Done.")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate predictions.db for the dashboard interface.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to predictions.db")
    args = parser.parse_args()
    migrate(Path(args.db))


if __name__ == "__main__":
    main()
