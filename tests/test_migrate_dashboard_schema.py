"""
test_migrate_dashboard_schema.py
--------------------------------
Guards the Phase 1 data-layer migration (scripts/migrate_dashboard_schema.py):

1. Fresh DB bootstrap creates all registry tables + full shared predictions schema.
2. Legacy DB (no asset column, real rows) gets asset added + backfilled 'BTC'.
3. idx_predictions_asset_tf exists after migration — on BOTH fresh and legacy paths.
4. Running the migration twice does not duplicate rows (idempotent).
5. Asset timeframes seed trained=1 iff a checkpoint file actually exists on disk.

Only stdlib (sqlite3) + pytest — no torch/pandas needed.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import migrate_dashboard_schema as mig  # noqa: E402

TABLES = ["predictions", "assets", "asset_timeframes", "jobs", "recipes"]


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def _index_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,))]


def _seed_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    return (
        conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM asset_timeframes").fetchone()[0],
    )


def test_fresh_bootstrap_creates_all_tables(tmp_path):
    db = tmp_path / "fresh.db"
    mig.migrate(db)

    conn = sqlite3.connect(db)
    try:
        tables = _table_names(conn)
        assert set(TABLES) <= tables, f"missing tables: {set(TABLES) - tables}"
        assert "sqlite_sequence" in tables  # AUTOINCREMENT side-effect

        cols = {c[1] for c in conn.execute("PRAGMA table_info(predictions)")}
        assert "asset" in cols
        for ob in ("ob_conviction", "ob_bid_ask_imbalance",
                   "ob_dominant_exchange", "ob_flag"):
            assert ob in cols, f"fresh bootstrap missing {ob}"
    finally:
        conn.close()


def test_fresh_bootstrap_has_index(tmp_path):
    db = tmp_path / "fresh.db"
    mig.migrate(db)

    conn = sqlite3.connect(db)
    try:
        assert "idx_predictions_asset_tf" in _index_names(conn, "predictions")
    finally:
        conn.close()


def test_legacy_backfill_asset_btc(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        timeframe TEXT, direction TEXT, btc_close_at_signal REAL)""")
    conn.execute("INSERT INTO predictions (timeframe, direction, btc_close_at_signal)"
                 " VALUES ('1D', 'long', 65000)")
    conn.commit()
    conn.close()

    mig.migrate(db)

    conn = sqlite3.connect(db)
    try:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(predictions)")]
        assert "asset" in cols
        rows = conn.execute("SELECT id, asset, timeframe FROM predictions").fetchall()
        assert rows == [(1, "BTC", "1D")]
        assert "idx_predictions_asset_tf" in _index_names(conn, "predictions")
    finally:
        conn.close()


def test_legacy_wave_degree_backfilled_by_timeframe(tmp_path):
    db = tmp_path / "legacy_degree.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        timeframe TEXT, direction TEXT, btc_close_at_signal REAL)""")
    conn.execute("INSERT INTO predictions (timeframe, direction) VALUES ('1D', 'long')")
    conn.execute("INSERT INTO predictions (timeframe, direction) VALUES ('4H', 'long')")
    conn.execute("INSERT INTO predictions (timeframe, direction) VALUES ('1W', 'long')")
    conn.commit()
    conn.close()

    mig.migrate(db)

    conn = sqlite3.connect(db)
    try:
        cols = {c[1] for c in conn.execute("PRAGMA table_info(predictions)")}
        assert "wave_degree" in cols
        by_tf = dict(conn.execute(
            "SELECT timeframe, wave_degree FROM predictions ORDER BY id").fetchall())
        assert by_tf == {"1D": "intermediate", "4H": "minute", "1W": "primary"}
    finally:
        conn.close()


def test_double_migration_is_idempotent(tmp_path):
    db = tmp_path / "twice.db"
    mig.migrate(db)
    first = _seed_counts(sqlite3.connect(db))
    sqlite3.connect(db).close()

    mig.migrate(db)
    second = _seed_counts(sqlite3.connect(db))
    sqlite3.connect(db).close()

    assert first == second, f"seeded rows drifted: {first} -> {second}"


def test_no_checkpoint_seeds_trained_0(tmp_path, monkeypatch):
    # Empty tmp ROOT -> neither wave_model.pt nor BMRI_JK.ckpt exists.
    monkeypatch.setattr(mig, "ROOT", tmp_path)
    db = tmp_path / "nockpt.db"
    mig.migrate(db)

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT a.symbol, t.trained, a.checkpoint_path FROM asset_timeframes t"
            " JOIN assets a ON a.id=t.asset_id").fetchall()
        assert rows, "expected seeded asset_timeframes"
        for symbol, trained, ckpt in rows:
            assert ckpt is None, f"{symbol} seeded checkpoint {ckpt!r} with no file on disk"
            assert trained == 0, f"{symbol} claimed trained=1 with no checkpoint"
    finally:
        conn.close()


def test_checkpoint_exists_seeds_trained_1(tmp_path, monkeypatch):
    # Fake trained artifacts on disk -> assets seed trained=1 with paths.
    models = tmp_path / "models"
    ckpts = tmp_path / "src" / "models" / "checkpoints"
    models.mkdir(parents=True)
    ckpts.mkdir(parents=True)
    (models / "wave_model.pt").write_bytes(b"fake")
    (ckpts / "BMRI_JK.ckpt").write_bytes(b"fake")

    monkeypatch.setattr(mig, "ROOT", tmp_path)
    db = tmp_path / "ckpt.db"
    mig.migrate(db)

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT a.symbol, a.checkpoint_path, t.trained"
            " FROM asset_timeframes t JOIN assets a ON a.id=t.asset_id"
            " GROUP BY a.symbol").fetchall()
        by_symbol = {r[0]: r for r in rows}
        assert by_symbol["BTC"][1] == "models/wave_model.pt"
        assert by_symbol["BTC"][2] == 1
        assert by_symbol["BMRI.JK"][1] == "src/models/checkpoints/BMRI_JK.ckpt"
        assert by_symbol["BMRI.JK"][2] == 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
