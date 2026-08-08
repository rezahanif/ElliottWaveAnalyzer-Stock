"""Register/finish systemd pipeline jobs in dashboard SQLite."""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--asset", default="BTC")
    p.add_argument("--timeframe", default="1D")
    p.add_argument("--finish", choices=("done", "failed"))
    p.add_argument("--job-id", type=int)
    args = p.parse_args()
    conn = sqlite3.connect(args.db, timeout=5)
    try:
        if args.finish:
            if not args.job_id:
                raise SystemExit("--job-id required with --finish")
            conn.execute(
                "UPDATE jobs SET status=?, finished_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND status='running'", (args.finish, args.job_id)
            )
            conn.commit()
            return 0
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT a.id FROM assets a WHERE a.symbol=?", (args.asset,)
        ).fetchone()
        if not row:
            raise SystemExit(f"unknown asset: {args.asset}")
        asset_id = row[0]
        active = conn.execute(
            "SELECT id FROM jobs WHERE asset_id=? AND timeframe=? "
            "AND status IN ('queued','running')", (asset_id, args.timeframe)
        ).fetchone()
        if active:
            conn.rollback()
            return 2
        cur = conn.execute(
            "INSERT INTO jobs(asset_id,timeframe,action,status,started_at) "
            "VALUES(?,?, 'cron', 'running', CURRENT_TIMESTAMP)",
            (asset_id, args.timeframe),
        )
        conn.commit()
        print(cur.lastrowid)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
