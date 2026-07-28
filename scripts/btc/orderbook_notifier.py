"""
notifier.py
-----------
Standalone 5-minute order-book poll job for the BTC pipeline.

Cadence: every 5 minutes via systemd elliott-orderbook.timer
          (modeled on elliott-notifier.timer / economic_notifier.py)

Per-cycle responsibilities:
    1. Fetch L2 order-book snapshots from Binance, Coinbase, OKX
       (via fetch_multi_exchange_orderbook).
    2. Persist snapshot to data/orderbook/BTC_snapshot_{ts}.json
       (via snapshot.write_snapshot, atomic).
    3. Detect NEW walls >= wall_threshold_usd (default $1M) — both bid
       (buy) and ask (sell) sides.
    4. For each new wall, send a Telegram alert. Walls are deduplicated
       via the orderbook_alerts SQLite table for dedup_window_minutes
       (default 30) so the same wall doesn't trigger repeated alerts
       every 5 minutes.
    5. Optionally, send a summary alert every summary_interval_minutes
       (default 60) even if no new walls appeared.
    6. Cleanup old snapshot files (storage.cleanup_days, default 7).

This is a sibling of scripts/btc/economic_notifier.py — same pattern:
SQLite dedup table, logging to data/orderbook_notifier.log, CLI flags
--dry-run / --force-summary.

Telegram alert format (per new wall):
    🧱 <b>BTC ORDER BOOK WALL DETECTED</b>
    <code>2026-07-26 14:30:05 UTC</code>
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🟦 <b>BID WALL (BUY)</b> on Binance
    💵 Notional: <b>$5,234,567</b> (~83.4 BTC)
    📍 Price: <b>$62,750.00</b>  (−0.50% from spot $63,065.40)
    📊 Cross-exchange context: 2/3 exchanges show BID dominance
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    <i>Wall above $1M threshold. Confirm before acting — single
    exchange walls may be pulled. Watch for cross-exchange
    confirmation.</i>
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── Project root on path ──────────────────────────────────────
# This file is at scripts/btc/orderbook_notifier.py — 3 parents up = project root
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── .env auto-loader (matches run_daily_analysis.py pattern) ─
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

# ── Project imports ──────────────────────────────────────────
from src.btc.orderbook.fetcher import fetch_multi_exchange_orderbook, OrderBookSnapshot
from src.btc.orderbook.scorer import OrderBookConvictionScorer, WallReport, ConvictionReport
from src.btc.orderbook.snapshot import write_snapshot, load_latest_snapshot, cleanup_old_snapshots
from src.shared.telegram.client import send_telegram

# ── Config ────────────────────────────────────────────────────
DB_PATH = "data/predictions.db"
LOG_PATH = ROOT / "data" / "orderbook_notifier.log"
CONFIG_PATH = "config/orderbook.yaml"

# ── Logging setup (matches economic_notifier.py) ─────────────
os.makedirs(ROOT / "data", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH),
    ],
)
logger = logging.getLogger("orderbook_notifier")


# ─────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────

def _load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    full = ROOT / config_path
    if not full.exists():
        raise FileNotFoundError(f"Config not found: {full}")
    with open(full, "r") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────
# SQLite dedup table
# ─────────────────────────────────────────────────────────────

def init_alert_db(db_path: str = DB_PATH):
    """Initialize orderbook_alerts table for wall-alert deduplication."""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wall_signature TEXT NOT NULL,
                exchange TEXT,
                side TEXT,
                price REAL,
                usd_value REAL,
                first_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_alerted_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logger.info("orderbook_alerts table initialized.")
    except Exception as e:
        logger.error(f"Error initializing orderbook_alerts table: {e}")
    finally:
        conn.close()


def _wall_signature(exchange: str, side: str, price: float) -> str:
    """
    Create a dedup signature for a wall. Round price to nearest 0.5% bucket
    so the same wall that drifts slightly across snapshots still dedupes.
    """
    bucket = round(price / (price * 0.005)) * (price * 0.005)
    return f"{exchange}:{side}:{bucket:.2f}"


def is_wall_alerted(
    signature: str,
    dedup_window_minutes: int = 30,
    db_path: str = DB_PATH,
) -> bool:
    """Check if a wall with this signature was alerted within the dedup window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=dedup_window_minutes)).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM orderbook_alerts "
            "WHERE wall_signature = ? AND last_alerted_at > ?",
            (signature, cutoff),
        )
        row = cur.fetchone()
        return bool(row and row[0] > 0)
    except Exception as e:
        logger.error(f"Error checking wall alert state: {e}")
        return False
    finally:
        conn.close()


def record_wall_alert(
    signature: str,
    exchange: str,
    side: str,
    price: float,
    usd_value: float,
    db_path: str = DB_PATH,
):
    """Insert or update a wall alert record."""
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    try:
        # Upsert — if signature exists, just bump last_alerted_at
        cur = conn.execute(
            "SELECT id FROM orderbook_alerts WHERE wall_signature = ?", (signature,)
        )
        row = cur.fetchone()
        if row:
            conn.execute(
                "UPDATE orderbook_alerts SET last_alerted_at = ?, "
                "usd_value = MAX(usd_value, ?), price = ? WHERE id = ?",
                (now, usd_value, price, row[0]),
            )
        else:
            conn.execute(
                "INSERT INTO orderbook_alerts "
                "(wall_signature, exchange, side, price, usd_value, "
                " first_seen_at, last_alerted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (signature, exchange, side, price, usd_value, now, now),
            )
        conn.commit()
    except Exception as e:
        logger.error(f"Error recording wall alert: {e}")
    finally:
        conn.close()


def last_summary_sent_at(summary_interval_minutes: int, db_path: str = DB_PATH) -> Optional[datetime]:
    """Return the timestamp of the most recent summary alert, or None."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=summary_interval_minutes)).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT last_alerted_at FROM orderbook_alerts "
            "WHERE wall_signature = '__summary__' AND last_alerted_at > ? "
            "ORDER BY last_alerted_at DESC LIMIT 1",
            (cutoff,),
        )
        row = cur.fetchone()
        if row:
            try:
                return datetime.fromisoformat(row[0])
            except Exception:
                return None
        return None
    except Exception as e:
        logger.error(f"Error checking last summary: {e}")
        return None
    finally:
        conn.close()


def record_summary_sent(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT INTO orderbook_alerts "
            "(wall_signature, exchange, side, price, usd_value, "
            " first_seen_at, last_alerted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("__summary__", "", "", 0.0, 0.0, now, now),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error recording summary: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# Telegram alert builders
# ─────────────────────────────────────────────────────────────

def build_wall_alert_message(
    wall: WallReport,
    snapshot_timestamp_ms: int,
    cross_exchange_context: str,
) -> str:
    """Format a per-wall Telegram alert."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    is_bid = wall.side == "bid"
    emoji_side = "🟦 BID WALL (BUY)" if is_bid else "🟥 ASK WALL (SELL)"
    direction_word = "below" if is_bid else "above"

    # Find a reference spot price — wall.distance_pct is computed against spot
    # of that exchange, so the price-distance is already informative.
    return (
        f"🧱 <b>BTC ORDER BOOK WALL DETECTED</b>\n"
        f"<code>{now}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji_side} on <b>{wall.exchange.upper()}</b>\n"
        f"💵 Notional: <b>${wall.usd_value:,.0f}</b> (~{wall.qty:.2f} BTC)\n"
        f"📍 Price: <b>${wall.price:,.2f}</b>  "
        f"({wall.distance_pct:+.2f}% {direction_word} spot)\n"
        f"📊 Cross-exchange: {cross_exchange_context}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Wall above $1M threshold. Confirm before acting — single "
        f"exchange walls may be pulled. Watch for cross-exchange "
        f"confirmation in subsequent snapshots.</i>"
    )


def build_summary_message(
    report: ConvictionReport,
    snapshot_timestamp_ms: int,
) -> str:
    """Periodic summary message — sent every summary_interval_minutes."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    snap_time = datetime.fromtimestamp(snapshot_timestamp_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S UTC")

    lines = [
        f"📋 <b>BTC ORDER BOOK PERIODIC SUMMARY</b>",
        f"<code>{now}</code>  (snapshot: {snap_time})",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🏦 Exchanges: <b>{report.exchanges_succeeded}/{report.exchanges_succeeded + report.exchanges_failed}</b> responded",
    ]
    if report.failed_exchanges:
        lines.append(f"   Failed: {', '.join(report.failed_exchanges)}")

    lines += [
        f"",
        f"🧱 <b>WALLS &gt;= $1M DETECTED</b>",
        f"   🟦 Bid walls: <b>{len(report.bid_walls)}</b>",
        f"   🟥 Ask walls: <b>{len(report.ask_walls)}</b>",
    ]
    if report.bid_walls:
        top = report.bid_walls[0]
        lines.append(f"   Top bid: ${top.usd_value/1e6:.2f}M @ ${top.price:,.0f} on {top.exchange}")
    if report.ask_walls:
        top = report.ask_walls[0]
        lines.append(f"   Top ask: ${top.usd_value/1e6:.2f}M @ ${top.price:,.0f} on {top.exchange}")

    lines += [
        f"",
        f"🧭 <b>CROSS-EXCHANGE SIGNAL</b>",
        f"   Dominant side: <b>{report.dominant_side or '—'}</b>",
        f"   Agreement: <b>{report.agreement_score:.2f}</b>",
        f"   Weighted imbalance: <b>{report.weighted_imbalance:+.4f}</b>",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"⏰ Next summary in 60 minutes",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def snapshot_to_dict(snaps: Dict[str, OrderBookSnapshot], report: Optional[ConvictionReport]) -> Dict[str, Any]:
    """Serialize fetcher + scorer output to a JSON-safe dict for disk storage."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exchanges": {name: snap.to_dict() for name, snap in snaps.items()},
        "conviction": report.to_dict() if report else None,
    }


def run_cycle(
    dry_run: bool = False,
    force_summary: bool = False,
    config_path: str = CONFIG_PATH,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """
    Run one notifier cycle. Returns a summary dict (for logging/testing).
    """
    cfg = _load_config(config_path)
    wall_threshold_usd = float(cfg.get("wall_threshold_usd", 1_000_000))
    alerts_cfg = cfg.get("alerts", {})
    dedup_minutes = int(alerts_cfg.get("dedup_window_minutes", 30))
    summary_minutes = int(alerts_cfg.get("summary_interval_minutes", 60))
    alerts_enabled = bool(alerts_cfg.get("enabled", True))

    # 1. Fetch snapshots
    logger.info("Fetching order-book snapshots from 3 exchanges...")
    snaps = fetch_multi_exchange_orderbook(config_path=config_path)
    success_count = sum(1 for s in snaps.values() if not s.fetch_error)
    fail_count = len(snaps) - success_count
    logger.info(f"Fetch complete: {success_count} ok, {fail_count} failed.")

    if success_count == 0:
        logger.error("All exchanges failed — skipping cycle.")
        return {"status": "all_failed"}

    # 2. Compute conviction report (use 'neutral' wave direction here —
    #    the notifier doesn't know the wave context, that's the main
    #    pipeline's job. The notifier just reports walls.)
    scorer = OrderBookConvictionScorer(config_path=config_path)
    report = scorer.compute(snaps, wave_direction="neutral", confluence_zone=None)

    # 3. Persist snapshot
    snapshot_dict = snapshot_to_dict(snaps, report)
    try:
        snap_path = write_snapshot(snapshot_dict, config_path=config_path)
        logger.info(f"Snapshot saved: {snap_path}")
    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")

    # 4. Cleanup old snapshots
    try:
        deleted = cleanup_old_snapshots(config_path=config_path)
        if deleted > 0:
            logger.info(f"Cleanup: deleted {deleted} old snapshots.")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

    # 5. Detect new walls + alert
    new_walls_alerted = 0
    if alerts_enabled:
        if not dry_run:
            init_alert_db(db_path)

        all_walls: List[WallReport] = list(report.bid_walls) + list(report.ask_walls)
        # Sort by USD descending — alert largest first
        all_walls.sort(key=lambda w: w.usd_value, reverse=True)

        for wall in all_walls:
            sig = _wall_signature(wall.exchange, wall.side, wall.price)
            if not dry_run and is_wall_alerted(sig, dedup_window_minutes, db_path):
                logger.info(f"Wall already alerted recently, skipping: {sig}")
                continue

            # Cross-exchange context line
            n_bid_exchanges = sum(
                1 for s in report.per_exchange
                if not s.fetch_error and s.dominant_wall_side == "bid"
            )
            n_ask_exchanges = sum(
                1 for s in report.per_exchange
                if not s.fetch_error and s.dominant_wall_side == "ask"
            )
            ctx = (
                f"{n_bid_exchanges}/3 show BID dominance, "
                f"{n_ask_exchanges}/3 show ASK dominance"
            )

            msg = build_wall_alert_message(wall, report.per_exchange[0].spot_price if report.per_exchange else 0, ctx)
            logger.info(f"Alerting new wall: {sig} (${wall.usd_value:,.0f})")
            send_telegram(msg, dry_run=dry_run, label="orderbook")

            if not dry_run:
                record_wall_alert(sig, wall.exchange, wall.side, wall.price, wall.usd_value, db_path)
            new_walls_alerted += 1

        # 6. Periodic summary
        if force_summary or last_summary_sent_at(summary_minutes, db_path) is None:
            logger.info("Sending periodic summary.")
            summary_msg = build_summary_message(report, snapshot_dict.get("timestamp_utc") and int(datetime.now(timezone.utc).timestamp() * 1000) or 0)
            # Use current ts for snapshot_timestamp_ms since snapshot_dict doesn't carry one cleanly
            send_telegram(summary_msg, dry_run=dry_run, label="orderbook-summary")
            if not dry_run:
                record_summary_sent(db_path)
    else:
        logger.info("Alerts disabled in config — skipping alert phase.")

    return {
        "status": "ok",
        "exchanges_ok": success_count,
        "exchanges_failed": fail_count,
        "bid_walls": len(report.bid_walls),
        "ask_walls": len(report.ask_walls),
        "new_walls_alerted": new_walls_alerted,
        "multiplier": report.multiplier,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone 5-min order-book poll + wall alert notifier."
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Don't write to SQLite or send Telegram alerts — just print.",
    )
    parser.add_argument(
        "--force-summary",
        action="store_true",
        help="Force-send the periodic summary even if not yet due.",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help=f"Path to orderbook config (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    logger.info("=" * 60)
    logger.info("Order-book notifier cycle starting")
    logger.info(f"  dry_run={args.dry_run}  force_summary={args.force_summary}")
    logger.info("=" * 60)

    try:
        result = run_cycle(
            dry_run=args.dry_run,
            force_summary=args.force_summary,
            config_path=args.config,
            db_path=args.db,
        )
        logger.info(f"Cycle result: {result}")
    except Exception as e:
        logger.exception(f"Cycle failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
