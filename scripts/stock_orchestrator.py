#!/usr/bin/env python3
"""
stock_orchestrator.py
---------------------
Main orchestrator for Stock Swing Trading (BMRI).
Fetches data, evaluates market cascade + fundamentals + news, and triggers
Telegram notifications at scheduled IDX market events.
Exits silently on IDX holidays and weekends.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, date
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stock.collectors.idx_holidays import is_idx_trading_day
from src.stock.telegram.handlers import handle_bmri, handle_report
from src.shared.telegram.client import send_telegram

# Load .env file if it exists to populate environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass


def run_closing_analysis(dry_run: bool = False):
    """Run full BMRI closing analysis and send alert."""
    print("Running BMRI closing analysis...")
    msg = handle_bmri("", [])
    send_telegram(msg, dry_run=dry_run, label="stock-alert")


def run_morning_update(dry_run: bool = False):
    """Run morning market briefing."""
    print("Running BMRI morning context update...")
    # Send a quick fundamental & news snapshot before market opens
    msg = handle_report("", [])
    header = (
        "🌅 <b>BMRI MORNING MARKET BRIEFING</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    send_telegram(header + msg, dry_run=dry_run, label="stock-alert")


def run_weekly_review(dry_run: bool = False):
    """Run weekly strategy review on Saturday."""
    print("Running BMRI weekly review...")
    # Summary of news, fundamentals, and general technical levels
    msg = handle_report("", [])
    header = (
        "📅 <b>BMRI WEEKLY STRATEGY REVIEW</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    send_telegram(header + msg, dry_run=dry_run, label="stock-alert")


def main():
    parser = argparse.ArgumentParser(description="BMRI Stock Swing Orchestrator")
    parser.add_argument(
        "--action",
        choices=["morning", "midday", "closing", "weekly"],
        default="closing",
        help="Scheduled action to run (morning, midday, closing, weekly)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending alerts to Telegram",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even on holidays/weekends",
    )
    args = parser.parse_args()

    today = date.today()

    # 1. Day Check: Skip execution on holidays/weekends unless forced or running weekly review
    if args.action != "weekly" and not args.force:
        if not is_idx_trading_day(today):
            print(f"Skipping execution: {today} is not an active IDX trading day.")
            return

    # 2. Route Action
    try:
        if args.action == "morning":
            run_morning_update(dry_run=args.dry_run)
        elif args.action == "midday":
            # Midday is a silent run or light update, default to quiet context check
            print("Midday context run completed.")
        elif args.action == "closing":
            run_closing_analysis(dry_run=args.dry_run)
        elif args.action == "weekly":
            run_weekly_review(dry_run=args.dry_run)
        print("Orchestration complete ✅")
    except Exception as e:
        print(f"❌ Orchestration failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
