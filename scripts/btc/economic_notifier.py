"""
economic_notifier.py
---------------------
Standalone economic calendar notification engine.
Parses Investing.com API and manages pre-event alerts, post-release alerts,
daily reminders, and weekly reminders.
Uses SQLite for state tracking to prevent duplicate notifications.
All timestamps are displayed in local server time (UTC+7 timezone / ICT).
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env file
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

from src.btc.ingestion.investing_api import (
    get_economic_calendar,
    is_high_risk,
    get_event_priority,
    parse_utc_to_local,
    HIGH_RISK_EVENTS
)

# Configuration
DB_PATH = "data/predictions.db"
LOCAL_TZ = timezone(timedelta(hours=7))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "data" / "economic_notifier.log")
    ]
)
logger = logging.getLogger("economic_notifier")

# USD high-impact releases are useful; GBP/EUR/CNY only send critical named events.
TRACKED_CURRENCIES = {"USD", "GBP", "EUR", "CNY"}

def is_notifiable_event(event: dict) -> bool:
    """USD permits all high-impact releases; other tracked currencies need critical status."""
    currency = event.get("currency", "").upper()
    return currency in TRACKED_CURRENCIES and (
        currency == "USD" or is_high_risk(event.get("event_name", ""))
    )

CURRENCY_FLAGS = {
    "USD": "🇺🇸 USD",
    "EUR": "🇪🇺 EUR",
    "GBP": "🇬🇧 GBP",
    "JPY": "🇯🇵 JPY",
    "AUD": "🇦🇺 AUD",
    "CAD": "🇨🇦 CAD",
    "CHF": "🇨🇭 CHF",
    "NZD": "🇳🇿 NZD",
    "CNY": "🇨🇳 CNY"
}

def get_flag(currency: str) -> str:
    return CURRENCY_FLAGS.get(currency.upper(), f"🏳️ {currency}")

# ─────────────────────────────────────────────────────────────
# State Tracking via SQLite
# ─────────────────────────────────────────────────────────────

def init_alert_db():
    """Initialize the economic_alerts table in predictions.db if not exists."""
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS economic_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                occurrence_id INTEGER,
                occurrence_time TEXT,
                alert_type TEXT,
                sent_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logger.info("Alert database table economic_alerts initialized.")
    except Exception as e:
        logger.error(f"Error initializing alert database table: {e}")
    finally:
        conn.close()

def is_alert_sent(occurrence_id: int | None, alert_type: str, date_str: str | None = None) -> bool:
    """Check if an alert of the specified type has already been sent for this occurrence or date."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        if alert_type in ('daily_reminder', 'weekly_reminder') and date_str:
            # For daily/weekly reminders, date_str contains the unique identifier (date or week string)
            # stored in occurrence_time column
            cursor.execute(
                "SELECT COUNT(*) FROM economic_alerts WHERE alert_type = ? AND occurrence_time = ?",
                (alert_type, date_str)
            )
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM economic_alerts WHERE occurrence_id = ? AND alert_type = ?",
                (occurrence_id, alert_type)
            )
        res = cursor.fetchone()
        return res[0] > 0 if res else False
    except Exception as e:
        logger.error(f"Error checking alert state: {e}")
        return False
    finally:
        conn.close()

def record_alert(occurrence_id: int | None, event_id: int | None, occurrence_time: str | None, alert_type: str):
    """Record an alert in the database to prevent duplicates."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO economic_alerts (event_id, occurrence_id, occurrence_time, alert_type) VALUES (?, ?, ?, ?)",
            (event_id, occurrence_id, occurrence_time, alert_type)
        )
        conn.commit()
        logger.info(f"Recorded alert of type '{alert_type}' for occurrence_id: {occurrence_id}, event_id: {event_id}")
    except Exception as e:
        logger.error(f"Error recording alert: {e}")
    finally:
        conn.close()

# ─────────────────────────────────────────────────────────────
# Telegram Alert Broadcaster
# ─────────────────────────────────────────────────────────────

def send_telegram(message: str, dry_run: bool = False):
    """Send Telegram message to all configured chat IDs."""
    if dry_run:
        print("\n[telegram] DRY RUN — Message content:")
        print(message)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids_str = os.environ.get("TELEGRAM_CHAT_IDS") or os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_ids_str:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_IDS not configured. Alert printed to console.")
        print(message)
        return

    chat_ids = [cid.strip() for cid in chat_ids_str.split(",") if cid.strip()]

    import urllib.request
    for chat_id in chat_ids:
        try:
            url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
            req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            logger.info(f"Telegram notification sent successfully to {chat_id} ✅")
        except Exception as e:
            logger.error(f"Failed to send Telegram message to {chat_id}: {e}")


# ─────────────────────────────────────────────────────────────
# Notification Engines
# ─────────────────────────────────────────────────────────────

def run_pre_event_checks(events: list[dict], dry_run: bool = False):
    """Check and send pre-event alerts 24h, 1h, and 15m before release."""
    now_utc = datetime.now(timezone.utc)
    
    for event in events:
        if not event.get("occurrence_time"):
            continue
        if not is_notifiable_event(event):
            continue
        if not is_high_risk(event["event_name"]):
            continue
            
        occ_id = event["occurrence_id"]
        evt_id = event["event_id"]
        occ_time_str = event["occurrence_time"]
        
        # Parse occurrence time (UTC)
        try:
            occ_time_utc = datetime.fromisoformat(occ_time_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception as ex:
            logger.error(f"Error parsing date {occ_time_str}: {ex}")
            continue
            
        time_diff = occ_time_utc - now_utc
        local_dt = parse_utc_to_local(occ_time_str)
        local_time_formatted = local_dt.strftime("%Y-%m-%d %H:%M (UTC+7)")
        
        # 1. 24 Hours Alert
        if timedelta(hours=0) <= time_diff <= timedelta(hours=24):
            if not is_alert_sent(occ_id, "24h"):
                msg = (
                    f"🔔 <b>24-HOUR MACRO RISK ALERT</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔴 <b>CRITICAL EVENT AHEAD</b>\n"
                    f"📅 <b>Event</b>: {event['event_name']}\n"
                    f"🏳️ <b>Currency</b>: {get_flag(event['currency'])}\n"
                    f"⏰ <b>Scheduled Time</b>: <code>{local_time_formatted}</code>\n"
                    f"📊 <b>Forecast</b>: {event.get('forecast') or 'N/A'} | <b>Previous</b>: {event.get('previous') or 'N/A'}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚠️ <i>Major macroeconomic release scheduled in 24 hours. Exercise extreme caution.</i>"
                )
                send_telegram(msg, dry_run=dry_run)
                if not dry_run:
                    record_alert(occ_id, evt_id, occ_time_str, "24h")

        # 2. 1 Hour Alert
        if timedelta(hours=0) <= time_diff <= timedelta(hours=1):
            if not is_alert_sent(occ_id, "1h"):
                msg = (
                    f"🔔 <b>1-HOUR MACRO RISK ALERT</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔴 <b>CRITICAL EVENT AHEAD</b>\n"
                    f"📅 <b>Event</b>: {event['event_name']}\n"
                    f"🏳️ <b>Currency</b>: {get_flag(event['currency'])}\n"
                    f"⏰ <b>Scheduled Time</b>: <code>{local_time_formatted}</code>\n"
                    f"📊 <b>Forecast</b>: {event.get('forecast') or 'N/A'} | <b>Previous</b>: {event.get('previous') or 'N/A'}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚠️ <i>Major macroeconomic release scheduled in 1 hour. Volatility expected.</i>"
                )
                send_telegram(msg, dry_run=dry_run)
                if not dry_run:
                    record_alert(occ_id, evt_id, occ_time_str, "1h")

        # 3. 15 Minutes Alert
        if timedelta(hours=0) <= time_diff <= timedelta(minutes=15):
            if not is_alert_sent(occ_id, "15m"):
                msg = (
                    f"🚨 <b>IMMINENT MACRO RISK ALERT (15 MINS)</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔴 <b>CRITICAL EVENT AHEAD</b>\n"
                    f"📅 <b>Event</b>: {event['event_name']}\n"
                    f"🏳️ <b>Currency</b>: {get_flag(event['currency'])}\n"
                    f"⏰ <b>Scheduled Time</b>: <code>{local_time_formatted}</code>\n"
                    f"📊 <b>Forecast</b>: {event.get('forecast') or 'N/A'} | <b>Previous</b>: {event.get('previous') or 'N/A'}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚠️ <i>Release in 15 minutes. Expect immediate market impact on Elliott Wave setups.</i>"
                )
                send_telegram(msg, dry_run=dry_run)
                if not dry_run:
                    record_alert(occ_id, evt_id, occ_time_str, "15m")

def run_post_release_checks(events: list[dict], dry_run: bool = False):
    """Check for recent releases and send post-release surprise comparisons."""
    now_utc = datetime.now(timezone.utc)
    
    for event in events:
        if not event.get("occurrence_time"):
            continue
        if not is_notifiable_event(event):
            continue
        if not is_high_risk(event["event_name"]):
            continue
            
        occ_id = event["occurrence_id"]
        evt_id = event["event_id"]
        occ_time_str = event["occurrence_time"]
        actual = event.get("actual")
        
        # We check events released in the last 2 hours that have a populated 'actual' value
        try:
            occ_time_utc = datetime.fromisoformat(occ_time_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception as ex:
            logger.error(f"Error parsing date {occ_time_str}: {ex}")
            continue
            
        time_diff = now_utc - occ_time_utc
        
        if timedelta(hours=0) <= time_diff <= timedelta(hours=2) and actual is not None and actual != "":
            if not is_alert_sent(occ_id, "post_release"):
                surprise = event.get("actual_to_forecast") or "neutral"
                msg = (
                    f"📊 <b>{get_flag(event['currency'])} {event['event_name']} released.</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Actual: <b>{actual}</b>\n"
                    f"Forecast: <b>{event.get('forecast') or 'N/A'}</b>\n"
                    f"Previous: <b>{event.get('previous') or 'N/A'}</b>\n"
                    f"Surprise: <b>{surprise.capitalize()}</b>"
                )
                send_telegram(msg, dry_run=dry_run)
                if not dry_run:
                    record_alert(occ_id, evt_id, occ_time_str, "post_release")

def send_daily_reminder(dry_run: bool = False):
    """Generate and send a grouped, chronological daily economic calendar reminder (UTC+7)."""
    now_local = datetime.now(LOCAL_TZ)
    date_str = now_local.strftime("%Y-%m-%d")
    
    # Check if daily reminder was already sent for this calendar day
    if is_alert_sent(None, "daily_reminder", date_str):
        logger.info(f"Daily reminder already sent for {date_str}. Skipping.")
        return
        
    logger.info(f"Generating daily reminder for local date {date_str}...")
    
    # Query for the entire calendar day (from 00:00 to 23:59:59 in UTC+7)
    start_local = datetime(now_local.year, now_local.month, now_local.day, 0, 0, 0, tzinfo=LOCAL_TZ)
    end_local = datetime(now_local.year, now_local.month, now_local.day, 23, 59, 59, tzinfo=LOCAL_TZ)
    
    try:
        events = get_economic_calendar(start_local, end_local, "high")
    except Exception as e:
        logger.error(f"Failed to fetch economic calendar for daily reminder: {e}")
        return
        
    # Filter occurrences falling inside this local day, sort chronologically
    day_events = []
    for event in events:
        if not event.get("occurrence_time"):
            continue
        if not is_notifiable_event(event):
            continue
        local_dt = parse_utc_to_local(event["occurrence_time"])
        if start_local <= local_dt <= end_local:
            day_events.append((local_dt, event))
            
    # Sort chronologically by time
    day_events.sort(key=lambda x: x[0])
    
    # Group by currency
    by_currency = {}
    for local_dt, event in day_events:
        curr = event["currency"]
        if curr not in by_currency:
            by_currency[curr] = []
        by_currency[curr].append((local_dt, event))
        
    # Construct message
    lines = [
        f"📅 <b>DAILY ECONOMIC CALENDAR</b>",
        f"<code>{now_local.strftime('%A, %b %d, %Y')} (UTC+7)</code>",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]
    
    if not by_currency:
        lines.append("No high-importance economic events scheduled for today.")
    else:
        # Sort currencies alphabetically
        for curr in sorted(by_currency.keys()):
            lines.append(f"\n{get_flag(curr)}")
            for local_dt, event in by_currency[curr]:
                time_str = local_dt.strftime("%a %H:%M")
                risk_indicator = " [🔴 CRITICAL]" if is_high_risk(event["event_name"]) else " [🟠 HIGH]"
                lines.append(f"  • {time_str} - {event['event_name']}{risk_indicator}")
                
    lines.append("\n⏰ <i>All times shown in server time (UTC+7).</i>")
    
    send_telegram("\n".join(lines), dry_run=dry_run)
    if not dry_run:
        record_alert(None, None, date_str, "daily_reminder")

def send_weekly_reminder(dry_run: bool = False):
    """Scan upcoming 7 days for HIGH_RISK_EVENTS and send a weekly risk summary."""
    now_local = datetime.now(LOCAL_TZ)
    
    # Only run weekly reminder on Mondays (weekday = 0), or if forced via CLI
    # We identify the week using Year + Week Number
    year_week_str = now_local.strftime("%Y-W%W")
    
    if is_alert_sent(None, "weekly_reminder", year_week_str):
        logger.info(f"Weekly reminder already sent for week {year_week_str}. Skipping.")
        return
        
    logger.info(f"Generating weekly risk reminder for week {year_week_str}...")
    
    # Scan from now up to 7 days ahead
    start_time = now_local
    end_time = now_local + timedelta(days=7)
    
    try:
        events = get_economic_calendar(start_time, end_time, "high")
    except Exception as e:
        logger.error(f"Failed to fetch economic calendar for weekly reminder: {e}")
        return
        
    # Find matching HIGH_RISK_EVENTS
    high_risk_occurrences = []
    for event in events:
        if not event.get("occurrence_time"):
            continue
        if not is_notifiable_event(event):
            continue
        if is_high_risk(event["event_name"]):
            local_dt = parse_utc_to_local(event["occurrence_time"])
            if start_time <= local_dt <= end_time:
                high_risk_occurrences.append((local_dt, event))
                
    # Sort chronologically
    high_risk_occurrences.sort(key=lambda x: x[0])
    
    lines = [
        f"📅 <b>WEEKLY MACRO RISK FORECAST</b>",
        f"<code>Week of {now_local.strftime('%b %d, %Y')} (UTC+7)</code>",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]
    
    if not high_risk_occurrences:
        lines.append("No major macroeconomic high-risk events scheduled this week.")
    else:
        lines.append("🔥 <b>High Risk Events This Week:</b>\n")
        for local_dt, event in high_risk_occurrences:
            date_time_str = local_dt.strftime("%A, %b %d at %H:%M")
            lines.append(
                f"🔴 <b>{event['event_name']}</b>\n"
                f"   Currency: {get_flag(event['currency'])}\n"
                f"   Time: <code>{date_time_str}</code>\n"
                f"   Impact: <b>CRITICAL</b>\n"
            )
            
    lines.append("⏰ <i>Exercise caution on forecasts around these release windows.</i>")
    
    send_telegram("\n".join(lines), dry_run=dry_run)
    if not dry_run:
        record_alert(None, None, year_week_str, "weekly_reminder")

# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run economic calendar notification engine.")
    parser.add_argument("--dry-run", action="store_true", help="Print messages instead of sending them.")
    parser.add_argument("--force-daily", action="store_true", help="Force send daily reminder immediately.")
    parser.add_argument("--force-weekly", action="store_true", help="Force send weekly reminder immediately.")
    args = parser.parse_args()

    os.chdir(ROOT)
    init_alert_db()
    
    # 1. Fetch upcoming and recent calendar (covering -6 hours to +28 hours for alerts)
    now = datetime.now(timezone.utc)
    start_fetch = now - timedelta(hours=6)
    end_fetch = now + timedelta(hours=28)
    
    try:
        events = get_economic_calendar(start_fetch, end_fetch, "high")
        
        # 2. Run Pre-Event Alerts
        run_pre_event_checks(events, dry_run=args.dry_run)
        
        # 3. Run Post-Release Alerts
        run_post_release_checks(events, dry_run=args.dry_run)
        
    except Exception as e:
        logger.error(f"Error in continuous notification checks: {e}")
        
    # 4. Handle Daily Reminder
    now_local = datetime.now(LOCAL_TZ)
    # Automatically send at 00:05 UTC+7 or later if not sent yet
    if args.force_daily:
        # Clear daily reminder state to force
        date_str = now_local.strftime("%Y-%m-%d")
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM economic_alerts WHERE alert_type = 'daily_reminder' AND occurrence_time = ?", (date_str,))
        conn.commit()
        conn.close()
        send_daily_reminder(dry_run=args.dry_run)
    else:
        # Standard daily check (usually run via notifier loop or cron)
        send_daily_reminder(dry_run=args.dry_run)
        
    # 5. Handle Weekly Reminder
    # Check if Monday (weekday == 0) and local hour >= 0
    if args.force_weekly:
        year_week_str = now_local.strftime("%Y-W%W")
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM economic_alerts WHERE alert_type = 'weekly_reminder' AND occurrence_time = ?", (year_week_str,))
        conn.commit()
        conn.close()
        send_weekly_reminder(dry_run=args.dry_run)
    elif now_local.weekday() == 0:  # Monday
        send_weekly_reminder(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
