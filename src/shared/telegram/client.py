"""
Shared Telegram alert client.
Single implementation used by all pipelines (BTC, stock, astro, economic).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Optional


def send_telegram(
    message: str,
    *,
    dry_run: bool = False,
    parse_mode: str = "HTML",
    bot_token: Optional[str] = None,
    chat_ids: Optional[list[str]] = None,
    label: str = "telegram",
) -> None:
    """
    Send a Telegram message to all configured chat IDs.

    Resolution order for bot_token / chat_ids:
      1. Explicit arguments
      2. Environment variables TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS (comma-sep) or TELEGRAM_CHAT_ID
    """
    if dry_run:
        print(f"\n  [{label}] DRY RUN — message that would be sent:")
        print("  " + "\n  ".join(message.split("\n")))
        return

    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if chat_ids is None:
        raw = os.environ.get("TELEGRAM_CHAT_IDS") or os.environ.get("TELEGRAM_CHAT_ID")
        chat_ids = [cid.strip() for cid in raw.split(",") if cid.strip()] if raw else []

    if not bot_token or not chat_ids:
        print(f"  [{label}] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_IDS not set — skipping", file=sys.stderr)
        return

    for chat_id in chat_ids:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": parse_mode}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            print(f"  [{label}] Alert sent to {chat_id} ✅")
        except Exception as e:
            print(f"  [{label}] Failed to send to {chat_id}: {e}", file=sys.stderr)
