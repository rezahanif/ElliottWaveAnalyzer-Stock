"""
bot.py
------
Lightweight, polling-based Telegram command listener.
Responds to registered commands like /bmri, /report, /weekly, /status, /help.
Limits access to authorized users only.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.request
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger("telegram_bot")


class TelegramBot:
    """Lightweight bot handler using getUpdates polling."""

    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.authorized_users = self._load_authorized_users()
        self.commands: Dict[str, Callable[[str, List[str]], str]] = {}
        self.offset = 0

    def _load_authorized_users(self) -> List[str]:
        # Load from environment variables (comma-separated list of IDs)
        raw = os.environ.get("TELEGRAM_CHAT_IDS") or os.environ.get("TELEGRAM_CHAT_ID") or ""
        return [uid.strip() for uid in raw.split(",") if uid.strip()]

    def register_command(self, name: str, handler: Callable[[str, List[str]], str]):
        """Register a handler function for a command (e.g. 'bmri')."""
        self.commands[name.lower().replace("/", "")] = handler

    def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML"):
        """Send message back to chat."""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": parse_mode}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")

    def poll_updates(self):
        """Fetch new updates and process commands."""
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not configured. Bot polling disabled.")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates?timeout=10&offset={self.offset}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode())
                
            if not res_data.get("ok"):
                return

            for update in res_data.get("result", []):
                self.offset = update["update_id"] + 1
                
                message = update.get("message")
                if not message:
                    continue
                    
                chat = message.get("chat", {})
                chat_id = str(chat.get("id", ""))
                from_user = message.get("from", {})
                user_id = str(from_user.get("id", ""))
                
                # Security Check: Authorization limit
                if self.authorized_users and chat_id not in self.authorized_users and user_id not in self.authorized_users:
                    logger.warning(f"Unauthorized access attempt from Chat ID: {chat_id}, User ID: {user_id}")
                    # Send polite refusal
                    self.send_message(chat_id, "❌ <b>Access Denied</b>. You are not authorized.")
                    continue
                    
                text = message.get("text", "").strip()
                if text.startswith("/"):
                    parts = text.split()
                    cmd = parts[0][1:].lower().split("@")[0]  # Remove leading slash and bot username suffix if any
                    args = parts[1:]
                    
                    if cmd in self.commands:
                        logger.info(f"Executing command /{cmd} for chat {chat_id}")
                        try:
                            response_text = self.commands[cmd](chat_id, args)
                            if response_text:
                                self.send_message(chat_id, response_text)
                        except Exception as e:
                            logger.error(f"Error handling /{cmd}: {e}")
                            self.send_message(chat_id, f"❌ Error executing /{cmd}: {e}")
                    else:
                        # Command not registered or default help
                        if cmd == "help":
                            help_msg = (
                                "🤖 <b>Available Commands:</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"{chr(10).join([f'• /{c}' for c in self.commands.keys()])}\n"
                            )
                            self.send_message(chat_id, help_msg)
        except Exception as e:
            logger.error(f"Error polling Telegram updates: {e}")
            time.sleep(2)  # brief backoff on connection error
