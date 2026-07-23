"""
handlers.py
-----------
Telegram command handlers for Stock Swing Trading Platform.
Binds /bmri, /report, /weekly, /status, /help to command execution.
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import List, Dict, Any

from src.stock.analyzer import StockAnalyzer
from src.stock.collectors.price import fetch_price_data
from src.stock.collectors.ihsg import fetch_ihsg_data
from src.stock.collectors.sector import fetch_sector_data
from src.stock.collectors.fundamentals import fetch_fundamentals
from src.stock.collectors.news import fetch_news_and_sentiment
from src.stock.features.market_context import generate_market_context
from src.stock.forecast.rule_engine import evaluate_rules

logger = logging.getLogger("stock_telegram_handlers")


def handle_bmri(chat_id: str, args: List[str]) -> str:
    """Handle /bmri command: run full pipeline on the fly and report."""
    logger.info("Executing on-the-fly BMRI analysis...")
    
    # 1. Fetch data
    df_stock = fetch_price_data("BMRI.JK")
    df_sector = fetch_sector_data()
    df_ihsg = fetch_ihsg_data()
    
    if df_stock is None or df_sector is None or df_ihsg is None:
        return "❌ <b>Error:</b> Failed to fetch stock or market index price data."

    # 2. Analyze
    analyzer = StockAnalyzer("BMRI.JK", "1D")
    analysis_res = analyzer.analyze(df_stock)
    
    # 3. Market Context
    market_ctx = generate_market_context(df_stock, df_sector, df_ihsg)
    
    # 4. Fundamentals & News
    fundamentals = fetch_fundamentals("BMRI.JK")
    news_sentiment = fetch_news_and_sentiment("BMRI.JK")
    
    # 5. Evaluate Rules
    current_price = df_stock.iloc[-1]["close"]
    forecast = evaluate_rules(
        symbol="BMRI.JK",
        current_price=current_price,
        analysis_res=analysis_res,
        market_ctx=market_ctx,
        fundamentals=fundamentals,
        news_sentiment=news_sentiment,
    )
    
    # Format Response Message
    signal = forecast["signal"]
    signal_emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "WATCH": "🟡 WATCH"}.get(signal, signal)
    
    reasons_text = "\n".join([f"• {r}" for r in forecast["reasons"]])
    
    msg = (
        f"📊 <b>BMRI SWING FORECAST [1D]</b>\n"
        f"<code>{datetime.now().strftime('%Y-%m-%d %H:%M')} (UTC+7)</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Current Price: <b>IDR {current_price:,.2f}</b>\n"
        f"📐 Signal: <b>{signal_emoji}</b>\n"
        f"🛑 Invalidation: <b>IDR {forecast['invalidation'] or 0:,.2f}</b>\n"
        f"📐 Fibonacci Zone: <code>{forecast['fib_zone']}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔍 <b>Market Context Cascade:</b>\n"
        f"• IHSG Bias: <b>{market_ctx['ihsg']['bias']}</b>\n"
        f"• Sector Outperforming: <b>{market_ctx['sector']['outperforming_market']}</b>\n"
        f"• Stock Outperforming: <b>{market_ctx['stock']['outperforming_sector']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <b>Reasoning Details:</b>\n"
        f"{reasons_text}\n"
    )
    return msg


def handle_report(chat_id: str, args: List[str]) -> str:
    """Handle /report command: concise overview of key parameters."""
    fundamentals = fetch_fundamentals("BMRI.JK")
    news = fetch_news_and_sentiment("BMRI.JK")
    
    msg = (
        f"📝 <b>BMRI FUNDAMENTAL & SENTIMENT REPORT</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Fundamentals:</b>\n"
        f"• P/E Ratio: <b>{fundamentals['pe_ratio']:.1f}</b>\n"
        f"• P/B Ratio: <b>{fundamentals['pb_ratio']:.2f}</b>\n"
        f"• ROE: <b>{fundamentals['roe']*100:.1f}%</b>\n"
        f"• Div Yield: <b>{fundamentals['div_yield']*100:.1f}%</b>\n"
        f"• Revenue Growth: <b>{fundamentals['revenue_growth']*100:.1f}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📰 <b>Sentiment:</b>\n"
        f"• Score: <b>{news['sentiment_score']:.2f} ({news['sentiment_class']})</b>\n"
    )
    if news["articles"]:
        msg += "\n🔥 <b>Recent Headlines:</b>\n"
        for idx, art in enumerate(news["articles"][:3]):
            msg += f"{idx+1}. <a href='{art['link']}'>{art['title']}</a>\n"
            
    return msg


def handle_weekly(chat_id: str, args: List[str]) -> str:
    """Handle /weekly command: reports the weekly trading schedule."""
    msg = (
        f"📅 <b>BMRI Weekly Schedule (WIB/Jakarta)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• Monday - Friday:\n"
        f"  - 08:30 WIB: Morning Context Update\n"
        f"  - 12:00 WIB: Midday Ingestion Update\n"
        f"  - 16:15 WIB: Closing Signal Analysis\n"
        f"• Saturday 09:00 WIB:\n"
        f"  - Weekly Strategy & Performance Review\n"
    )
    return msg


def handle_status(chat_id: str, args: List[str]) -> str:
    """Handle /status command: returns status check details."""
    import sys
    msg = (
        f"⚙️ <b>SYSTEM STATUS DETAILS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• OS Platform: <b>Linux</b>\n"
        f"• Python Environment: <b>Conda (elliott)</b>\n"
        f"• Python Version: <b>{sys.version.split()[0]}</b>\n"
        f"• Database Status: <b>Active (predictions.db)</b>\n"
        f"• Execution Mode: <b>Docker Containerized</b>\n"
        f"• Status: 🟢 <b>OPERATIONAL</b>\n"
    )
    return msg


def register_stock_handlers(bot: Any):
    """Register all stock commands to the TelegramBot instance."""
    bot.register_command("bmri", handle_bmri)
    bot.register_command("report", handle_report)
    bot.register_command("weekly", handle_weekly)
    bot.register_command("status", handle_status)
