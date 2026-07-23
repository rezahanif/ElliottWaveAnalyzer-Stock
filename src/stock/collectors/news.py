"""
news.py
-------
Fetches financial news feed for stock symbols and computes sentiment.
Uses Yahoo Finance RSS feeds as the primary data source.
"""

from __future__ import annotations

import logging
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional

logger = logging.getLogger("stock_news")

# Simple sentiment lexicon for market news
SENTIMENT_WORDS = {
    "positive": ["buy", "growth", "bullish", "profit", "gain", "rise", "increase", "upbeat", "outperform", "dividend", "surge"],
    "negative": ["sell", "decline", "bearish", "loss", "fall", "decrease", "drop", "warn", "slump", "deficit", "risk", "down"],
}


def parse_yahoo_rss(xml_data: str) -> List[Dict[str, str]]:
    """Parse news headlines and links from Yahoo RSS XML."""
    news_items = []
    try:
        root = ET.fromstring(xml_data)
        for item in root.findall(".//item"):
            title = item.find("title")
            link = item.find("link")
            pub_date = item.find("pubDate")
            
            title_text = title.text if title is not None else ""
            link_text = link.text if link is not None else ""
            date_text = pub_date.text if pub_date is not None else ""
            
            if title_text:
                news_items.append({
                    "title": title_text,
                    "link": link_text,
                    "pub_date": date_text,
                })
    except Exception as e:
        logger.error(f"Failed to parse RSS XML: {e}")
    return news_items


def calculate_sentiment(text: str) -> float:
    """Calculate simple sentiment score from -1.0 (very negative) to +1.0 (very positive)."""
    text_lower = text.lower()
    pos_count = sum(1 for w in SENTIMENT_WORDS["positive"] if w in text_lower)
    neg_count = sum(1 for w in SENTIMENT_WORDS["negative"] if w in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def fetch_news_and_sentiment(symbol: str) -> Dict[str, Any]:
    """Fetch news headlines from Yahoo RSS and compute sentiment metrics."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}"
    logger.info(f"Fetching news for {symbol} from Yahoo RSS...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            xml_data = r.read().decode("utf-8")
        
        items = parse_yahoo_rss(xml_data)
        if not items:
            return {"sentiment_score": 0.0, "articles": [], "sentiment_class": "NEUTRAL"}
            
        scores = []
        for item in items:
            score = calculate_sentiment(item["title"])
            item["sentiment"] = score
            scores.append(score)
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        if avg_score > 0.1:
            sentiment_class = "POSITIVE"
        elif avg_score < -0.1:
            sentiment_class = "NEGATIVE"
        else:
            sentiment_class = "NEUTRAL"
            
        return {
            "sentiment_score": avg_score,
            "sentiment_class": sentiment_class,
            "articles": items[:5],  # Latest 5 articles
        }
    except Exception as e:
        logger.warning(f"Failed to fetch news for {symbol}: {e}")
        return {"sentiment_score": 0.0, "articles": [], "sentiment_class": "NEUTRAL"}
