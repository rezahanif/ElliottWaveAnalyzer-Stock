"""
test_collectors.py
------------------
Verify that our stock, IHSG, sector, fundamentals, and news collectors work correctly.
"""

from __future__ import annotations

import logging
from src.stock.collectors.price import fetch_price_data
from src.stock.collectors.ihsg import fetch_ihsg_data
from src.stock.collectors.sector import fetch_sector_data
from src.stock.collectors.fundamentals import fetch_fundamentals
from src.stock.collectors.news import fetch_news_and_sentiment

logging.basicConfig(level=logging.INFO)


def test_collectors_smoke():
    print("Testing BMRI.JK fetch...")
    df_bmri = fetch_price_data("BMRI.JK", start_date="2025-01-01", end_date="2025-02-01")
    assert df_bmri is not None and len(df_bmri) > 0
    print(f"BMRI.JK fetched successfully: {len(df_bmri)} rows")

    print("\nTesting IHSG (^JKSE) fetch...")
    df_ihsg = fetch_ihsg_data(start_date="2025-01-01", end_date="2025-02-01")
    assert df_ihsg is not None and len(df_ihsg) > 0
    print(f"IHSG (^JKSE) fetched successfully: {len(df_ihsg)} rows")

    print("\nTesting Sector (^JKLQ45) fetch...")
    df_sector = fetch_sector_data(start_date="2025-01-01", end_date="2025-02-01")
    assert df_sector is not None and len(df_sector) > 0
    print(f"Sector (^JKLQ45) fetched successfully: {len(df_sector)} rows")

    print("\nTesting Fundamentals fetch...")
    fundamentals = fetch_fundamentals("BMRI.JK")
    assert fundamentals["pe_ratio"] == 10.8
    assert fundamentals["pb_ratio"] == 2.05
    assert fundamentals["roe"] == 0.195
    print("Fundamentals loaded successfully:", fundamentals)

    print("\nTesting News fetch...")
    news = fetch_news_and_sentiment("BMRI.JK")
    assert "sentiment_score" in news
    print(f"News fetched. Sentiment score: {news['sentiment_score']:.2f} ({news['sentiment_class']})")
    print(f"Latest headline: {news['articles'][0]['title'] if news['articles'] else 'No headlines'}")
    print("Smoke tests PASSED! ✅")


if __name__ == "__main__":
    test_collectors_smoke()
