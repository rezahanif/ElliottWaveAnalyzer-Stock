"""
test_collectors.py
------------------
Verify that our stock, IHSG, and sector collectors work correctly using fallbacks.
"""

from __future__ import annotations

import logging
from src.stock.collectors.price import fetch_price_data
from src.stock.collectors.ihsg import fetch_ihsg_data
from src.stock.collectors.sector import fetch_sector_data

logging.basicConfig(level=logging.INFO)


def test_collectors_smoke():
    print("Testing BMRI.JK fetch...")
    # Fetch a short window
    df_bmri = fetch_price_data("BMRI.JK", start_date="2025-01-01", end_date="2025-02-01")
    if df_bmri is not None:
        print(f"BMRI.JK fetched successfully: {len(df_bmri)} rows")
        print(df_bmri.head(3))
    else:
        print("BMRI.JK fetch FAILED")

    print("\nTesting IHSG (^JKSE) fetch...")
    df_ihsg = fetch_ihsg_data(start_date="2025-01-01", end_date="2025-02-01")
    if df_ihsg is not None:
        print(f"IHSG (^JKSE) fetched successfully: {len(df_ihsg)} rows")
        print(df_ihsg.head(3))
    else:
        print("IHSG fetch FAILED")

    print("\nTesting Sector (^JKFIN) fetch...")
    df_sector = fetch_sector_data(start_date="2025-01-01", end_date="2025-02-01")
    if df_sector is not None:
        print(f"Sector (^JKFIN) fetched successfully: {len(df_sector)} rows")
        print(df_sector.head(3))
    else:
        print("Sector fetch FAILED")


if __name__ == "__main__":
    test_collectors_smoke()
