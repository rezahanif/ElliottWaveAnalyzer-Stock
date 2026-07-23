#!/usr/bin/env python3
"""
test_calendar.py — Quick smoke test for the Playwright-based economic calendar fetcher.

Usage:
    cd /home/rezaserver/ElliottWaveAnalyzer   (or /project/ElliottWaveAnalyzer in container)
    conda activate elliott
    python scripts/test_calendar.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datetime import datetime, timezone, timedelta
from src.waveconf.ingestion.investing_api import get_economic_calendar, _fetch_via_playwright

def test_playwright_raw():
    """Test raw Playwright interception (no date filtering)."""
    print("=" * 60)
    print("TEST 1: Raw Playwright interception")
    print("=" * 60)
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (now + timedelta(hours=28)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = _fetch_via_playwright(start, end, "high")
    if data:
        print("✅ Playwright intercepted data successfully!")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            print(f"   Events: {len(data.get('events', []))}")
            print(f"   Occurrences: {len(data.get('occurrences', []))}")
        else:
            print(f"   Type: {type(data)}, Length: {len(data)}")
    else:
        print("❌ Playwright returned None — no data intercepted.")
    print()
    return data is not None

def test_get_economic_calendar():
    """Test full get_economic_calendar with the Playwright fallback chain."""
    print("=" * 60)
    print("TEST 2: Full get_economic_calendar() pipeline")
    print("=" * 60)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=6)
    end = now + timedelta(hours=28)

    try:
        events = get_economic_calendar(start, end, "high")
        print(f"✅ Fetched {len(events)} high-importance events.")
        for evt in events[:5]:
            print(f"   • {evt.get('event_name', '?')} | {evt.get('currency', '?')} | {evt.get('occurrence_time', '?')}")
        if len(events) > 5:
            print(f"   ... and {len(events) - 5} more")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    ok1 = test_playwright_raw()
    print()
    ok2 = test_get_economic_calendar()
    print()
    print("=" * 60)
    if ok1 and ok2:
        print("🎉 ALL TESTS PASSED — Cloudflare bypass working!")
    elif ok2:
        print("⚠️  Playwright raw failed but pipeline succeeded (fallback worked)")
    else:
        print("💥 TESTS FAILED — Check logs above")
    sys.exit(0 if ok2 else 1)
