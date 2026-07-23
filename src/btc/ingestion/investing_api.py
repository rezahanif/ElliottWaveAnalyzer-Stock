"""
investing_api.py
---------------------
Ingestion layer for Investing.com Economic Calendar occurrences API.
Bypasses Cloudflare protection using Playwright (primary), curl_cffi, or requests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import requests

try:
    from curl_cffi import requests as curl_requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from playwright_stealth import Stealth
    PLAYWRIGHT_STEALTH_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_STEALTH_AVAILABLE = False

PLAYWRIGHT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# ponytail: temporary binary pin. Remove after Playwright supports Ubuntu 26.04.
PLAYWRIGHT_CHROMIUM_CANDIDATES = (
    os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", ""),
    "/home/rezaserver/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome",
    "/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome",
)

def _chromium_executable() -> Path:
    """Return configured compatible Chromium binary."""
    for candidate in PLAYWRIGHT_CHROMIUM_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError(
        "Compatible Chromium executable not found. Set PLAYWRIGHT_CHROMIUM_EXECUTABLE."
    )


# Set up logging
logger = logging.getLogger("investing_api")

# Define canonical HIGH_RISK_EVENTS
HIGH_RISK_EVENTS = {
    "Interest Rate Decision",
    "FOMC",
    "CPI",
    "Core CPI",
    "Nonfarm Payrolls",
    "GDP",
    "Unemployment Rate"
}

# Priority list for sorting (lower index = higher priority)
PRIORITY_ORDER = [
    "Interest Rate Decision",
    "FOMC",
    "CPI",
    "Core CPI",
    "Nonfarm Payrolls",
    "GDP",
    "Unemployment Rate"
]

def is_high_risk(event_name: str) -> bool:
    """Check if the event name matches any defined high risk event as a substring."""
    name_lower = event_name.lower()
    return any(hre.lower() in name_lower for hre in HIGH_RISK_EVENTS)

def get_event_priority(event_name: str, importance: str) -> int:
    """
    Get sorting priority of an event.
    Returns 1-7 for matching HIGH_RISK_EVENTS, 8 for other High Impact, and 9 otherwise.
    """
    name_lower = event_name.lower()
    
    # Check 'core cpi' specifically before general 'cpi'
    if "core cpi" in name_lower:
        return 4 # Priority 4
    elif "cpi" in name_lower:
        return 3 # Priority 3
        
    # Check other events by priority index
    order_without_cpi = [
        ("Interest Rate Decision", 1),
        ("FOMC", 2),
        ("Nonfarm Payrolls", 5),
        ("GDP", 6),
        ("Unemployment Rate", 7)
    ]
    for pattern, prio in order_without_cpi:
        if pattern.lower() in name_lower:
            return prio
            
    if importance.lower() == "high":
        return 8
    return 9


def parse_utc_to_local(utc_time_str: str) -> datetime:
    """Parse UTC ISO8601 string and convert to UTC+7 timezone."""
    # Handle Zulu timezone indicator
    t_str = utc_time_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(t_str)
    local_tz = timezone(timedelta(hours=7))
    return dt.astimezone(local_tz)

def _fetch_via_playwright_once(start_utc_str: str, end_utc_str: str, importance: str) -> dict | None:
    """Load Investing.com's calendar page, then call the API via fetch within page context."""
    if not PLAYWRIGHT_AVAILABLE:
        return None

    calendar_json = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=str(_chromium_executable()),
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            context = browser.new_context(
                user_agent=PLAYWRIGHT_USER_AGENT,
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/New_York",
            )
            page = context.new_page()
            if PLAYWRIGHT_STEALTH_AVAILABLE:
                Stealth().apply_stealth_sync(page)
            else:
                logger.warning("playwright-stealth is not installed; Investing.com may block Chromium.")

            # Block heavy assets
            page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in ["image", "font", "media"]
                else route.continue_(),
            )

            # 1. Access calendar homepage to initialize Cloudflare cookies
            response = page.goto(
                "https://www.investing.com/economic-calendar/",
                wait_until="domcontentloaded",
                timeout=45000,
            )
            if response is None or response.status != 200:
                logger.warning(
                    "Investing.com calendar page returned status %s.",
                    response.status if response else "no response",
                )

            # 2. Build occurrences URL with actual start_date/end_date parameters
            api_url = (
                "https://endpoints.investing.com/pd-instruments/v1/calendars/economic/events/occurrences?"
                f"domain_id=1&limit=200&start_date={start_utc_str}&end_date={end_utc_str}"
            )
            if importance:
                api_url += f"&importance={importance}"

            # 3. Call the API from the page context to leverage session cookies
            logger.info("Executing calendar API fetch inside browser context...")
            calendar_json = page.evaluate(
                """async (url) => {
                    const res = await fetch(url);
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
                    return await res.json();
                }""",
                api_url,
            )
            browser.close()
    except Exception as e:
        logger.warning(f"Playwright fetch failed: {e}")

    return calendar_json


def _fetch_via_playwright(start_utc_str: str, end_utc_str: str, importance: str) -> dict | None:
    """Retry browser fetch once; Cloudflare can reject a fresh browser session."""
    for attempt in range(2):
        data = _fetch_via_playwright_once(start_utc_str, end_utc_str, importance)
        if data:
            return data
        if attempt == 0:
            logger.warning("Playwright returned no calendar data; retrying once.")
    return None



def get_economic_calendar(
    start_date: datetime,
    end_date: datetime,
    importance: str = "high"
) -> list[dict]:
    """
    Fetch economic calendar events and occurrences from Investing.com.
    Fallback chain: Playwright → curl_cffi → requests.
    Joins events[] and occurrences[] by event_id, converting to canonical structure.
    """
    url = "https://endpoints.investing.com/pd-instruments/v1/calendars/economic/events/occurrences"
    
    # Format start and end date to ISO8601 UTC strings
    start_utc = start_date.astimezone(timezone.utc)
    end_utc = end_date.astimezone(timezone.utc)
    
    params = {
        "domain_id": 1,
        "limit": 200,
        "start_date": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "importance": importance
    }
    
    headers = {
        "Origin": "https://www.investing.com",
        "Referer": "https://www.investing.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    data = None

    # 1. Try Playwright (bypasses Cloudflare)
    if PLAYWRIGHT_AVAILABLE:
        logger.info("Fetching economic calendar via Playwright...")
        start_utc_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc_str = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = _fetch_via_playwright(start_utc_str, end_utc_str, importance)
        if data:
            logger.info("Playwright fetch succeeded.")
        else:
            logger.warning("Playwright fetch returned no data. Falling back.")

    # 2. Fallback: curl_cffi
    if data is None and CURL_CFFI_AVAILABLE:
        try:
            logger.info("Fetching economic calendar via curl_cffi...")
            resp = curl_requests.get(url, params=params, headers=headers, impersonate="chrome120", timeout=15)
            if resp.status_code == 200:
                data = resp.json()
            else:
                logger.warning(f"curl_cffi fetch returned status code {resp.status_code}. Falling back.")
        except Exception as e:
            logger.warning(f"curl_cffi fetch failed: {e}. Falling back to standard requests.")
            
    # 3. Fallback: standard requests
    if data is None:
        try:
            logger.info("Fetching economic calendar via requests...")
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch economic calendar: {e}")
            raise RuntimeError(f"Could not retrieve economic calendar data: {e}")

    events = data.get("events", [])
    occurrences = data.get("occurrences", [])
    
    # Map event_id to event data
    event_map = {}
    for evt in events:
        if isinstance(evt, dict) and "event_id" in evt:
            event_map[evt["event_id"]] = evt
            
    canonical_list = []
    for occ in occurrences:
        if not isinstance(occ, dict):
            continue
        event_id = occ.get("event_id")
        event = event_map.get(event_id, {})
        
        # Build canonical dict
        canonical_list.append({
            "event_id": event_id,
            "occurrence_id": occ.get("occurrence_id"),
            "event_name": event.get("event_translated", ""),
            "currency": event.get("currency", ""),
            "importance": event.get("importance", ""),
            "occurrence_time": occ.get("occurrence_time", ""),
            "actual": occ.get("actual"),
            "forecast": occ.get("forecast"),
            "previous": occ.get("previous"),
            "actual_to_forecast": occ.get("actual_to_forecast"),
            "revised_to_previous": occ.get("revised_to_previous")
        })
        
    return canonical_list
