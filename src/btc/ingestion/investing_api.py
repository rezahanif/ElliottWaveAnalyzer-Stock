"""
Backward-compatible re-export.
Real implementation lives in src/shared/economic_calendar/investing_api.py
"""
from src.shared.economic_calendar.investing_api import *  # noqa: F401,F403
from src.shared.economic_calendar.investing_api import fetch_economic_calendar
