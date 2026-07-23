"""
idx_holidays.py
---------------
Deterministic IDX (Indonesia Stock Exchange) holiday registry.
Ensures swing trading operations only execute on active market days (mon-fri, non-holidays).
"""

from __future__ import annotations

from datetime import date


# 2026 IDX / Indonesian Public Holidays & Joint Decrees (Cuti Bersama)
IDX_HOLIDAYS_2026 = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 15),   # Isra Mi'raj
    date(2026, 2, 16),   # Lunar New Year (Imlek)
    date(2026, 2, 17),   # Joint Decree (Lunar New Year)
    date(2026, 3, 20),   # Hari Raya Nyepi
    date(2026, 3, 23),   # Joint Decree (Hari Raya Nyepi)
    date(2026, 3, 20),   # Good Friday
    date(2026, 4, 3),    # Easter Monday / Joint Decree
    date(2026, 3, 19),   # Eid al-Fitr (estimation starts)
    date(2026, 3, 20),   # Eid al-Fitr
    date(2026, 3, 21),   # Eid al-Fitr Joint Decree
    date(2026, 3, 22),   # Eid al-Fitr Joint Decree
    date(2026, 3, 23),   # Eid al-Fitr Joint Decree
    date(2026, 5, 1),    # Labor Day
    date(2026, 5, 14),   # Ascension of Jesus Christ
    date(2026, 5, 27),   # Hari Raya Waisak
    date(2026, 5, 28),   # Joint Decree (Hari Raya Waisak)
    date(2026, 6, 1),    # Pancasila Day
    date(2026, 5, 27),   # Eid al-Adha (estimation)
    date(2026, 5, 28),   # Joint Decree (Eid al-Adha)
    date(2026, 7, 7),    # Islamic New Year 1448 H
    date(2026, 8, 17),   # Independence Day
    date(2026, 9, 15),   # Prophet Muhammad's Birthday
    date(2026, 12, 25),  # Christmas Day
    date(2026, 12, 28),  # Joint Decree (Christmas)
}


def is_idx_trading_day(d: date) -> bool:
    """Check if the given date is a valid trading day for the IDX."""
    # 1. Weekday check (0 = Monday, 6 = Sunday)
    if d.weekday() >= 5:
        return False

    # 2. Holiday check
    if d in IDX_HOLIDAYS_2026:
        return False

    return True
