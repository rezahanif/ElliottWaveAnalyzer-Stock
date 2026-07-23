"""
economic_calender.py
---------------------
EconomicCalendarEngine — deterministic, no ML.

Reads scheduled high-impact US macro events (FOMC, CPI, NFP, PCE, GDP)
from config/economic_calender.yaml and computes:
    1. days_to_event for each event type, relative to a given date
    2. the single nearest upcoming high-impact event
    3. a confidence adjustment multiplier for ConfluenceChecker output

All thresholds and multipliers are loaded from config/economic_calender.yaml.
That file is explicitly marked as an unvalidated prior — see calibration_status
and calibration_todo at the top of the YAML before trusting these numbers
in a live signal. This module does not decide whether the prior is correct,
it only applies whatever is currently configured.

NFP is computed algorithmically (first Friday of the month) since that
rule is reliable; FOMC dates are read from a verified static list since
meeting dates are not rule-derivable. CPI/PCE/GDP currently ship empty —
see the YAML file for why and what's needed to fill them in.

Usage:
    from src.waveconf.ingestion.economic_calender import EconomicCalendarEngine

    engine = EconomicCalendarEngine()
    ctx = engine.get_context(date(2026, 6, 20))

    ctx.days_to_fomc          # 39  (next FOMC: 2026-07-29)
    ctx.days_since_last_fomc  # 3   (last FOMC: 2026-06-17)
    ctx.nearest_event         # EventInfo(type="FOMC", date=..., days_away=3, is_past=True)
    ctx.high_impact_within_5d # False (3 days since, not before)

    adjusted = engine.adjust_confidence(0.89, ctx)
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Config loader — matches fib_engine.fibonacci._load_yaml convention
# ─────────────────────────────────────────────────────────────

def _load_yaml(relative_path: str) -> dict:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    full = os.path.join(base, relative_path)
    with open(full, "r") as f:
        return yaml.safe_load(f)


def _parse_dates(raw: List[str]) -> List[date]:
    return sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in (raw or []))


def _first_friday(year: int, month: int) -> date:
    """NFP release date: first Friday of the month, 08:30 ET."""
    d = date(year, month, 1)
    offset = (4 - d.weekday()) % 7  # Friday = weekday 4
    return d + timedelta(days=offset)


# ─────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────

@dataclass
class EventInfo:
    event_type: str
    event_date: date
    days_away: int          # negative if in the past
    is_past: bool


@dataclass
class CalendarContext:
    as_of: date

    days_to_fomc: Optional[int]
    days_since_last_fomc: Optional[int]
    next_fomc_date: Optional[date]
    last_fomc_date: Optional[date]

    days_to_nfp: Optional[int]
    next_nfp_date: Optional[date]

    days_to_cpi: Optional[int]
    next_cpi_date: Optional[date]

    days_to_pce: Optional[int]
    next_pce_date: Optional[date]

    days_to_gdp: Optional[int]
    next_gdp_date: Optional[date]

    nearest_event: Optional[EventInfo]
    high_impact_within_5d: bool
    high_impact_within_2d: bool
    post_event_window: bool   # within post_event_window_days AFTER last FOMC

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "days_to_fomc": self.days_to_fomc,
            "days_since_last_fomc": self.days_since_last_fomc,
            "next_fomc_date": self.next_fomc_date.isoformat() if self.next_fomc_date else None,
            "days_to_nfp": self.days_to_nfp,
            "next_nfp_date": self.next_nfp_date.isoformat() if self.next_nfp_date else None,
            "days_to_cpi": self.days_to_cpi,
            "days_to_pce": self.days_to_pce,
            "days_to_gdp": self.days_to_gdp,
            "nearest_event": (
                f"{self.nearest_event.event_type} "
                f"{'in' if not self.nearest_event.is_past else 'was'} "
                f"{abs(self.nearest_event.days_away)}d"
                if self.nearest_event else None
            ),
            "high_impact_within_5d": self.high_impact_within_5d,
            "high_impact_within_2d": self.high_impact_within_2d,
            "post_event_window": self.post_event_window,
        }


# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────

class EconomicCalendarEngine:

    def __init__(self, config_path: str = "config/economic_calender.yaml"):
        cfg = _load_yaml(config_path)

        self.calibration_status = cfg.get("calibration_status", "unknown")

        rules = cfg.get("adjustment_rules", {})
        self.risk_discount_2d        = rules.get("risk_discount_2d", 0.60)
        self.risk_discount_5d        = rules.get("risk_discount_5d", 0.80)
        self.post_event_boost        = rules.get("post_event_boost", 1.10)
        self.post_event_window_days  = rules.get("post_event_window_days", 2)

        self.high_impact_types = set(cfg.get("high_impact_event_types", []))

        self.fomc_dates = _parse_dates(cfg.get("FOMC", []))
        self.nfp_overrides = _parse_dates(cfg.get("NFP_overrides", []))
        self.cpi_dates = _parse_dates(cfg.get("CPI", []))
        self.pce_dates = _parse_dates(cfg.get("PCE", []))
        self.gdp_dates = _parse_dates(cfg.get("GDP", []))

    # ── public API ──────────────────────────────────────────

    def get_context(self, as_of: Optional[date] = None) -> CalendarContext:
        as_of = as_of or date.today()

        next_fomc, last_fomc = self._surrounding(self.fomc_dates, as_of)
        next_nfp = self._next_nfp(as_of)
        next_cpi, _ = self._surrounding(self.cpi_dates, as_of)
        next_pce, _ = self._surrounding(self.pce_dates, as_of)
        next_gdp, _ = self._surrounding(self.gdp_dates, as_of)

        days_to_fomc = (next_fomc - as_of).days if next_fomc else None
        days_since_last_fomc = (as_of - last_fomc).days if last_fomc else None
        days_to_nfp = (next_nfp - as_of).days if next_nfp else None
        days_to_cpi = (next_cpi - as_of).days if next_cpi else None
        days_to_pce = (next_pce - as_of).days if next_pce else None
        days_to_gdp = (next_gdp - as_of).days if next_gdp else None

        # Collect every event (past + future) within a wide window to find
        # the single nearest one, since "nearest" might be a recent past
        # event (e.g. 1 day after FOMC) rather than an upcoming one.
        candidates: List[EventInfo] = []
        for label, dates_list in [
            ("FOMC", self.fomc_dates),
            ("NFP", [next_nfp] if next_nfp else []),
            ("CPI", self.cpi_dates),
            ("PCE", self.pce_dates),
            ("GDP", self.gdp_dates),
        ]:
            if label not in self.high_impact_types:
                continue
            for d in dates_list:
                delta = (d - as_of).days
                candidates.append(EventInfo(label, d, delta, is_past=delta < 0))

        nearest_event = min(candidates, key=lambda e: abs(e.days_away)) if candidates else None

        high_impact_within_5d = any(
            0 <= e.days_away <= 5 for e in candidates
        )
        high_impact_within_2d = any(
            0 <= e.days_away <= 2 for e in candidates
        )
        post_event_window = (
            last_fomc is not None
            and 0 <= days_since_last_fomc <= self.post_event_window_days
        )

        return CalendarContext(
            as_of=as_of,
            days_to_fomc=days_to_fomc,
            days_since_last_fomc=days_since_last_fomc,
            next_fomc_date=next_fomc,
            last_fomc_date=last_fomc,
            days_to_nfp=days_to_nfp,
            next_nfp_date=next_nfp,
            days_to_cpi=days_to_cpi,
            next_cpi_date=next_cpi,
            days_to_pce=days_to_pce,
            next_pce_date=next_pce,
            days_to_gdp=days_to_gdp,
            next_gdp_date=next_gdp,
            nearest_event=nearest_event,
            high_impact_within_5d=high_impact_within_5d,
            high_impact_within_2d=high_impact_within_2d,
            post_event_window=post_event_window,
        )

    def adjust_confidence(self, confluence_strength: float, ctx: Optional[CalendarContext] = None,
                           as_of: Optional[date] = None) -> float:
        """
        Apply the calendar risk adjustment to a raw confluence_strength score.

        Priority order when multiple conditions could apply on the same day
        (e.g. discount window and post-event window technically overlapping):
            1. post_event_boost takes priority over discounts — if we are
               within post_event_window_days AFTER the last FOMC, we are not
               also "approaching" that same FOMC, so this is unambiguous in
               practice. Kept as explicit priority for clarity, not because
               an overlap is expected to occur.
            2. risk_discount_2d > risk_discount_5d (tighter window wins)
        """
        if ctx is None:
            ctx = self.get_context(as_of)

        if ctx.post_event_window:
            return confluence_strength * self.post_event_boost

        if ctx.high_impact_within_2d:
            return confluence_strength * self.risk_discount_2d

        if ctx.high_impact_within_5d:
            return confluence_strength * self.risk_discount_5d

        return confluence_strength

    # ── internal helpers ────────────────────────────────────

    def _surrounding(self, dates_list: List[date], as_of: date):
        """Return (next_upcoming_or_today, most_recent_past) from a sorted date list."""
        next_date = None
        last_date = None
        for d in dates_list:
            if d >= as_of and next_date is None:
                next_date = d
            if d < as_of:
                last_date = d
        return next_date, last_date

    def _nfp_for_month(self, year: int, month: int) -> date:
        """Return the NFP release date for a given month/year (overridden or algorithmic)."""
        for d in self.nfp_overrides:
            if d.year == year and d.month == month:
                return d
        return _first_friday(year, month)

    def _next_nfp(self, as_of: date) -> Optional[date]:
        # Check current month's NFP
        candidate = self._nfp_for_month(as_of.year, as_of.month)
        if candidate >= as_of:
            return candidate

        # Algorithmic fallback: next month's NFP
        year, month = as_of.year, as_of.month + 1
        if month > 12:
            year, month = year + 1, 1
        return self._nfp_for_month(year, month)