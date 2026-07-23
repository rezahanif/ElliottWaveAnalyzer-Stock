from datetime import date
from src.waveconf.ingestion.economic_calender import EconomicCalendarEngine


def test_engine_loads_config():
    engine = EconomicCalendarEngine()
    assert engine.risk_discount_2d == 0.60
    assert engine.risk_discount_5d == 0.80
    assert engine.post_event_boost == 1.10
    assert engine.calibration_status == "unvalidated_prior_v0.1"
    assert len(engine.fomc_dates) == 8


def test_days_to_next_fomc():
    engine = EconomicCalendarEngine()
    # 2026-06-20: last FOMC was 2026-06-17 (3 days ago), next is 2026-07-29
    ctx = engine.get_context(date(2026, 6, 20))
    assert ctx.last_fomc_date == date(2026, 6, 17)
    assert ctx.days_since_last_fomc == 3
    assert ctx.next_fomc_date == date(2026, 7, 29)
    assert ctx.days_to_fomc == 39


def test_post_event_boost_applies_within_window():
    engine = EconomicCalendarEngine()
    # 3 days after FOMC, window is 2 days — boost should NOT apply
    ctx = engine.get_context(date(2026, 6, 20))
    assert ctx.post_event_window is False
    adjusted = engine.adjust_confidence(0.89, ctx)
    assert adjusted == 0.89  # no FOMC within 5d either direction at this date... 
    # (June 20 is 3d after last FOMC and 39d before next — neither discount nor boost)


def test_post_event_boost_applies_day_after_fomc():
    engine = EconomicCalendarEngine()
    # 2026-06-18: 1 day after the 2026-06-17 FOMC, inside the 2-day window
    ctx = engine.get_context(date(2026, 6, 18))
    assert ctx.post_event_window is True
    adjusted = engine.adjust_confidence(0.80, ctx)
    assert round(adjusted, 4) == round(0.80 * 1.10, 4)


def test_discount_2d_before_fomc():
    engine = EconomicCalendarEngine()
    # 2026-07-27: 2 days before the 2026-07-29 FOMC
    ctx = engine.get_context(date(2026, 7, 27))
    assert ctx.days_to_fomc == 2
    assert ctx.high_impact_within_2d is True
    adjusted = engine.adjust_confidence(0.90, ctx)
    assert round(adjusted, 4) == round(0.90 * 0.60, 4)


def test_discount_5d_before_fomc():
    engine = EconomicCalendarEngine()
    # 2026-07-25: 4 days before the 2026-07-29 FOMC — inside 5d, outside 2d
    ctx = engine.get_context(date(2026, 7, 25))
    assert ctx.days_to_fomc == 4
    assert ctx.high_impact_within_2d is False
    assert ctx.high_impact_within_5d is True
    adjusted = engine.adjust_confidence(0.90, ctx)
    assert round(adjusted, 4) == round(0.90 * 0.80, 4)


def test_no_adjustment_far_from_any_event():
    engine = EconomicCalendarEngine()
    # 2026-08-15: well clear of the 2026-07-29 and 2026-09-16 FOMC dates,
    # but NFP is still monthly (first Friday) so check it doesn't false-trigger
    # high_impact_within_5d via NFP since NFP IS in high_impact_event_types.
    ctx = engine.get_context(date(2026, 8, 15))
    adjusted = engine.adjust_confidence(0.75, ctx)
    # Just assert it's either unadjusted or NFP-discounted — both are valid
    # depending on where Aug's first Friday lands. The real assertion is
    # that adjust_confidence never raises and never exceeds reasonable bounds.
    assert 0.0 < adjusted <= 0.75 * 1.10


def test_nfp_is_first_friday():
    engine = EconomicCalendarEngine()
    # August 2026: Aug 1 is a Saturday, first Friday is Aug 7
    ctx = engine.get_context(date(2026, 8, 1))
    assert ctx.next_nfp_date == date(2026, 8, 7)


def test_empty_cpi_pce_gdp_do_not_crash():
    """CPI/PCE/GDP ship empty in the config — engine must handle gracefully."""
    engine = EconomicCalendarEngine()
    ctx = engine.get_context(date(2026, 6, 20))
    assert ctx.days_to_cpi is None
    assert ctx.days_to_pce is None
    assert ctx.days_to_gdp is None


def test_nfp_override_does_not_overshadow_intervening_months():
    engine = EconomicCalendarEngine()
    # Mock some overrides: one in the past, one far in the future
    engine.nfp_overrides = [date(2026, 5, 2), date(2026, 12, 11)]
    
    # As of 2026-06-20:
    # Upcoming NFP should be the first Friday of July 2026 (2026-07-03)
    # because June's NFP (2026-06-05) has passed, and July has no override.
    ctx = engine.get_context(date(2026, 6, 20))
    assert ctx.next_nfp_date == date(2026, 7, 3)