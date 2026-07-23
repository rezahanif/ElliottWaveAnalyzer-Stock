from datetime import date
from src.waveconf.wave_model.astro_features import AstroFeaturesEngine


def test_engine_loads_config():
    engine = AstroFeaturesEngine()
    assert "jupiter" in engine.planet_cfg[2]["planet"] or any(
        p["planet"] == "jupiter" for p in engine.planet_cfg
    )
    assert any(a["pair"] == ["jupiter", "uranus"] for a in engine.aspect_cfg)
    assert engine.mercury_rx_enabled is True


def test_mercury_retrograde_dec_2017():
    """Manually verified against ephemeris: TRUE on this date."""
    engine = AstroFeaturesEngine()
    f = engine.get_daily_features(date(2017, 12, 17))
    assert f.mercury_retrograde == 1


def test_mercury_not_retrograde_may_2021():
    """
    The original reference table claimed 'Mercury Retrograde window' on
    2021-05-26. Verified FALSE — Mercury did not station retrograde
    until 2021-05-29, three days later. This test pins the correction.
    """
    engine = AstroFeaturesEngine()
    f = engine.get_daily_features(date(2021, 5, 26))
    assert f.mercury_retrograde == 0


def test_full_moon_may_2021_total_lunar_eclipse():
    """This claim WAS verified true — well documented Super Flower Blood Moon."""
    engine = AstroFeaturesEngine()
    f = engine.get_daily_features(date(2021, 5, 26))
    assert abs(f.lunar_phase_deg - 180) < 2.0  # within 2 deg of exact full moon


def test_full_moon_claim_march_2020_is_false():
    """
    The original reference table claimed 'Full Moon' on 2020-03-13
    (Black Thursday). Verified FALSE — actual phase was 232.9 deg
    (127 deg from Sun, a waning gibbous), the real full moon was
    March 9. This test documents the correction so nobody re-encodes
    the original claim later.
    """
    engine = AstroFeaturesEngine()
    f = engine.get_daily_features(date(2020, 3, 13))
    assert abs(f.lunar_phase_deg - 180) > 50  # nowhere near full moon


def test_jupiter_uranus_conjunction_april_2024():
    """
    Strongest verified claim in the original table: near-exact
    Jupiter-Uranus conjunction (0.1 deg orb) on the 4th halving date.
    """
    engine = AstroFeaturesEngine()
    f = engine.get_daily_features(date(2024, 4, 20))
    aspect = f.aspects["jupiter_uranus"]
    assert aspect.nearest_angle == 0
    assert aspect.orb < 0.5
    assert aspect.intensity > 0.85


def test_aspect_intensity_is_zero_outside_orb():
    engine = AstroFeaturesEngine()
    f = engine.get_daily_features(date(2020, 3, 13))
    # Jupiter-Uranus separation was 102.5 deg that day, nearest configured
    # angle is 120 (orb 17.5), way outside max_orb=3.5 -> intensity 0
    assert f.aspects["jupiter_uranus"].intensity == 0.0


def test_lunar_anomalistic_normalized_in_bounds():
    engine = AstroFeaturesEngine()
    f = engine.get_daily_features(date(2026, 6, 20))
    assert 0.0 <= f.lunar_anomalistic_normalized <= 1.0


def test_sin_cos_encoding_continuity():
    """sin/cos pair must not show a discontinuity across the 0/360 boundary."""
    engine = AstroFeaturesEngine()
    f1 = engine.get_daily_features(date(2026, 1, 1))
    f2 = engine.get_daily_features(date(2026, 1, 2))
    # Adjacent days should never produce a huge jump in sin/cos space
    delta = ((f1.lunar_phase_sin - f2.lunar_phase_sin) ** 2 +
             (f1.lunar_phase_cos - f2.lunar_phase_cos) ** 2) ** 0.5
    assert delta < 0.5


def test_to_flat_dict_has_no_unsigned_polarity_keys():
    """
    Design guardrail: this engine must never output a hand-tuned signed
    'bradley_score' or bullish/bearish label. Only raw unsigned features.
    """
    engine = AstroFeaturesEngine()
    f = engine.get_daily_features(date(2026, 6, 20))
    flat = f.to_flat_dict()
    forbidden_substrings = ["bullish", "bearish", "bradley_score", "polarity"]
    for key in flat:
        for bad in forbidden_substrings:
            assert bad not in key.lower(), f"Found disallowed signed feature: {key}"