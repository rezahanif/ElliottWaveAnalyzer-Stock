"""
astro_features.py
------------------
AstroFeaturesEngine — deterministic astronomical calculation. No ML, no
hardcoded directional belief.

Computes, for any given date (past OR future — these are all mechanically
predictable from orbital mechanics, which is exactly why they belong in
the TFT's known-future input channel rather than the observed-past channel):

    1. Lunar synodic phase   (Sun-Moon angle: raw / sin / cos)
    2. Lunar anomalistic cycle (perigee/apogee normalized distance)
    3. Lunar draconic node distance (eclipse-season proximity)
    4. Planetary longitude + retrograde flag (mars, uranus, jupiter, saturn)
    5. Angular aspects between configured planet pairs (continuous 0-1
       intensity per configured angle — UNSIGNED, no bullish/bearish label)
    6. Mercury retrograde binary flag

All config (which planets, which aspects, which orbs) is loaded from
config/astro_features.yaml. This module does not decide whether any of
these features matter for BTC — it only computes them. Significance is
for the TFT to learn from training data.

IMPORTANT — verified 2026-06-20 against the historical reference table
that originally motivated this module: several "Full Moon" claims in
that table did not match computed lunar phase by 3-5 days, and one
claimed "trine" was actually a 98.6 degree angle. Always trust this
engine's output over any hand-curated historical claim — see
config/astro_features.yaml `calibration_notes`.

Usage:
    from datetime import date
    from src.waveconf.wave_model.astro_features import AstroFeaturesEngine

    engine = AstroFeaturesEngine()
    feats = engine.get_daily_features(date(2026, 6, 20))

    feats.lunar_phase_deg       # 0-360
    feats.lunar_phase_sin       # cyclical encoding
    feats.mercury_retrograde    # 0 or 1
    feats.aspects['jupiter_uranus']   # AspectResult with .intensity (0-1)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import yaml

try:
    import swisseph as swe
except ImportError as e:
    raise ImportError(
        "pyswisseph is required: pip install pyswisseph --break-system-packages"
    ) from e


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


# ─────────────────────────────────────────────────────────────
# Swiss Ephemeris planet IDs
# ─────────────────────────────────────────────────────────────

_PLANET_IDS = {
    "sun": swe.SUN, "moon": swe.MOON, "mercury": swe.MERCURY,
    "venus": swe.VENUS, "mars": swe.MARS, "jupiter": swe.JUPITER,
    "saturn": swe.SATURN, "uranus": swe.URANUS, "neptune": swe.NEPTUNE,
    "pluto": swe.PLUTO,
}
_TRUE_NODE = swe.TRUE_NODE  # Rahu / North Node; Ketu = +180 deg

# Lunar distance bounds in km, stable enough for 2015-2026 normalization
_PERIGEE_KM = 356_500.0
_APOGEE_KM = 406_700.0
_AU_KM = 149_597_870.7


def _angle_diff(a: float, b: float) -> float:
    """Shortest angular distance between two longitudes, 0-180."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


# ─────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────

@dataclass
class AspectResult:
    pair: str
    nearest_angle: int          # which configured angle (0/90/120/180) is closest
    exact_separation: float     # actual angular separation, degrees
    orb: float                  # |exact_separation - nearest_angle|
    intensity: float            # 1.0 at exact angle, 0.0 at max_orb edge. UNSIGNED.


@dataclass
class PlanetState:
    planet: str
    longitude: float
    lon_sin: float
    lon_cos: float
    retrograde: Optional[bool]   # None if track_velocity=False for this planet


@dataclass
class DailyAstroFeatures:
    as_of: date

    lunar_phase_deg: float
    lunar_phase_sin: float
    lunar_phase_cos: float

    lunar_anomalistic_normalized: float   # 0=perigee, 1=apogee

    lunar_node_distance: float            # degrees, 0=on a node (eclipse season)

    planets: Dict[str, PlanetState]
    aspects: Dict[str, AspectResult]

    mercury_retrograde: int               # 0 or 1

    def to_flat_dict(self) -> dict:
        """Flatten into a single-level dict suitable for a feature matrix row."""
        out = {
            "lunar_phase_deg": self.lunar_phase_deg,
            "lunar_phase_sin": self.lunar_phase_sin,
            "lunar_phase_cos": self.lunar_phase_cos,
            "lunar_anomalistic_normalized": self.lunar_anomalistic_normalized,
            "lunar_node_distance": self.lunar_node_distance,
            "mercury_retrograde": self.mercury_retrograde,
        }
        for name, p in self.planets.items():
            out[f"{name}_lon_sin"] = p.lon_sin
            out[f"{name}_lon_cos"] = p.lon_cos
            if p.retrograde is not None:
                out[f"{name}_retrograde"] = int(p.retrograde)
        for pair_key, a in self.aspects.items():
            out[f"aspect_{pair_key}_intensity"] = a.intensity
        return out


# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────

class AstroFeaturesEngine:

    def __init__(self, config_path: str = "config/astro_features.yaml"):
        cfg = _load_yaml(config_path)

        astro_cfg = cfg.get("astronomy_settings", {})
        ephe_path = astro_cfg.get("ephemeris_path")
        if ephe_path:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            ))))
            full_path = os.path.join(base, ephe_path)
            if os.path.isdir(full_path):
                swe.set_ephe_path(full_path)
            # If the path doesn't exist or has no .se1 files, pyswisseph
            # silently falls back to the built-in Moshier ephemeris.
            # No error needed — this is documented in the YAML comments.

        features_cfg = cfg.get("features", {})

        self.lunar_cfg = features_cfg.get("lunar_dynamics", {})
        self.planet_cfg: List[dict] = features_cfg.get("planetary_tracking", [])
        self.aspect_cfg: List[dict] = features_cfg.get("angular_aspects", [])
        self.mercury_rx_enabled = (
            features_cfg.get("noise_filters", {})
            .get("mercury_retrograde", {})
            .get("enabled", True)
        )

    # ── public API ──────────────────────────────────────────

    def get_daily_features(self, as_of: date) -> DailyAstroFeatures:
        jd = swe.julday(as_of.year, as_of.month, as_of.day, 12.0)  # noon UTC

        sun_lon, _ = self._calc(jd, "sun")
        moon_lon, _ = self._calc(jd, "moon")
        moon_dist_au = swe.calc_ut(jd, swe.MOON)[0][2]
        node_lon, _ = self._calc(jd, "node")

        # ── Lunar synodic phase ──
        phase_deg = (moon_lon - sun_lon) % 360
        phase_sin = math.sin(math.radians(phase_deg))
        phase_cos = math.cos(math.radians(phase_deg))

        # ── Lunar anomalistic (distance) ──
        moon_dist_km = moon_dist_au * _AU_KM
        anomalistic_norm = (moon_dist_km - _PERIGEE_KM) / (_APOGEE_KM - _PERIGEE_KM)
        anomalistic_norm = max(0.0, min(1.0, anomalistic_norm))

        # ── Lunar draconic node distance ──
        node_dist = min(_angle_diff(moon_lon, node_lon),
                         _angle_diff(moon_lon, (node_lon + 180) % 360))

        # ── Planetary tracking ──
        planets: Dict[str, PlanetState] = {}
        planet_lons: Dict[str, float] = {}
        for entry in self.planet_cfg:
            name = entry["planet"]
            lon, speed = self._calc(jd, name)
            planet_lons[name] = lon
            retro = (speed < 0) if entry.get("track_velocity", False) else None
            planets[name] = PlanetState(
                planet=name,
                longitude=lon,
                lon_sin=math.sin(math.radians(lon)),
                lon_cos=math.cos(math.radians(lon)),
                retrograde=retro,
            )

        # ── Angular aspects (unsigned intensity) ──
        aspects: Dict[str, AspectResult] = {}
        for entry in self.aspect_cfg:
            a_name, b_name = entry["pair"]
            lon_a = planet_lons.get(a_name) or self._calc(jd, a_name)[0]
            lon_b = planet_lons.get(b_name) or self._calc(jd, b_name)[0]
            sep = _angle_diff(lon_a, lon_b)

            best_angle, best_orb = None, 999.0
            for target_angle in entry["angles"]:
                orb = abs(sep - target_angle)
                if orb < best_orb:
                    best_angle, best_orb = target_angle, orb

            max_orb = entry["max_orb"]
            intensity = max(0.0, 1.0 - (best_orb / max_orb)) if best_orb <= max_orb else 0.0

            key = f"{a_name}_{b_name}"
            aspects[key] = AspectResult(
                pair=key, nearest_angle=best_angle,
                exact_separation=sep, orb=best_orb, intensity=intensity,
            )

        # ── Mercury retrograde flag ──
        _, merc_speed = self._calc(jd, "mercury")
        mercury_rx = int(merc_speed < 0) if self.mercury_rx_enabled else 0

        return DailyAstroFeatures(
            as_of=as_of,
            lunar_phase_deg=phase_deg,
            lunar_phase_sin=phase_sin,
            lunar_phase_cos=phase_cos,
            lunar_anomalistic_normalized=anomalistic_norm,
            lunar_node_distance=node_dist,
            planets=planets,
            aspects=aspects,
            mercury_retrograde=mercury_rx,
        )

    # ── internal helpers ────────────────────────────────────

    def _calc(self, jd: float, planet_name: str):
        """Returns (longitude, speed). speed < 0 means retrograde."""
        if planet_name == "node":
            pos, _ = swe.calc_ut(jd, _TRUE_NODE)
        else:
            pos, _ = swe.calc_ut(jd, _PLANET_IDS[planet_name])
        return pos[0], pos[3]