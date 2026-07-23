#!/usr/bin/env python3
"""
astro_notifier.py
------------------
Autonomous astronomical notification engine (Astro Agent v1.0).
Computes and formats zodiac sign changes, retrograde entries/stations,
planetary aspects, lunar cycles (phases, apogee/perigee, node transitions),
eclipses, and solstices/equinoxes.

Produces Daily and Weekly summaries with clean layouts and outputs raw JSON format.
"""

from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Set up project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load environment variables
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

import swisseph as swe

# Timezone configurations
LOCAL_TZ = timezone(timedelta(hours=7))

# Constants & Astrology Mappings
PLANETS = {
    "mercury": swe.MERCURY,
    "venus": swe.VENUS,
    "mars": swe.MARS,
    "jupiter": swe.JUPITER,
    "saturn": swe.SATURN,
    "uranus": swe.URANUS,
    "neptune": swe.NEPTUNE,
    "pluto": swe.PLUTO,
}

PLANET_SYMBOLS = {
    "mercury": "☿ Mercury",
    "venus": "♀ Venus",
    "mars": "♂ Mars",
    "jupiter": "♃ Jupiter",
    "saturn": "♄ Saturn",
    "uranus": "♅ Uranus",
    "neptune": "♆ Neptune",
    "pluto": "♇ Pluto",
    "sun": "☉ Sun",
    "moon": "☾ Moon",
}

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

ASPECTS = {
    0: {"name": "Conjunction", "symbol": "☌"},
    60: {"name": "Sextile", "symbol": "⚹"},
    90: {"name": "Square", "symbol": "□"},
    120: {"name": "Trine", "symbol": "△"},
    180: {"name": "Opposition", "symbol": "☍"},
}

def get_zodiac_sign(longitude: float) -> str:
    idx = int(longitude / 30.0) % 12
    return ZODIAC_SIGNS[idx]

def get_jd_at_noon(d: date) -> float:
    return swe.julday(d.year, d.month, d.day, 12.0)

def calculate_planet_state(jd: float, planet_id: int) -> Tuple[float, float, bool]:
    """Returns (longitude, speed, is_retrograde)"""
    pos, _ = swe.calc_ut(jd, planet_id)
    lon = pos[0]
    speed = pos[3]
    return lon, speed, speed < 0

# ─────────────────────────────────────────────────────────────
# Notification Engines & Detections
# ─────────────────────────────────────────────────────────────

class AstroAgent:
    def __init__(self, target_date: date):
        self.target_date = target_date
        self.jd = get_jd_at_noon(target_date)
        self.prev_jd = get_jd_at_noon(target_date - timedelta(days=1))
        self.next_jd = get_jd_at_noon(target_date + timedelta(days=1))

    def detect_planetary_events(self) -> List[Dict[str, Any]]:
        events = []
        for name, pid in PLANETS.items():
            lon, speed, retro = calculate_planet_state(self.jd, pid)
            prev_lon, prev_speed, prev_retro = calculate_planet_state(self.prev_jd, pid)

            # 1. Retrograde/Direct entries & Station events
            # We check speed crossing zero threshold.
            if prev_speed >= 0 and speed < 0:
                events.append({
                    "type": "station_retrograde",
                    "planet": name.capitalize(),
                    "description": f"{PLANET_SYMBOLS[name]} is stationing retrograde."
                })
                events.append({
                    "type": "retrograde_start",
                    "planet": name.capitalize(),
                    "description": f"{PLANET_SYMBOLS[name]} enters retrograde."
                })
            elif prev_speed < 0 and speed >= 0:
                events.append({
                    "type": "station_direct",
                    "planet": name.capitalize(),
                    "description": f"{PLANET_SYMBOLS[name]} is stationing direct."
                })
                events.append({
                    "type": "retrograde_end",
                    "planet": name.capitalize(),
                    "description": f"{PLANET_SYMBOLS[name]} turns direct."
                })

            # 2. Zodiac sign changes
            sign = get_zodiac_sign(lon)
            prev_sign = get_zodiac_sign(prev_lon)
            if sign != prev_sign:
                events.append({
                    "type": "sign_change",
                    "planet": name.capitalize(),
                    "from_sign": prev_sign,
                    "to_sign": sign,
                    "description": f"{PLANET_SYMBOLS[name]} enters {sign} (moving from {prev_sign})."
                })

        # 3. Aspects (between all pairs of outer/inner tracking planets)
        keys = list(PLANETS.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                p1, p2 = keys[i], keys[j]
                pid1, pid2 = PLANETS[p1], PLANETS[p2]
                
                lon1, _, _ = calculate_planet_state(self.jd, pid1)
                lon2, _, _ = calculate_planet_state(self.jd, pid2)
                
                diff = abs(lon1 - lon2) % 360
                sep = min(diff, 360 - diff)
                
                for angle, aspect_info in ASPECTS.items():
                    orb = abs(sep - angle)
                    # We flag aspects that are within a tight exactness orb of 0.5 degrees
                    if orb <= 0.5:
                        # Determine if it is getting closer or moving away (speed vector comparison)
                        prev_lon1, _, _ = calculate_planet_state(self.prev_jd, pid1)
                        prev_lon2, _, _ = calculate_planet_state(self.prev_jd, pid2)
                        prev_diff = abs(prev_lon1 - prev_lon2) % 360
                        prev_sep = min(prev_diff, 360 - prev_diff)
                        
                        next_lon1, _, _ = calculate_planet_state(self.next_jd, pid1)
                        next_lon2, _, _ = calculate_planet_state(self.next_jd, pid2)
                        next_diff = abs(next_lon1 - next_lon2) % 360
                        next_sep = min(next_diff, 360 - next_diff)
                        
                        # Detect if exactness occurs today or tomorrow
                        is_exact_today = (prev_sep - angle) * (sep - angle) < 0 or abs(sep - angle) < 0.05
                        exact_in_days = 0 if is_exact_today else (1 if abs(next_sep - angle) < abs(sep - angle) else -1)
                        
                        events.append({
                            "type": "aspect",
                            "planet1": p1.capitalize(),
                            "planet2": p2.capitalize(),
                            "aspect": aspect_info["name"],
                            "symbol": aspect_info["symbol"],
                            "orb": round(orb, 2),
                            "exact_in_days": exact_in_days,
                            "description": f"{PLANET_SYMBOLS[p1]} {aspect_info['symbol']} {PLANET_SYMBOLS[p2]} (Orb: {orb:.2f}°)" + 
                                           (" (Exact Today)" if exact_in_days == 0 else (f" (Exact in 1 day)" if exact_in_days == 1 else ""))
                        })
                        
        return events

    def get_moon_state(self) -> Dict[str, Any]:
        """Calculates current Moon phase, sign, and perigee/apogee nodes."""
        pos, _ = swe.calc_ut(self.jd, swe.MOON)
        lon = pos[0]
        sign = get_zodiac_sign(lon)
        
        # Calculate Sun-Moon elongation for phase
        sun_pos, _ = swe.calc_ut(self.jd, swe.SUN)
        sun_lon = sun_pos[0]
        elongation = (lon - sun_lon) % 360
        
        # Phase categorization
        if elongation < 6.0 or elongation > 354.0:
            phase = "New Moon"
        elif 84.0 <= elongation <= 96.0:
            phase = "First Quarter"
        elif 174.0 <= elongation <= 186.0:
            phase = "Full Moon"
        elif 264.0 <= elongation <= 276.0:
            phase = "Last Quarter"
        else:
            if 0 < elongation < 90:
                phase = "Waxing Crescent"
            elif 90 < elongation < 180:
                phase = "Waxing Gibbous"
            elif 180 < elongation < 270:
                phase = "Waning Gibbous"
            else:
                phase = "Waning Crescent"

        # Check Perigee / Apogee transitions
        # swisseph.nod_aps_ut computes node/apsides.
        # Index 2 = perihelion/perigee, 3 = aphelion/apogee
        _, _, peri, aphe = swe.nod_aps_ut(self.jd, swe.MOON)
        
        # Determine if perigee/apogee is exact today
        # Check Julian Day distance to today's noon JD
        perigee_today = abs(peri[0] - self.jd) <= 0.5
        apogee_today = abs(aphe[0] - self.jd) <= 0.5

        return {
            "phase": phase,
            "sign": sign,
            "perigee": perigee_today,
            "apogee": apogee_today
        }

    def detect_eclipses(self) -> Optional[Dict[str, Any]]:
        """Search for next solar or lunar eclipse within the next 24 hours."""
        # Find next solar eclipse globally
        # If max eclipse time falls within [jd - 0.5, jd + 0.5] range, it occurs today.
        res_sol, tret_sol = swe.sol_eclipse_when_glob(self.jd)
        sol_jd = tret_sol[0]
        if abs(sol_jd - self.jd) <= 0.5:
            y, m, d, h = swe.revjul(sol_jd)
            # Format hours to UTC time
            hour = int(h)
            minute = int((h - hour) * 60)
            return {
                "type": "Solar Eclipse",
                "date": f"{y}-{m:02d}-{d:02d}",
                "time_utc": f"{hour:02d}:{minute:02d} UTC"
            }
            
        res_lun, tret_lun = swe.lun_eclipse_when(self.jd)
        lun_jd = tret_lun[0]
        if abs(lun_jd - self.jd) <= 0.5:
            y, m, d, h = swe.revjul(lun_jd)
            hour = int(h)
            minute = int((h - hour) * 60)
            return {
                "type": "Lunar Eclipse",
                "date": f"{y}-{m:02d}-{d:02d}",
                "time_utc": f"{hour:02d}:{minute:02d} UTC"
            }
            
        return None

    def detect_seasonal_events(self) -> Optional[str]:
        """Detect Solstice & Equinox transitions."""
        # Solstices/Equinoxes occur when the Sun enters 0, 90, 180, 270 degrees.
        pos, _ = swe.calc_ut(self.jd, swe.SUN)
        lon = pos[0]
        
        prev_pos, _ = swe.calc_ut(self.prev_jd, swe.SUN)
        prev_lon = prev_pos[0]
        
        # Check crossings
        crossings = [
            (0, "Spring Equinox"),
            (90, "Summer Solstice"),
            (180, "Autumn Equinox"),
            (270, "Winter Solstice")
        ]
        for target, name in crossings:
            # Handle wrap around boundary at 0/360
            if target == 0:
                if prev_lon > 350 and lon < 10:
                    return name
            else:
                if prev_lon < target <= lon:
                    return name
        return None

    def run_daily_analysis(self) -> Dict[str, Any]:
        """Assemble complete Astro payload for the day."""
        planetary = self.detect_planetary_events()
        moon = self.get_moon_state()
        eclipse = self.detect_eclipses()
        seasonal = self.detect_seasonal_events()

        return {
            "timestamp": datetime.combine(self.target_date, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
            "planetary_events": planetary,
            "moon": moon,
            "eclipse": eclipse,
            "seasonal": seasonal
        }

# ─────────────────────────────────────────────────────────────
# Text Summaries formatting
# ─────────────────────────────────────────────────────────────

def generate_daily_summary(agent: AstroAgent, payload: Dict[str, Any]) -> str:
    """Format daily Astro Watch Telegram summary."""
    now_local = datetime.now(LOCAL_TZ)
    lines = [
        f"🌌 <b>CELESTIAL RISK WATCH</b>",
        f"<code>{now_local.strftime('%A, %b %d, %Y')} (UTC+7)</code>",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]
    
    # 1. Planetary positions & retrogrades status
    lines.append("🪐 <b>PLANETARY STATUS:</b>")
    for name, pid in PLANETS.items():
        lon, _, retro = calculate_planet_state(agent.jd, pid)
        sign = get_zodiac_sign(lon)
        status_suffix = " (Retrograde)" if retro else ""
        lines.append(f"  • {PLANET_SYMBOLS[name]} in {sign}{status_suffix}")
    
    # Add Mercury rx state specifically
    m_lon, m_speed, m_retro = calculate_planet_state(agent.jd, swe.MERCURY)
    if m_retro:
        lines.append(f"  • {PLANET_SYMBOLS['mercury']} is Retrograde")
        
    # 2. Lunar phase
    m_state = payload["moon"]
    lines.append(f"\n🌕 <b>LUNAR CYCLE:</b>")
    lines.append(f"  • Moon in {m_state['sign']}")
    lines.append(f"  • Phase: <b>{m_state['phase']}</b>")
    if m_state["perigee"]:
        lines.append("  • Moon at Perigee (Exact today)")
    elif m_state["apogee"]:
        lines.append("  • Moon at Apogee (Exact today)")

    # 3. Dynamic planetary crossings / aspects today
    p_events = payload["planetary_events"]
    aspect_lines = []
    station_lines = []
    
    for ev in p_events:
        if ev["type"] == "aspect":
            aspect_lines.append(f"  • {ev['description']}")
        elif ev["type"] in ("station_retrograde", "station_direct", "sign_change"):
            station_lines.append(f"  • {ev['description']}")
            
    if station_lines:
        lines.append(f"\n🚨 <b>ALIGNMENT ALERTS:</b>")
        lines.extend(station_lines)
        
    if aspect_lines:
        lines.append(f"\n📐 <b>EXACT ASPECTS:</b>")
        lines.extend(aspect_lines)

    # 4. Eclipse state
    if payload["eclipse"]:
        e = payload["eclipse"]
        lines.append(f"\n🚨 <b>ECLIPSE WARNING:</b>\n  • {e['type']} occurring at {e['time_utc']} ({e['date']})")
        
    # 5. Seasonal solstices
    if payload["seasonal"]:
        lines.append(f"\n☀️ <b>SEASONAL EQUINOX:</b>\n  • {payload['seasonal']} occurs today")

    lines.append("\n⏰ <i>Calculated dynamically at UTC+7 midnight. No directional bias.</i>")
    return "\n".join(lines)


def generate_weekly_summary(start_date: date) -> str:
    """Scan and build the Monday weekly summary forecast for the upcoming 7 days."""
    lines = [
        f"📅 <b>WEEKLY CELESTIAL FORECAST</b>",
        f"<code>Week of {start_date.strftime('%b %d, %Y')} (UTC+7)</code>",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]
    
    events_found = False
    
    # Scan day-by-day
    for idx in range(7):
        current_date = start_date + timedelta(days=idx)
        agent = AstroAgent(current_date)
        payload = agent.run_daily_analysis()
        
        day_name = current_date.strftime("%A")
        day_header = f"<b>{day_name} ({current_date.strftime('%b %d')}):</b>"
        day_events = []
        
        # Check moon sign entry today
        lon, _, _ = calculate_planet_state(agent.jd, swe.MOON)
        prev_lon, _, _ = calculate_planet_state(agent.prev_jd, swe.MOON)
        sign = get_zodiac_sign(lon)
        prev_sign = get_zodiac_sign(prev_lon)
        if sign != prev_sign:
            day_events.append(f"Moon enters {sign}")
            
        # Check planetary crossings / stations
        for ev in payload["planetary_events"]:
            if ev["type"] in ("retrograde_start", "retrograde_end", "sign_change"):
                day_events.append(ev["description"])
            elif ev["type"] == "aspect" and ev.get("exact_in_days") == 0:
                day_events.append(f"{ev['planet1']} {ev['symbol']} {ev['planet2']}")
                
        # Check eclipse
        if payload["eclipse"]:
            day_events.append(f"🚨 {payload['eclipse']['type']} ({payload['eclipse']['time_utc']})")
            
        # Check seasonal
        if payload["seasonal"]:
            day_events.append(f"☀️ {payload['seasonal']}")
            
        if day_events:
            events_found = True
            lines.append(day_header)
            for de in day_events:
                lines.append(f"  • {de}")
            lines.append("")

    if not events_found:
        lines.append("No major transit changes or exact aspects scheduled for this week.")
        
    lines.append("⏰ <i>All astronomical events are computed deterministically.</i>")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

from src.shared.telegram.client import send_telegram


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Astro notification agent.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-daily", action="store_true")
    parser.add_argument("--force-weekly", action="store_true")
    args = parser.parse_args()

    # Determine local date (UTC+7)
    now_local = datetime.now(LOCAL_TZ)
    target_date = now_local.date()
    
    agent = AstroAgent(target_date)
    payload = agent.run_daily_analysis()
    
    # Save raw JSON state to predictions file or local path
    output_dir = ROOT / "data" / "astro"
    os.makedirs(output_dir, exist_ok=True)
    json_path = output_dir / f"astro_{target_date.isoformat()}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
        
    # Generate daily summary
    daily_msg = generate_daily_summary(agent, payload)
    send_telegram(daily_msg, dry_run=args.dry_run)
    
    # Generate weekly summary if Monday (weekday == 0) or forced
    if args.force_weekly or now_local.weekday() == 0:
        weekly_msg = generate_weekly_summary(target_date)
        send_telegram(weekly_msg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
