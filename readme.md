

> **Purpose:** This document is the single source of truth for building an automated Bitcoin Elliott Wave analysis system. Every architectural decision, philosophical rationale, and implementation detail has been deliberated. Do not deviate from the decisions made here without explicit instruction. When in doubt, ask — do not assume.
>
> **v2.0 Changes from v1.0:** Model upgraded from vanilla Transformer to Temporal Fusion Transformer (TFT). Wave correction and impulse rule library fully expanded. Economic calendar integration added to all outputs. Known-future input channel formally defined.

---

## 0. The Vision

The goal is to replicate the analytical mind of a specific human trader — not to build a generic TA bot. The trader uses a **Multi-Layer Confluence Method**: running two completely independent structural analyses on the same price action, then only acting when both analyses converge on the same price zone. The AI must do exactly this, in the same sequence, with the same rules.

The system runs on a **home Linux server (8 GB RAM, 512 GB SSD)**, updated every **6 and 12 hours**, outputting annotated charts, price zones, and confidence scores to a Streamlit dashboard and Telegram alerts. Telegram reports conviction market directions updated each 6 and 12 hours based on moving directions.

The target asset is **Bitcoin (BTC/USDT)** on **daily (1D) and weekly (1W)** timeframes. The system must also maintain **4H candle awareness** as a timing validation layer.

---

## 1. Core Philosophy — Never Trust a Single Count

The entire system is built around one rule:

> **Never act on a single wave count or a single Fibonacci tool. Only treat a zone as actionable when two completely independent analytical paths converge on the same price zone.**

This is not optional. It is the fundamental design constraint. Every component, every output, every confidence score must reflect this dual-validation principle.

Additionally: astrology and planetary cycle data are treated as **behavioral features**, not mystical inputs. If enough market participants trade around lunar phases, Mercury retrograde, or Jupiter-Saturn cycles, those patterns become real in price data. The model learns their statistical weight from historical data rather than from hardcoded belief.

Economic calendar events (FOMC, CPI, NFP) are treated as **known future risk events** — scheduled dates that the market structurally reacts to. They are fed into the TFT's known-future input channel alongside astro features.

---

## 2. System Architecture — Pipeline, Not Monolith

The system is a sequential pipeline of specialized components, not a single end-to-end neural network. This is a deliberate design decision.

### Why a pipeline, not one unified model

The tasks the system performs split into two fundamentally different types:

**Tasks requiring machine learning (pattern recognition in ambiguous sequential data):**
- Identifying the containing chart pattern (ABW, wedge, triangle, channel)
- Classifying impulse and correction structure types (see Section 3 for full taxonomy)
- Probabilistic multi-horizon price forecasting with quantile confidence bands
- Scoring final confluence confidence

**Tasks that are deterministic math given correct inputs:**
- Trendline construction from pivot points (HH, HL, LH, LL)
- Measured move calculation × empirical completion rate
- All Fibonacci extension and retracement calculations
- Invalidation level derivation

Training a neural network to approximate arithmetic is wasteful and less accurate than computing it directly. The ML components handle what ML is actually good at. The math components handle math.

### Full pipeline

```
INPUT LAYER — Three distinct channels fed to TFT:
│
├── OBSERVED PAST (known only after candle closes)
│   OHLCV (1D + 1W + 4H)
│   Technical indicators (RSI, MACD, ATR, BB)
│   Structure tokens (HH/HL/LH/LL/BOS/CHOCH/DIV_H/DIV_L)
│   On-chain metrics (optional enrichment)
│
├── KNOWN FUTURE (calculable for any future date right now)
│   Astro: lunar phase, Bradley score, Mercury retrograde schedule
│   Economic calendar: FOMC dates, CPI release, NFP, PCE, GDP
│   Days to next scheduled high-impact event
│
└── STATIC METADATA
    Asset: BTC, Timeframe: 1D/1W
    │
    ▼
[1] PIVOT DETECTOR              ← rule-based, ZigZag algorithm
    Output: swing high/low list with timestamps and prices
    │
    ▼
[2] STRUCTURE TOKENIZER         ← rule-based conversion
    Input: pivot list
    Output: token sequence (HH/HL/LH/LL/BOS/CHOCH/W3_EXT/DIV_H/DIV_L/FIB_T/ABW_T/SWEEP)
    Each token carries: price, timestamp, degree, fib_context, volume_surge flag
    │
    ▼
[3] TEMPORAL FUSION TRANSFORMER ← ML model, replaces vanilla transformer
    Inputs: all three channels above + structure token sequence
    Outputs:
      A: macro pattern type (ABW/wedge/triangle/channel) + confidence
      B: wave structure type (impulse/correction subtype) + confidence
      C: macro wave position (A/B/C/D or 1-5 at primary degree)
      D: micro wave position (sub-wave at intermediate degree)
      E: b_breach_expected (bool, derived from correction subtype)
      F: quantile price forecasts q10/q50/q90 at t+7/14/30/60 days
    │
    ▼
[4] FIBONACCI ENGINE            ← pure math module, no ML
    Input: pivot list + wave labels from [3]
    Computes all price targets and invalidation levels (see Section 6)
    │
    ▼
[5] CONFLUENCE SCORER           ← rule-based + statistical
    Compares TFT quantile bands against Fibonacci price targets
    Confluence = TFT q50 overlaps Fibonacci cluster within 2%
    Outputs: scenario A, scenario B, probability per scenario, cluster strength
    │
    ▼
[6] ECONOMIC CALENDAR OVERLAY   ← deterministic, no ML
    Fetches next scheduled high-impact events within forecast horizon
    Flags if FOMC/CPI/NFP falls inside or near a confluence zone
    Risk-adjusts confidence score if major event precedes zone by <5 days
    │
    ▼
OUTPUT: structured forecast record → SQLite → Streamlit + Telegram
```

---

## 3. Complete Wave Structure Taxonomy

This is the full rule library the system must know. The classifier must be able to identify ALL types listed here, not only the flat correction seen in the reference chart. Every type has appeared in BTC history.

### 3A — Five-Wave Impulse Structures

All impulse structures share three inviolable hard rules:
- Wave 2 never retraces more than 100% of Wave 1
- Wave 3 is never the shortest among waves 1, 3, and 5
- Wave 4 never overlaps Wave 1's price territory (in a non-diagonal)

| Type | Description | Key Diagnostic |
|---|---|---|
| Standard Impulse | Classic 1-2-3-4-5, no extensions | Wave 3 = 1.618× Wave 1, Wave 4 alternates with Wave 2 |
| Wave 3 Extension | Wave 3 > 1.618× and subdivides into 9 waves | Most common extension; explosive momentum |
| Wave 5 Extension | Wave 5 > Wave 3; Wave 3 ≈ Wave 1 | Common at market tops; often shows divergence |
| Wave 1 Extension | Wave 1 > 1.618× subsequent waves | Less common; first waves of major new trends |
| Leading Diagonal | Waves 1-2-3-4-5 overlap in wedge shape | Appears as Wave 1 or Wave A only; sub-waves are 3-3-3-3-3 |
| Ending Diagonal | Terminal wedge, all sub-waves overlap | Appears as Wave 5 or Wave C only; signals exhaustion; 3-3-3-3-3 |

**Ending Diagonal special rule:** Wave 4 ALWAYS overlaps Wave 1. This is the only impulse structure where overlap is expected, not an invalidation. The system must not flag this overlap as an error when pattern_type = `ending_diagonal`.

### 3B — Three-Wave Correction Structures (Flat Family)

**Regular Flat (3-3-5)**
- Wave A: 3 waves down
- Wave B: 3 waves up, retraces 81–100% of Wave A
- Wave C: 5 waves down, terminates near Wave A's end
- b_breach_expected: False (B barely reaches A's start)
- Key rule: Wave C ≈ 1.0× Wave A in length

**Expanded Flat / Irregular Flat (3-3-5)**
- Wave A: 3 waves down
- Wave B: 3 waves up, **exceeds** Wave A's starting point (>100%, typically 105–138%)
- Wave C: 5 waves down, **exceeds** Wave A's end (typically 1.236–1.618× Wave A)
- b_breach_expected: True (B breaks above Wave A origin)
- Key rule: Most aggressive flat; C often = 1.618× A
- This is the type identified in the reference BTC chart analysis

**Running Flat (3-3-5)**
- Wave A: 3 waves down
- Wave B: 3 waves up, exceeds Wave A's start significantly
- Wave C: 5 waves down, **fails** to reach Wave A's end (truncated)
- b_breach_expected: True
- Key rule: Very bullish context; C truncation signals strong underlying demand
- Rare; requires confirmation from higher-degree structure

### 3C — Three-Wave Correction Structures (Zigzag Family)

**Single Zigzag (5-3-5)**
- Wave A: 5 waves (impulse)
- Wave B: 3 waves, retraces 38.2–78.6% of Wave A (rarely exceeds 78.6%)
- Wave C: 5 waves (impulse), typically equals Wave A or = 0.618×/1.618× Wave A
- b_breach_expected: False
- Key rule: Sharp, clean correction; deepest of all correction types

**Double Zigzag (W-X-Y)**
- Two zigzags connected by a 3-wave X connector
- X wave retraces 38.2–78.6% of W wave
- Y wave usually equals W wave in length or is 0.618× W
- b_breach_expected: False
- Key rule: Appears as a deep, prolonged correction; often mistaken for impulse

**Triple Zigzag (W-X-Y-X-Z)**
- Three zigzags connected by two X waves
- Extremely rare; forms only in strong trending markets
- Key rule: If you think you see this, re-examine — it is usually a different structure

### 3D — Three-Wave Correction Structures (Triangle Family)

All triangles subdivide into five 3-wave legs (3-3-3-3-3). They appear as Wave 4 of an impulse, Wave B of a zigzag/flat, or Wave X connectors.

**Contracting Symmetrical Triangle**
- Both trendlines converge toward a point (apex)
- Breakout in direction of the prior trend (continuation pattern)
- Wave E typically retraces 0.618× Wave D
- Key rule: Volume contracts during formation; expands on breakout

**Contracting Ascending Triangle**
- Flat upper resistance line + rising lower support line
- Bullish bias; usually resolves upward
- Key rule: Each successive low is higher; buyers absorbing at flat resistance

**Contracting Descending Triangle**
- Declining upper resistance line + flat lower support line
- Bearish bias; usually resolves downward

**Expanding Triangle (Reverse)**
- Both trendlines diverge
- Rare; typically appears in Wave B positions
- Key rule: Each successive swing is larger; signals indecision at major structural levels

### 3E — Complex Correction Structures (Combination Family)

**Double Three (W-X-Y)**
- W: any simple correction (flat, zigzag, or triangle)
- X: 3-wave connector, retraces 38.2–78.6% of W
- Y: any simple correction (cannot be same type as W if W is zigzag)
- Key rule: Sideways to slightly directional; time-consuming correction
- b_breach_expected: depends on Y type

**Triple Three (W-X-Y-X-Z)**
- Three simple corrections connected by two X waves
- Very rare; extremely time-consuming
- Key rule: If market appears to be doing nothing for extended period, examine for this structure

### 3F — Elliott Wave Degree Hierarchy

The system must track wave position at multiple degrees simultaneously:

```
Supercycle    → Roman numerals in circles: (I)(II)(III)(IV)(V)
Cycle         → Roman numerals: I II III IV V
Primary       → Numbers in circles: (1)(2)(3)(4)(5)
Intermediate  → Numbers in parentheses: (1)(2)(3)(4)(5)
Minor         → Plain numbers: 1 2 3 4 5
Minute        → Lowercase: i ii iii iv v
```

The weekly (1W) timeframe covers Primary and Cycle degree.
The daily (1D) timeframe covers Intermediate and Minor degree.
The 4H timeframe covers Minute degree (used for timing/invalidation only).

---

## 4. The Temporal Fusion Transformer — ML Model Specification

### Why TFT replaces the vanilla Transformer

The system's actual goal is **probabilistic multi-scenario price forecasting** — not wave position classification. The wave and pattern labels are intermediate representations that inform the forecast, not the final output. TFT was purpose-built for exactly this:

- Outputs **quantile forecast bands** (q10/q50/q90) at multiple future horizons simultaneously — this is the confidence rate output
- Has a **dedicated known-future input channel** — astro features and economic calendar dates are calculable for any future date right now; TFT exploits this as privileged information that a vanilla transformer cannot
- Contains an internal **Variable Selection Network** that learns which features matter most per market regime — automatically de-weights irrelevant inputs
- Produces an **interpretable attention output** showing which past timesteps most influenced each prediction (useful for debugging and for explaining why the model reached a conclusion)

### Architecture specification

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Dataset construction
dataset = TimeSeriesDataSet(
    df,
    time_idx            = "time_idx",
    target              = "close_pct_change",   # normalized % change, not raw price
    group_ids           = ["asset"],            # "BTC_1D" or "BTC_1W"

    # KNOWN FUTURE inputs — TFT's key advantage
    # These are calculable for any future date right now
    time_varying_known_reals = [
        "lunar_phase_sin",          # sin of lunar phase angle
        "lunar_phase_cos",          # cos of lunar phase angle
        "bradley_score",            # normalized -1 to +1
        "mercury_retrograde",       # 0 or 1
        "days_to_new_moon",         # integer, 0–29
        "days_to_fomc",             # integer, days until next FOMC
        "days_to_cpi",              # integer, days until next CPI release
        "days_to_nfp",              # integer, days until next NFP
        "fomc_week_flag",           # 1 if current week contains FOMC
        "high_impact_event_flag",   # 1 if any high-impact event within 3 days
    ],

    # OBSERVED PAST inputs — only known after candle closes
    time_varying_unknown_reals = [
        "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "atr_14_norm", "bb_width",
        "fib_382", "fib_618", "fib_786",
        "bars_since_last_high", "bars_since_last_low",
        "pivot_high_magnitude", "pivot_low_magnitude",
        "structure_token_id",       # encoded integer from token vocabulary
        "pattern_type_id",          # encoded integer: ABW/wedge/triangle/etc
        "correction_type_id",       # encoded integer: flat/zigzag/triangle/combo
        "wave_degree_id",           # encoded integer: primary/intermediate/minor
    ],

    # STATIC metadata
    static_categoricals = ["asset_timeframe"],  # "BTC_1D", "BTC_1W"

    max_encoder_length  = 90,   # 90-candle lookback
    max_prediction_length = 60, # forecast 60 days forward (covers 2 FOMC cycles)
)

model = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate       = 1e-3,
    hidden_size         = 32,           # small for CPU inference on home server
    attention_head_size = 2,
    dropout             = 0.1,
    hidden_continuous_size = 16,
    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    log_interval        = 10,
    reduce_on_plateau_patience = 4,
)
```

### TFT output interpretation

```
At each inference cycle, TFT outputs:

Horizon    q10 (pessimistic)   q50 (median)   q90 (optimistic)
t+7d       $XX,XXX             $XX,XXX        $XX,XXX
t+14d      $XX,XXX             $XX,XXX        $XX,XXX
t+30d      $XX,XXX             $XX,XXX        $XX,XXX
t+60d      $XX,XXX             $XX,XXX        $XX,XXX

These bands feed into the Confluence Scorer (pipeline step [5]).
Confluence = TFT q50 at any horizon overlaps Fibonacci cluster within 2%.
Scenario probability = percentage of TFT probability mass inside Fibonacci zone.
```

### Training vs inference separation

- **Train** on external machine (Google Colab T4 free tier, Kaggle P100, or any CUDA GPU). BTC daily dataset ~5,000 rows trains in 15–25 minutes on T4.
- **Deploy** frozen `.pt` weights to home server. TFT inference on CPU uses ~300–500 MB RAM, well within 8 GB constraint.
- **Retrain cadence:** monthly, or when 30-day rolling MAPE on q50 forecasts exceeds 15%. Retrain on external machine, drop new `.pt` file into server — hot-reload picks it up automatically.

---

## 5. Structure Token Vocabulary

This is the extended token vocabulary for the Structure Tokenizer (pipeline step [2]). Standard SMC tokens are extended with Elliott Wave-specific tokens.

```python
STRUCTURE_TOKENS = {
    # Directional structure — standard
    'HH':     0,   # Higher High: impulse continuation
    'HL':     1,   # Higher Low: corrective floor in uptrend
    'LH':     2,   # Lower High: corrective ceiling in downtrend
    'LL':     3,   # Lower Low: impulse continuation downward

    # Structural events — standard
    'BOS':    4,   # Break of Structure: prior swing broken, continuation confirmed
    'CHOCH':  5,   # Change of Character: first opposing BOS, potential reversal

    # Elliott Wave specific extensions
    'W3_EXT': 6,   # Wave 3 extension: HH with volume surge + momentum acceleration
    'W4_REJ': 7,   # Wave 4 rejection test: price approached W1 territory, held
    'DIV_H':  8,   # Divergence High: price HH but RSI/MACD lower — structural top signal
    'DIV_L':  9,   # Divergence Low: price LL but RSI/MACD higher — structural bottom signal
    'FIB_T':  10,  # Fibonacci Tag: price touched a key Fib level (38.2/61.8/78.6/161.8)
    'ABW_T':  11,  # ABW Trendline Touch: price touched upper or lower ABW boundary
    'SWEEP':  12,  # Liquidity Sweep: fast pierce of prior swing, immediate reversal
                   # In flat correction context, this is the B-breach event
    'DIAG':   13,  # Diagonal signal: overlapping waves detected (leading or ending)
    'TRUNC':  14,  # Truncation: Wave 5 fails to exceed Wave 3 (bearish for impulse)
}

# Each token record carries metadata
token_record = {
    'token':         'DIV_H',
    'price':          82720,
    'timestamp':      ...,
    'degree':         'intermediate',  # supercycle/cycle/primary/intermediate/minor
    'fib_context':    0.618,           # nearest Fib ratio at this pivot (or None)
    'volume_surge':   False,           # True if volume > 1.5× 20-period average
    'rsi_value':      62.3,            # RSI at time of pivot (for divergence tokens)
    'macd_hist':      -0.0023,         # MACD histogram at pivot
}
```

---

## 6. Labeled Dataset Schema

This is the output of the `DualCountLabelGenerator` (Phase 1). Each row is one timestep (one daily candle). This CSV trains the TFT model.

```
# Identifiers
timestamp               datetime
time_idx                int          monotonically increasing integer (required by TFT)
asset_timeframe         str          "BTC_1D" or "BTC_1W"

# OHLCV (normalized)
open_norm, high_norm, low_norm, close_norm   float   % change from 20-SMA
close_pct_change        float        target variable for TFT (% change next candle)
volume_norm             float

# Momentum indicators
rsi_14                  float        normalized 0–1
macd_line               float        normalized by ATR
macd_signal             float        normalized by ATR
macd_hist               float        normalized by ATR

# Volatility
atr_14_norm             float        ATR / close
bb_width                float        Bollinger Band width / close

# Fibonacci context
fib_382                 float        distance to nearest 38.2% level / ATR
fib_618                 float        distance to nearest 61.8% level / ATR
fib_786                 float        distance to nearest 78.6% level / ATR

# Pivot / structure context
bars_since_last_high    float        normalized by sequence length
bars_since_last_low     float        normalized by sequence length
pivot_high_magnitude    float        size of last swing high / ATR
pivot_low_magnitude     float        size of last swing low / ATR
structure_token_id      int          encoded from STRUCTURE_TOKENS vocabulary
pattern_type_id         int          ABW=0 / rising_wedge=1 / falling_wedge=2 / etc
correction_type_id      int          flat_regular=0 / flat_expanded=1 / flat_running=2 /
                                     zigzag=3 / dbl_zigzag=4 / tri_zigzag=5 /
                                     triangle_sym=6 / triangle_asc=7 / triangle_desc=8 /
                                     triangle_exp=9 / double_three=10 / triple_three=11
impulse_type_id         int          standard=0 / w3_ext=1 / w5_ext=2 / w1_ext=3 /
                                     leading_diag=4 / ending_diag=5
wave_degree_id          int          supercycle=0 / cycle=1 / primary=2 /
                                     intermediate=3 / minor=4 / minute=5

# KNOWN FUTURE — astro (all calculable for any future date)
lunar_phase_sin         float        sin(lunar_phase_degrees)
lunar_phase_cos         float        cos(lunar_phase_degrees)
bradley_score           float        normalized -1 to +1
mercury_retrograde      int          0 or 1
days_to_new_moon        int          0–29
jupiter_saturn_aspect   float        angular separation in degrees

# KNOWN FUTURE — economic calendar (all scheduled in advance)
days_to_fomc            int          days until next FOMC meeting
days_to_cpi             int          days until next CPI release
days_to_nfp             int          days until next Non-Farm Payrolls
days_to_pce             int          days until next PCE release
days_to_gdp             int          days until next GDP print
fomc_week_flag          int          1 if current week contains FOMC
high_impact_event_flag  int          1 if any high-impact event within 3 days
post_fomc_flag          int          1 if within 2 days after FOMC (vol decay period)

# Track 1 labels (macro harmonic structure)
macro_wave              str          A/B/C/D/1/2/3/4/5/none
pattern_type            str          ABW/rising_wedge/falling_wedge/triangle/channel/none
pattern_confidence      float        0.0–1.0

# Track 2 labels (pure Elliott re-count)
ew_wave                 str          (A)/(B)/(C)/1/2/3/4/5/none
ew_sub_wave             str          a/b/c/d/e/none
correction_type         str          full string label (e.g. flat_expanded_335)
impulse_type            str          full string label (e.g. impulse_w3_extension)
b_breach_expected       int          0 or 1
diagonal_overlap_ok     int          1 if ending_diag or leading_diag (overlap is valid)

# Confluence ground truth
confluence_valid        int          0 or 1
cluster_zone_upper      float        NaN if not valid
cluster_zone_lower      float        NaN if not valid
cluster_strength        float        0.0–1.0, NaN if not valid
scenario_a_prob         float        probability of Scenario A (first zone)
scenario_b_prob         float        probability of Scenario B (main zone)
invalidation_level      float        C_top + 0.5% buffer
```

### Labeling conflict resolution
1. Score each competing count by Fibonacci SSE (sum of squared errors vs expected ratios)
2. Prefer higher-degree cycles when scores within 10% margin
3. If truly ambiguous: `confluence_valid = 0`, exclude from high-confidence training subset

---

## 7. Fibonacci Engine — Exact Rules

```python
FIBONACCI_RULES = {
    # Impulse wave ratios
    'wave2_retrace':           (0.382, 0.786),   # of wave 1, hard upper = 1.0
    'wave3_extend':            (1.000, 4.236),   # 1.0 is HARD MINIMUM, not soft
    'wave4_retrace':           (0.236, 0.500),   # of wave 3
    'wave5_extend':            (0.382, 1.618),   # of wave 1
    'wave5_equal_wave1':        1.000,            # most common wave 5 target

    # Correction wave ratios
    'waveA_retrace':           (0.382, 0.618),   # of prior impulse
    'waveB_regular_retrace':   (0.810, 1.000),   # regular flat B
    'waveB_expanded_retrace':  (1.000, 1.382),   # expanded flat B
    'waveB_running_retrace':   (1.100, 1.382),   # running flat B (exceeds A significantly)
    'waveB_zigzag_retrace':    (0.382, 0.786),   # zigzag B (shallow)
    'waveC_regular':           (0.618, 1.000),   # of wave A (regular flat)
    'waveC_expanded':          (1.236, 1.618),   # of wave A (expanded flat)
    'waveC_zigzag':            (0.618, 1.618),   # of wave A (zigzag)

    # Cluster Fibonacci tools
    'fib_cluster_ext_A':        2.618,            # extension from C top (tool A)
    'fib_cluster_ext_B':        1.618,            # extension from B→C range (tool B)

    # Pattern empirical rates
    'abw_empirical_rate':       0.70,             # broadening wedge: 70% of theoretical
    'wedge_empirical_rate':     0.85,             # standard wedge: 85% of theoretical
    'triangle_empirical_rate':  1.00,             # triangle: full measured move typical

    # System thresholds
    'tolerance':                0.05,             # ±5% soft tolerance on all bounds
    'cluster_threshold':        0.02,             # 2% proximity = valid cluster
    'invalidation_buffer':      0.005,            # 0.5% above C top
}

# HARD INVALIDATION RULES — no tolerance applies
HARD_RULES = {
    'wave3_min':        'wave3 MUST be longer than wave1 in price range',
    'wave2_max':        'wave2 MUST NOT retrace more than 100% of wave1',
    'wave4_no_overlap': 'wave4 MUST NOT enter wave1 price territory (non-diagonal)',
    'wave4_diag_ok':    'wave4 MAY overlap wave1 if pattern = ending_diagonal or leading_diagonal',
    'wave5_truncation': 'flag TRUNC token if wave5 fails to exceed wave3 high — valid but rare',
}
```

---

## 8. Economic Calendar Module

### Data source
Use the `investpy` library or `pandas_datareader` for historical economic events. For future dates, maintain a local JSON of scheduled FOMC/CPI/NFP dates (published by the Fed and BLS 12 months in advance).

```python
# high_impact_events.json — update annually
{
  "FOMC": [
    "2026-01-29", "2026-03-19", "2026-05-07",
    "2026-06-18", "2026-07-30", "2026-09-17",
    "2026-11-05", "2026-12-17"
  ],
  "CPI": [
    "2026-01-15", "2026-02-12", "2026-03-12",
    ...
  ],
  "NFP": [
    "2026-01-09", "2026-02-06", "2026-03-06",
    ...
  ],
  "PCE": [...],
  "GDP": [...]
}
```

### Risk-adjustment rule
If a high-impact event (FOMC, CPI, NFP) falls **within 5 days before** a confluence entry zone, apply a confidence penalty:

```python
def adjust_for_calendar_risk(confluence_strength, days_to_event):
    if days_to_event <= 2:
        return confluence_strength * 0.60   # major risk, strong penalty
    elif days_to_event <= 5:
        return confluence_strength * 0.80   # moderate risk
    else:
        return confluence_strength           # no adjustment
```

Post-FOMC (within 2 days after): apply 1.10× boost to confluence strength — volatility has resolved, direction is clearer.

---

## 9. Scheduling Architecture

```python
# Job 1: Fast invalidation check — 4H candles every 6 hours
scheduler.add_job(check_invalidation_4h,   'interval', hours=6)

# Job 2: Full TFT inference — daily candles, twice per day
scheduler.add_job(run_full_pipeline_daily, 'cron', hour='0,12')

# Job 3: Weekly context update — Monday 01:00 UTC
scheduler.add_job(update_weekly_context,   'cron', day_of_week='mon', hour=1)

# Job 4: Economic calendar sync — daily, pulls next 90 days of events
scheduler.add_job(sync_economic_calendar,  'cron', hour=2)

# Job 5: Model hot-reload check — every 30 minutes
scheduler.add_job(maybe_reload_model,      'interval', minutes=30)
```

### Timeframe hierarchy
```
Weekly (1W)  → Cycle / Primary degree     → updates Monday 01:00 UTC
Daily  (1D)  → Intermediate / Minor       → full inference 00:00 + 12:00 UTC
4H           → Minor / Minute (timing)    → invalidation check every 6 hours
```

---

## 10. Data Sources

**Primary OHLCV:** Binance via CCXT — no API key needed, deepest BTC liquidity, longest clean history. Weekly candles open Monday 00:00 UTC (do NOT use Yahoo Finance — Sunday open misaligns wave counts).

**Extended history (pre-2017):** CryptoCompare free tier — daily data back to 2010 for supercycle training context.

**Astro data:** `pyswisseph` (Swiss Ephemeris Python wrapper) — free, highly accurate.

**Economic calendar:** Local JSON of scheduled dates + `investpy` for historical actual vs forecast values. Update JSON annually when Fed/BLS publish the year's schedule.

**On-chain metrics (optional):** Glassnode free tier or CryptoQuant. Not required for wave analysis core.

**Local storage:** SQLite (prediction records, signal history) + Parquet (OHLCV cache). Never re-fetch stored data. Incremental updates only.

---

## 11. Output Schema

### SQLite predictions table

```sql
CREATE TABLE predictions (
    id                      INTEGER PRIMARY KEY,
    timestamp               DATETIME,
    timeframe               TEXT,

    -- TFT quantile forecasts
    q10_7d    REAL,  q50_7d    REAL,  q90_7d    REAL,
    q10_14d   REAL,  q50_14d   REAL,  q90_14d   REAL,
    q10_30d   REAL,  q50_30d   REAL,  q90_30d   REAL,
    q10_60d   REAL,  q50_60d   REAL,  q90_60d   REAL,

    -- Wave structure (Track 1)
    pattern_type            TEXT,
    macro_wave              TEXT,
    micro_wave              TEXT,

    -- Wave structure (Track 2)
    correction_type         TEXT,
    impulse_type            TEXT,
    ew_wave                 TEXT,
    b_breach_expected       INTEGER,
    diagonal_overlap_ok     INTEGER,

    -- Fibonacci outputs
    abw_target              REAL,
    fib_target_a            REAL,
    fib_target_b            REAL,
    invalidation_level      REAL,

    -- Confluence
    confluence_valid        INTEGER,
    cluster_upper           REAL,
    cluster_lower           REAL,
    cluster_strength        REAL,
    cluster_strength_adj    REAL,   -- after calendar risk adjustment

    -- Scenarios
    scenario_a_zone_upper   REAL,
    scenario_a_zone_lower   REAL,
    scenario_a_prob         REAL,
    scenario_b_zone_upper   REAL,
    scenario_b_zone_lower   REAL,
    scenario_b_prob         REAL,

    -- Entry plan
    entry_zone_1            REAL,   -- 10-20% position
    entry_zone_2            REAL,   -- full position

    -- Astro context
    lunar_phase_deg         REAL,
    bradley_score           REAL,
    mercury_retrograde      INTEGER,
    next_astro_event        TEXT,
    days_to_astro_event     INTEGER,

    -- Economic calendar context
    next_fomc_date          TEXT,
    days_to_fomc            INTEGER,
    next_cpi_date           TEXT,
    days_to_cpi             INTEGER,
    next_nfp_date           TEXT,
    days_to_nfp             INTEGER,
    next_pce_date           TEXT,
    days_to_pce             INTEGER,
    high_impact_within_5d   INTEGER,    -- 0 or 1
    calendar_risk_flag      TEXT,       -- "FOMC in 3 days", "CPI tomorrow", etc

    -- Accuracy tracking (filled retrospectively)
    btc_close_at_signal     REAL,
    actual_outcome          TEXT,
    prediction_correct      INTEGER
);
```

### Telegram alert format

Sent when `confluence_valid = 1` (new signal), when `confluence_valid` flips 1→0 (invalidated), or when scenario probability shifts >10% from prior update.

```
🔍 BTC ELLIOTT WAVE FORECAST [1D]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 STRUCTURE
Track 1:  Ascending Broadening Wedge
          Macro wave C complete → D projected
Track 2:  Expanded Flat 3-3-5
          B-breach expected (liquidity sweep, not invalidation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ CONFLUENCE CONFIRMED (strength: 0.89 → adj: 0.71)
⚠️  Adjusted: FOMC in 3 days

SCENARIO A   $63,800–$63,900   [31% prob]  → 10-20% entry
SCENARIO B   $59,230–$61,000   [48% prob]  → full entry
COMBINED     79% one scenario plays out

TFT BANDS (30-day)
  Optimistic  q90: $71,000
  Median      q50: $60,100  ← inside Scenario B ✅
  Pessimistic q10: $51,000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ INVALIDATED above: $83,200
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌙 ASTRO
  Lunar: 287° (waning gibbous)
  Bradley score: -0.42 (bearish pressure)
  Mercury: direct (4 days ago)
  Next event: New Moon in 6 days

📅 ECONOMIC CALENDAR (next 30 days)
  ⚠️  FOMC Meeting      Jun 18  (3 days)  HIGH IMPACT
  📊  CPI Release       Jun 11  (PASSED)
  📊  NFP               Jul 4   (16 days)
  📊  PCE               Jun 27  (12 days)
  📊  GDP Q1 Final      Jun 26  (11 days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 DELTA FROM LAST UPDATE (12h ago)
  Scenario B prob:    45% → 48%  ▲ tightening
  Cluster strength:   0.87 → 0.89 ▲
  FOMC days:          4 → 3  ⚠️  approaching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Next update: 12:00 UTC
```

---

## 12. Phase Implementation Roadmap

### Phase 1 — Rule-Based Pipeline (Weeks 1–3)
All deterministic components. No ML involved yet.

**Deliverables:**
- `PivotDetector.py` — ZigZag swing H/L detection, configurable threshold
- `StructureTokenizer.py` — converts pivot list to extended token vocabulary (Section 5)
- `WaveRules.py` — complete Fibonacci validation rules (Section 7), all wave types from Section 3
- `PatternDetector.py` — geometric pattern recognition (ABW, wedge, triangle, channel)
- `CorrectionClassifier.py` — all correction subtypes: flat (3 variants), zigzag (3 variants), triangle (4 variants), combination (2 variants)
- `ImpulseClassifier.py` — all impulse subtypes: standard, 3 extension types, leading diagonal, ending diagonal
- `TrendlineBuilder.py` — linear regression trendlines from pivot subsets
- `FibonacciEngine.py` — all Fibonacci calculations per Section 7
- `ConfluenceChecker.py` — cluster validation, scenario generation
- `EconomicCalendar.py` — loads local JSON, computes days-to-event features, applies calendar risk adjustment
- `DualCountLabelGenerator.py` — produces full labeled CSV per Section 6
- `IncrementalChecker.py` — candle-by-candle invalidation check per structure type rules
- Unit tests for all modules

**Verification:** Run on `btc-usd_1d.csv`. Cross-check confluence zones against reference analysis (Target 1: ~$63,850, Main Target: ~$59,230–61,000).

### Phase 2 — TFT Model (Weeks 4–8, train on external machine)
Build and train TFT using labeled CSV from Phase 1.

**Deliverables:**
- `FeatureBuilder.py` — constructs all features, separates known-future from observed-past
- `AstroFeatures.py` — PySwisseph integration, 6 astro columns
- `CalendarFeatures.py` — economic calendar feature columns from JSON
- `TFTModel.py` — model architecture per Section 4
- `train.py` — training loop, checkpointing, export to `.pt`
- `evaluate.py` — MAPE per quantile, scenario accuracy metrics

**Training target:** Google Colab T4 (free) or Kaggle P100. Export `wave_model.pt` for deployment.

### Phase 3 — Scheduler + Deployment (Weeks 8–10)
Wire all components into the live system on the home server.

**Deliverables:**
- `scheduler.py` — APScheduler with 5 jobs (Section 9)
- `pipeline.py` — orchestrates the full 9-step process
- `database.py` — SQLite read/write
- `dashboard.py` — Streamlit: annotated BTC chart, quantile bands, current zones, astro panel, economic calendar panel
- `alerts.py` — Telegram bot with full format from Section 11
- `model_watcher.py` — hot-reload on `.pt` file change

### Phase 4 — Accuracy Tracking + Retraining Loop (Week 10+)
- Retrospective outcome labeler (fills `actual_outcome` in SQLite)
- Monthly retraining script on external machine
- Accuracy dashboard in Streamlit: MAPE per horizon, scenario hit rate, astro correlation stats, calendar event impact analysis

---

## 13. Key Decisions Log

Do not relitigate these decisions without explicit instruction.

| Decision | Choice | Rationale |
|---|---|---|
| Core ML model | TFT, not vanilla Transformer | Goal is probabilistic forecasting, not classification; TFT outputs quantile bands natively; known-future channel for astro + calendar is a genuine structural advantage |
| Wave/pattern labels | Intermediate inputs to TFT | Classification is the means, not the end; TFT learns wave context as internal representation |
| Architecture | Pipeline, not monolith | Math is math; separate from ML for debuggability and interpretability |
| Training machine | External (Colab/Kaggle) | ~5,000 rows trains in 15-25 min on T4; home server inference only |
| Inference RAM | ~300–500 MB (TFT on CPU) | model.eval() + torch.no_grad(); well within 8 GB |
| Forecast horizons | t+7/14/30/60 days | Covers 2 FOMC cycles; matches daily + weekly wave degree timing |
| Astro treatment | Known-future TFT channel | Planetary positions are deterministic future knowledge; TFT exploits this; vanilla transformer cannot |
| Economic calendar | Known-future TFT channel + calendar risk adjustment | FOMC/CPI/NFP are scheduled; known in advance; structurally impact BTC volatility |
| Calendar risk rule | Confidence × 0.60–0.80 within 5 days of event | High-impact events create unpredictable volatility that overrides wave structure temporarily |
| Post-FOMC boost | Confidence × 1.10 within 2 days after | Volatility resolved; directional clarity improves |
| Primary OHLCV source | Binance via CCXT, Monday 00:00 UTC weekly | Deepest liquidity; TradingView-aligned weekly candles; do NOT use Yahoo Finance |
| Fibonacci tolerance | ±5% soft, wave 3 min = HARD, wave 2 max = HARD | Wave 3 shortest and wave 2 >100% are structural violations, not measurement noise |
| Cluster threshold | 2% price proximity | Tighter than traditional TA; crypto allows precision given 24/7 market |
| B-wave breach | Expected in expanded flat and running flat; not invalidation | Confirmed by flat correction type classifier; flag as potential liquidity sweep |
| Diagonal overlap | Valid ONLY in leading/ending diagonal types | System must check `diagonal_overlap_ok` flag before applying wave 4 overlap rule |
| Token vocabulary | Extended 15-token set including Elliott-specific tokens | Standard 6-token SMC vocabulary insufficient for wave degree and divergence detection |
| Wave taxonomy | Full 15 structure types (6 impulse + 9 correction) | BTC history contains all types; limiting to flat only creates blind spots |
| Scenario output | Two scenarios with individual probabilities | Mirrors trader's mental model: partial entry at first zone, full at main zone |
| Alert trigger | Confluence valid + >10% probability shift | Reduces noise; only sends when something meaningful changes |
| Telegram format | Includes economic calendar and delta from last update | Gives trader immediate context on risk events and conviction direction changes |

---

## 14. What This System Is Not

- It is **not** a fully automated trading bot. It generates analysis and probability-weighted zones. A human makes the final trade decision.
- It is **not** a universal market analyzer. Calibrated for Bitcoin on daily and weekly timeframes. Other assets or timeframes require retraining.
- It is **not** a certainty engine. The confluence method narrows probability — it does not eliminate risk. Every output acknowledges this explicitly.
- It does **not** replace liquidation/leverage confirmation. That requires real-time order book data outside this system's scope, flagged as human review in every signal.
- It does **not** trade FOMC week autonomously. Calendar risk adjustment reduces confidence — the human must decide whether to act despite the reduced score.

