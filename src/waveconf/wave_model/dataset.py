"""
dataset.py
----------
DatasetBuilder — assembles the labeled training DataFrame for the TFT.

Wires together every Phase 1 module built so far:

    OHLCV (enriched JSON)
        -> indicators.add_indicators()          [RSI/MACD/ATR/BB/normalization]
        -> ZigZagDetector.run()                 [macro + micro pivots]
        -> StructureTokenizer.run()              [HH/HL/LH/LL/BOS/CHOCH/DIV/FIB_T/SWEEP]
        -> fit_trendline() + PatternDetector     [Track 1: geometric pattern]
        -> ImpulseClassifier / CorrectionClassifier  [Track 2: wave structure]
        -> AstroFeaturesEngine                   [known future]
        -> EconomicCalendarEngine                [known future]
        -> target: close_pct_change (next bar)

Output columns are explicitly partitioned to match pytorch_forecasting's
TimeSeriesDataSet contract directly:

    known_future_columns   -> time_varying_known_reals
    unknown_past_columns   -> time_varying_unknown_reals
    target_column          -> target
    group_id_column        -> group_ids

CLASSIFICATION UPDATE CADENCE (important design decision):
Pattern type (Track 1) and wave structure (Track 2) are NOT recomputed
every bar -- they're recomputed only when a NEW macro pivot is confirmed,
then forward-filled across the bars until the next pivot. This isn't an
optimization shortcut, it reflects what's actually true: the wave count
genuinely doesn't change between pivot confirmations, so recomputing it
every bar would just be re-deriving the same answer from the same inputs.

Usage:
    from src.waveconf.wave_model.dataset import DatasetBuilder

    builder = DatasetBuilder(asset_timeframe="BTC_1D")
    labeled = builder.build('data/pivots/BTC_1D_with_layers.json')

    labeled.df                     # the full DataFrame
    labeled.known_future_columns   # list[str]
    labeled.unknown_past_columns   # list[str]
    labeled.target_column          # 'close_pct_change'
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.waveconf.ingestion.indicators import add_indicators
from src.waveconf.pivots.zigzag import ZigZagDetector
from src.waveconf.pivots.pivot_schema import PivotPoint
from src.waveconf.structure.structure_tokenizer import StructureTokenizer, StructureToken
from src.waveconf.fib_engine.trendline import fit_trendline
from src.waveconf.pivots.pattern_detector import PatternDetector
from src.waveconf.pivots.classifiers import ImpulseClassifier, CorrectionClassifier, ClassificationResult
from src.waveconf.wave_model.astro_features import AstroFeaturesEngine
from src.waveconf.ingestion.economic_calender import EconomicCalendarEngine


# ─────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────

@dataclass
class LabeledDataset:
    df: pd.DataFrame
    known_future_columns: List[str] = field(default_factory=list)
    unknown_past_columns: List[str] = field(default_factory=list)
    static_columns: List[str] = field(default_factory=list)
    target_column: str = "close_pct_change"
    time_idx_column: str = "time_idx"
    group_id_column: str = "asset_timeframe"

    def summary(self) -> str:
        return (
            f"LabeledDataset: {len(self.df)} rows\n"
            f"  known_future   ({len(self.known_future_columns)}): {self.known_future_columns}\n"
            f"  unknown_past   ({len(self.unknown_past_columns)}): {self.unknown_past_columns}\n"
            f"  target         : {self.target_column}\n"
            f"  NaN target rows: {self.df[self.target_column].isna().sum()}"
        )


# ─────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────

class DatasetBuilder:

    def __init__(
        self,
        asset_timeframe: str = "BTC_1D",
        pattern_window_pivots: int = 8,
        astro_config_path: str = "config/astro_features.yaml",
        calendar_config_path: str = "config/economic_calender.yaml",
    ):
        self.asset_timeframe = asset_timeframe
        self.pattern_window_pivots = pattern_window_pivots

        self.pattern_detector = PatternDetector()
        self.impulse_clf = ImpulseClassifier()
        self.correction_clf = CorrectionClassifier()
        self.tokenizer = StructureTokenizer()
        self.astro_engine = AstroFeaturesEngine(astro_config_path)
        self.calendar_engine = EconomicCalendarEngine(calendar_config_path)

        self.pattern_type_map: Dict[str, int] = {"none": 0}
        self.wave_type_map: Dict[str, int] = {"none": 0}

    # ── public API ──────────────────────────────────────────

    def build(self, enriched_json_path: str) -> LabeledDataset:
        df = self._load_enriched(enriched_json_path)
        df = add_indicators(df)
        df = df.reset_index(drop=True)
        df["bar_index"] = df.index

        macro_pivots, micro_pivots = self._detect_pivots(df, enriched_json_path)

        rsi_series = dict(zip(df["bar_index"], df["rsi_14"]))
        macd_hist_series = dict(zip(df["bar_index"], df["macd_hist"]))

        macro_tokens = self.tokenizer.run(macro_pivots, rsi_series, macd_hist_series)
        micro_tokens = self.tokenizer.run(micro_pivots, rsi_series, macd_hist_series)

        df = self._attach_structure_token(df, sorted(macro_tokens + micro_tokens, key=lambda t: t.bar_index))
        df = self._attach_wave_degree(df, macro_pivots)
        df = self._attach_track1_pattern(df, macro_pivots)
        df = self._attach_track2_wave(df, macro_pivots)
        df = self._attach_known_future(df)
        df = self._attach_target(df)

        df["asset_timeframe"] = self.asset_timeframe
        df["time_idx"] = df["bar_index"]

        known_future_columns = [
            "lunar_phase_sin", "lunar_phase_cos", "lunar_anomalistic_normalized",
            "lunar_node_distance", "mercury_retrograde",
            "aspect_jupiter_uranus_intensity", "aspect_mars_uranus_intensity",
            "days_to_fomc", "days_since_last_fomc", "days_to_nfp",
            "high_impact_within_5d", "high_impact_within_2d", "post_event_window",
        ]
        unknown_past_columns = [
            "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
            "rsi_14", "macd_line", "macd_signal", "macd_hist",
            "atr_14_norm", "bb_width",
            "structure_token_id", "wave_degree_id",
            "pattern_type_id", "pattern_confidence",
            "correction_or_impulse_type_id", "wave_match_confidence",
        ]

        missing = [c for c in known_future_columns + unknown_past_columns if c not in df.columns]
        if missing:
            raise RuntimeError(f"DatasetBuilder produced a DataFrame missing expected columns: {missing}")

        return LabeledDataset(
            df=df,
            known_future_columns=known_future_columns,
            unknown_past_columns=unknown_past_columns,
            static_columns=["asset_timeframe"],
            target_column="close_pct_change",
        )

    # ── loading + pivots ─────────────────────────────────────

    def _load_enriched(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Enriched layer file not found: {path}\n"
                f"Run fetch_ohlcv.py then calculate_layers.py first."
            )
        with open(path, "r") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw["data"], columns=raw["columns"])
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        return df

    def _detect_pivots(self, df: pd.DataFrame, source_path: str):
        timeframe = "1D"
        for tf in ["1W", "1D", "4H", "1H"]:
            if f"_{tf}_" in os.path.basename(source_path):
                timeframe = tf
                break

        detector = ZigZagDetector(timeframe=timeframe)
        result = detector.run(df, asset=self.asset_timeframe.split("_")[0])
        return result.macro, result.micro

    # ── attach: structure tokens ─────────────────────────────

    def _attach_structure_token(self, df: pd.DataFrame, tokens: List[StructureToken]) -> pd.DataFrame:
        """
        Forward-filled: the most recently emitted token's token_id holds
        until the next token fires at a later bar_index. Bars before the
        first token get token_id = -1 (no structure established yet).
        """
        df["structure_token_id"] = -1
        for tok in tokens:
            df.loc[df["bar_index"] >= tok.bar_index, "structure_token_id"] = tok.token_id
        return df

    def _attach_wave_degree(self, df: pd.DataFrame, macro_pivots: List[PivotPoint]) -> pd.DataFrame:
        df["wave_degree_id"] = -1
        if not macro_pivots:
            return df
        degree_order = list(type(macro_pivots[0].degree))
        for p in macro_pivots:
            df.loc[df["bar_index"] >= p.bar_index, "wave_degree_id"] = degree_order.index(p.degree)
        return df

    # ── attach: Track 1 (pattern geometry) ───────────────────

    def _attach_track1_pattern(self, df: pd.DataFrame, macro_pivots: List[PivotPoint]) -> pd.DataFrame:
        """
        At each macro pivot confirmation, fit resistance/support trendlines
        over the trailing `pattern_window_pivots` window and classify the
        geometry. Forward-fill the result until the next pivot.
        """
        df["pattern_type_id"] = 0
        df["pattern_confidence"] = 0.0

        for i in range(len(macro_pivots)):
            window = macro_pivots[max(0, i - self.pattern_window_pivots + 1): i + 1]
            if len(window) < 4:
                continue  # need at least 2 highs + 2 lows to attempt a fit

            highs = [p for p in window if p.is_high()]
            lows = [p for p in window if p.is_low()]
            resistance = fit_trendline(highs)
            support = fit_trendline(lows)

            match = self.pattern_detector.detect(resistance, support)
            if match is None or match.confidence == 0.0:
                continue

            if match.pattern_type not in self.pattern_type_map:
                self.pattern_type_map[match.pattern_type] = len(self.pattern_type_map)

            anchor_bar = window[-1].bar_index
            df.loc[df["bar_index"] >= anchor_bar, "pattern_type_id"] = self.pattern_type_map[match.pattern_type]
            df.loc[df["bar_index"] >= anchor_bar, "pattern_confidence"] = match.confidence

        return df

    # ── attach: Track 2 (wave structure) ─────────────────────

    def _attach_track2_wave(self, df: pd.DataFrame, macro_pivots: List[PivotPoint]) -> pd.DataFrame:
        """
        At each macro pivot confirmation, try classifying the trailing
        window as an impulse (exactly 6 pivots) and as a correction
        (4 pivots for ABC, 6 for ABCDE triangle) -- per the hard
        requirements confirmed directly in classifiers.py. Keep whichever
        candidate matched with the highest confidence. Forward-fill.
        """
        df["correction_or_impulse_type_id"] = 0
        df["wave_match_confidence"] = 0.0

        for i in range(len(macro_pivots)):
            candidates: List[ClassificationResult] = []

            if i + 1 >= 6:
                candidates.append(self.impulse_clf.classify(macro_pivots[i - 5: i + 1]))
            if i + 1 >= 4:
                candidates.append(self.correction_clf.classify(macro_pivots[i - 3: i + 1]))
            if i + 1 >= 6:
                candidates.append(self.correction_clf.classify(macro_pivots[i - 5: i + 1]))

            matched = [c for c in candidates if c.matched]
            if not matched:
                continue

            best = max(matched, key=lambda c: c.confidence)
            if best.pattern_type not in self.wave_type_map:
                self.wave_type_map[best.pattern_type] = len(self.wave_type_map)

            anchor_bar = macro_pivots[i].bar_index
            df.loc[df["bar_index"] >= anchor_bar, "correction_or_impulse_type_id"] = self.wave_type_map[best.pattern_type]
            df.loc[df["bar_index"] >= anchor_bar, "wave_match_confidence"] = best.confidence

        return df

    # ── attach: known future (astro + calendar) ──────────────

    def _attach_known_future(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date

        astro_rows = [self.astro_engine.get_daily_features(d).to_flat_dict() for d in dates]
        astro_df = pd.DataFrame(astro_rows, index=df.index)
        for col in ["lunar_phase_sin", "lunar_phase_cos", "lunar_anomalistic_normalized",
                    "lunar_node_distance", "mercury_retrograde"]:
            df[col] = astro_df[col]
        for aspect_col in ["aspect_jupiter_uranus_intensity", "aspect_mars_uranus_intensity"]:
            df[aspect_col] = astro_df[aspect_col] if aspect_col in astro_df.columns else 0.0

        cal_rows = []
        for d in dates:
            ctx = self.calendar_engine.get_context(d)
            cal_rows.append({
                "days_to_fomc": ctx.days_to_fomc if ctx.days_to_fomc is not None else 999,
                "days_since_last_fomc": ctx.days_since_last_fomc if ctx.days_since_last_fomc is not None else 999,
                "days_to_nfp": ctx.days_to_nfp if ctx.days_to_nfp is not None else 999,
                "high_impact_within_5d": int(ctx.high_impact_within_5d),
                "high_impact_within_2d": int(ctx.high_impact_within_2d),
                "post_event_window": int(ctx.post_event_window),
            })
        cal_df = pd.DataFrame(cal_rows, index=df.index)
        for col in cal_df.columns:
            df[col] = cal_df[col]

        return df

    # ── attach: target ────────────────────────────────────────

    def _attach_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df["close_pct_change"] = df["close"].pct_change().shift(-1)
        return df