"""
fusion.py
---------
Fusion Layer — merges deterministic engine output with TFT AI forecast.

Target Architecture position:
    Deterministic Engine ──┐
                           ├─→ Fusion Layer ─→ Confidence Engine ─→ Telegram
    TFT Forecast (or None)─┘

When TFT returns None (no checkpoint): pass deterministic output through
unchanged, set ai_forecast_status = "unavailable". This is the expected
operating state until a BMRI TFT model is trained.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from src.stock.predict import TFTResult


class FusionResult:
    """Output of the Fusion Layer — what the Confidence Engine consumes."""

    def __init__(
        self,
        deterministic: Dict[str, Any],
        tft_result: Optional[TFTResult],
        ai_forecast_status: str,        # "enabled" | "unavailable"
        ai_forecast_reason: str,        # human-readable reason
        ai_forecast_data: Optional[Dict[str, Any]] = None,
    ):
        self.deterministic = deterministic
        self.tft_result = tft_result
        self.ai_forecast_status = ai_forecast_status
        self.ai_forecast_reason = ai_forecast_reason
        self.ai_forecast_data = ai_forecast_data

    @property
    def ai_enabled(self) -> bool:
        return self.ai_forecast_status == "enabled"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deterministic": self.deterministic,
            "ai_forecast_status": self.ai_forecast_status,
            "ai_forecast_reason": self.ai_forecast_reason,
            "ai_forecast_data": self.ai_forecast_data,
        }


def fuse(
    deterministic: Dict[str, Any],
    tft_result: Optional[TFTResult],
) -> FusionResult:
    """
    Merge deterministic analysis with TFT forecast.

    If tft_result is None: pass-through (deterministic output unchanged).
    If tft_result is present: record agreement/conflict between AI direction
    and deterministic direction.

    The Fusion Layer does NOT alter the Rule Engine's decision — it only
    annotates. The Rule Engine remains the sole decision authority.
    """
    if tft_result is None:
        return FusionResult(
            deterministic=deterministic,
            tft_result=None,
            ai_forecast_status="unavailable",
            ai_forecast_reason="No trained BMRI checkpoint",
            ai_forecast_data=None,
        )

    # TFT produced a forecast — record agreement/conflict metadata
    det_direction = deterministic.get("direction", "neutral")
    ai_direction = tft_result.direction

    if det_direction == ai_direction:
        agreement = "agreement"
    elif ai_direction == "neutral":
        agreement = "neutral"
    else:
        agreement = "conflict"

    ai_data = {
        "direction": ai_direction,
        "confidence": round(tft_result.confidence, 3),
        "quantiles": tft_result.quantiles,
        "horizons": tft_result.horizons,
        "agreement_with_deterministic": agreement,
        "model_version": tft_result.model_version,
    }

    return FusionResult(
        deterministic=deterministic,
        tft_result=tft_result,
        ai_forecast_status="enabled",
        ai_forecast_reason=f"TFT model active ({tft_result.model_version})",
        ai_forecast_data=ai_data,
    )
