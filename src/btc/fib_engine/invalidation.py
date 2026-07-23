"""
invalidation.py
---------------
Evaluates real-time price validation and invalidation checks.
"""

def is_invalidated(
    price: float,
    invalidation_level: float,
    direction: str = "bearish",
) -> bool:
    """
    Check if the current price has breached the invalidation level.

    For a bearish scenario: any price above the invalidation level is an invalidation.
    For a bullish scenario: any price below the invalidation level is an invalidation.
    """
    if direction == "bearish":
        return price > invalidation_level
    else:
        return price < invalidation_level
