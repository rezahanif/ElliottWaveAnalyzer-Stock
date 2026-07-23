"""
cluster_check.py
----------------
Helper functions to perform statistical overlap and probability mass checks.
"""

def compute_probability_mass(lower: float, upper: float, q10: float, q50: float, q90: float) -> float:
    """
    Calculate the percentage of TFT probability mass falling within the interval [lower, upper].
    Uses a piecewise linear CDF constructed from the 10th, 50th, and 90th quantiles.
    """
    if upper <= lower:
        return 0.0

    d1 = q50 - q10
    d2 = q90 - q50
    if d1 <= 0:
        d1 = 1e-5
    if d2 <= 0:
        d2 = 1e-5

    lower_bound = q10 - d1
    upper_bound = q90 + d2

    def cdf(x: float) -> float:
        if x <= lower_bound:
            return 0.0
        if x <= q10:
            return 0.10 * (x - lower_bound) / (q10 - lower_bound)
        if x <= q50:
            return 0.10 + 0.40 * (x - q10) / (q50 - q10)
        if x <= q90:
            return 0.50 + 0.40 * (x - q50) / (q90 - q50)
        if x <= upper_bound:
            return 0.90 + 0.10 * (x - q90) / (upper_bound - q90)
        return 1.0

    return max(0.0, min(1.0, cdf(upper) - cdf(lower)))


def is_confluent(q50: float, cluster_lower: float, cluster_upper: float, tolerance_pct: float = 2.0) -> bool:
    """
    Check if a median forecast (q50) overlaps a Fibonacci cluster within tolerance_pct.
    """
    expanded_lower = cluster_lower * (1.0 - tolerance_pct / 100.0)
    expanded_upper = cluster_upper * (1.0 + tolerance_pct / 100.0)
    return expanded_lower <= q50 <= expanded_upper
