"""
Cross-sectional feature transformations.
Extracted from AhnLab_LGBM_rank_0.19231/train.py lines 284-298.
"""

import pandas as pd

from .constants import RANK_BASE_COLS, ZS_BASE_COLS


def add_cross_sectional_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional z-score normalized features.

    For each date, computes z-score normalization across all tickers:
    z = (x - mean) / std

    This captures relative positioning within each trading day.

    Args:
        panel: DataFrame with columns ['date'] and ZS_BASE_COLS features.

    Returns:
        DataFrame with additional z-score feature columns (suffix '_zs').
    """
    panel = panel.copy()
    grouped = panel.groupby("date")

    for col in ZS_BASE_COLS:
        z = grouped[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-6)
        )
        panel[f"{col}_zs"] = z

    return panel


def add_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional percentile rank features.

    For each date, computes percentile rank across all tickers.
    Rank is normalized to [0, 1] range.

    This captures relative ordering within each trading day.

    Args:
        panel: DataFrame with columns ['date'] and RANK_BASE_COLS features.

    Returns:
        DataFrame with additional rank feature columns (suffix '_rank').
    """
    panel = panel.copy()
    grouped = panel.groupby("date")

    for col in RANK_BASE_COLS:
        panel[f"{col}_rank"] = grouped[col].rank(pct=True)

    return panel
