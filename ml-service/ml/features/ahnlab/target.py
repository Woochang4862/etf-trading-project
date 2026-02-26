"""
Target variable and relevance label generation.
Extracted from AhnLab_LGBM_rank_0.19231/train.py lines 350-368.
"""

import pandas as pd

from .constants import RELEVANCE_BINS


def add_relevance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add relevance labels for LambdaRank training.

    Converts continuous target (3-month return) into discrete relevance labels
    using quantile-based binning within each date group.

    Higher relevance = higher expected return rank.

    Args:
        df: DataFrame with columns ['date', 'target_3m'].

    Returns:
        DataFrame with additional 'relevance' column (int, 0 to RELEVANCE_BINS-1).
    """
    df = df.copy()

    def _label(series: pd.Series) -> pd.Series:
        q = min(RELEVANCE_BINS, series.shape[0])
        if q <= 1:
            return pd.Series(0, index=series.index, dtype=int)
        ranks = series.rank(method="first")
        labels = pd.qcut(
            ranks,
            q=q,
            labels=False,
            duplicates="drop",
        )
        return labels.fillna(0).astype(int)

    df["relevance"] = df.groupby("date")["target_3m"].transform(_label)
    df["relevance"] = df["relevance"].fillna(0).astype(int)
    return df
