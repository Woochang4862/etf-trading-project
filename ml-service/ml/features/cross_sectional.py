"""
Cross-sectional features for ETF Stock Prediction

CRITICAL: These features capture relative position among stocks on the same date.
This is essential for ranking prediction tasks.
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from ..config import config


def calculate_percentile_rank(
    panel: pd.DataFrame,
    column: str,
    group_col: str = 'date'
) -> pd.Series:
    """
    Calculate percentile rank within each group (0-1)

    Args:
        panel: Panel DataFrame
        column: Column to rank
        group_col: Grouping column (typically 'date')

    Returns:
        Series of percentile ranks
    """
    return panel.groupby(group_col)[column].transform(
        lambda x: x.rank(pct=True, na_option='keep')
    )


def calculate_zscore(
    panel: pd.DataFrame,
    column: str,
    group_col: str = 'date'
) -> pd.Series:
    """
    Calculate z-score within each group

    Args:
        panel: Panel DataFrame
        column: Column to standardize
        group_col: Grouping column

    Returns:
        Series of z-scores
    """
    def zscore(x):
        mean = x.mean()
        std = x.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=x.index)
        return (x - mean) / std

    return panel.groupby(group_col)[column].transform(zscore)


def calculate_relative_to_mean(
    panel: pd.DataFrame,
    column: str,
    group_col: str = 'date'
) -> pd.Series:
    """
    Calculate value relative to group mean

    Returns:
        Series of (value / mean - 1)
    """
    return panel.groupby(group_col)[column].transform(
        lambda x: x / x.mean() - 1 if x.mean() != 0 else 0
    )


def add_cross_sectional_features(
    panel: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add cross-sectional features to panel data

    These features represent each stock's RELATIVE position
    among all stocks on the same date. Critical for ranking.

    Args:
        panel: Panel DataFrame with columns: date, ticker, features...
        feature_cols: Columns to create cross-sectional features for.
                     If None, uses config defaults.

    Returns:
        Panel DataFrame with added cross-sectional features
    """
    panel = panel.copy()

    # Ensure required columns exist
    if 'date' not in panel.columns:
        raise ValueError("Panel must have 'date' column")
    if 'ticker' not in panel.columns:
        raise ValueError("Panel must have 'ticker' column")

    # Default features to rank
    if feature_cols is None:
        feature_cols = config.features.rank_features

    # Filter to existing columns
    available_cols = [c for c in feature_cols if c in panel.columns]

    # === Percentile Ranks ===
    for col in available_cols:
        rank_col = f'{col}_rank'
        panel[rank_col] = calculate_percentile_rank(panel, col)

    # === Z-Scores ===
    zscore_cols = [
        'ret_1d', 'ret_5d', 'ret_20d', 'ret_63d',
        'rsi_14', 'volume_ratio', 'bb_position'
    ]
    for col in zscore_cols:
        if col in panel.columns:
            panel[f'{col}_zscore'] = calculate_zscore(panel, col)

    # === Relative to Mean ===
    relative_cols = ['volume', 'atr_14', 'volatility_20']
    for col in relative_cols:
        if col in panel.columns:
            panel[f'{col}_relative'] = calculate_relative_to_mean(panel, col)

    # === Quintile Groups ===
    # Categorize into quintiles (1-5) for key features
    def safe_qcut(x, q=5):
        x_nonnull = x.dropna()
        if x_nonnull.nunique() < 2:
            return pd.Series(np.nan, index=x.index)

        try:
            bins = pd.qcut(
                x_nonnull,
                q=min(q, x_nonnull.nunique()),
                duplicates="drop"
            )
            return bins.cat.codes.reindex(x.index) + 1
        except ValueError:
            return pd.Series(np.nan, index=x.index)
    quintile_cols = ['ret_20d', 'ret_63d', 'rsi_14']
    for col in quintile_cols:
        if col in panel.columns:
            panel[f'{col}_quintile'] = (
                panel.groupby('date')[col]
                .transform(safe_qcut)
                .astype(float)
            )

    # === Market-Relative Returns ===
    # Compare stock return to market (mean) return
    return_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'ret_63d']
    for col in return_cols:
        if col in panel.columns:
            market_return = panel.groupby('date')[col].transform('mean')
            panel[f'{col}_vs_market'] = panel[col] - market_return

    # === Winner/Loser Flags ===
    # Top 10% and Bottom 10%
    if 'ret_20d' in panel.columns:
        panel['is_winner_20d'] = (
            panel.groupby('date')['ret_20d'].transform(
                lambda x: x >= x.quantile(0.9)
            )
        ).astype(int)

        panel['is_loser_20d'] = (
            panel.groupby('date')['ret_20d'].transform(
                lambda x: x <= x.quantile(0.1)
            )
        ).astype(int)

    # === Momentum Rank Changes ===
    # How has the rank changed over time?
    if 'ret_20d_rank' in panel.columns:
        # Rank improvement over last 5 days
        panel['rank_momentum_5d'] = panel.groupby('ticker')['ret_20d_rank'].transform(
            lambda x: x - x.shift(5)
        )

    # === Cross-Sectional Momentum (Industry/Sector neutral) ===
    # Note: This requires sector information which may not be available
    # Placeholder for sector-relative return

    # === Dispersion Measures ===
    # How dispersed are returns on this date?
    if 'ret_1d' in panel.columns:
        panel['market_dispersion'] = panel.groupby('date')['ret_1d'].transform('std')

        # Is this a high dispersion day? (Good for stock picking)
        dispersion_mean = panel.groupby('date')['ret_1d'].transform(
            lambda x: x.std()
        ).rolling(window=20).mean()
        panel['high_dispersion_day'] = (panel['market_dispersion'] > dispersion_mean).astype(int)

    # === Count Statistics ===
    # Number of stocks in universe on this date
    panel['universe_size'] = panel.groupby('date')['ticker'].transform('count')

    # === Extreme Value Indicators ===
    # Is this stock at extreme levels vs peers?
    if 'rsi_14_rank' in panel.columns:
        panel['rsi_extreme_low'] = (panel['rsi_14_rank'] < 0.1).astype(int)
        panel['rsi_extreme_high'] = (panel['rsi_14_rank'] > 0.9).astype(int)

    if 'bb_position_rank' in panel.columns:
        panel['bb_extreme_low'] = (panel['bb_position_rank'] < 0.1).astype(int)
        panel['bb_extreme_high'] = (panel['bb_position_rank'] > 0.9).astype(int)

    return panel


def create_lagged_rank_features(
    panel: pd.DataFrame,
    rank_cols: List[str],
    lags: List[int] = [1, 5, 10]
) -> pd.DataFrame:
    """
    Create lagged rank features (how was rank in the past?)

    Args:
        panel: Panel DataFrame
        rank_cols: Rank columns to lag
        lags: Lag periods

    Returns:
        Panel with lagged rank features
    """
    panel = panel.copy()

    for col in rank_cols:
        if col not in panel.columns:
            continue

        for lag in lags:
            lag_col = f'{col}_lag{lag}'
            panel[lag_col] = panel.groupby('ticker')[col].shift(lag)

    return panel


# List of features added by this module
CROSS_SECTIONAL_FEATURES = [
    # Ranks
    'ret_1d_rank', 'ret_5d_rank', 'ret_20d_rank', 'ret_63d_rank',
    'rsi_14_rank', 'volume_ratio_rank', 'price_to_sma_20_rank', 'bb_position_rank',
    # Z-scores
    'ret_1d_zscore', 'ret_5d_zscore', 'ret_20d_zscore', 'ret_63d_zscore',
    'rsi_14_zscore', 'volume_ratio_zscore', 'bb_position_zscore',
    # Relative
    'volume_relative', 'atr_14_relative', 'volatility_20_relative',
    # Quintiles
    'ret_20d_quintile', 'ret_63d_quintile', 'rsi_14_quintile',
    # Market-relative
    'ret_1d_vs_market', 'ret_5d_vs_market', 'ret_20d_vs_market', 'ret_63d_vs_market',
    # Flags
    'is_winner_20d', 'is_loser_20d',
    # Rank momentum
    'rank_momentum_5d',
    # Dispersion
    'market_dispersion', 'high_dispersion_day',
    # Universe
    'universe_size',
    # Extremes
    'rsi_extreme_low', 'rsi_extreme_high',
    'bb_extreme_low', 'bb_extreme_high'
]
