"""
AhnLab LGBM Rank feature engineering module.

This module contains feature engineering logic extracted from the AhnLab
competition solution (AhnLab_LGBM_rank_0.19231/train.py).

Usage:
    from features.ahnlab import (
        add_engineered_features,
        add_cross_sectional_zscores,
        add_cross_sectional_ranks,
        add_relevance_labels,
        ALL_FEATURE_COLS,
        LGB_PARAMS,
        add_technical_indicators,
        MacroDataCollector,
    )

    # Feature engineering pipeline
    panel = add_engineered_features(panel)
    panel = add_cross_sectional_zscores(panel)
    panel = add_cross_sectional_ranks(panel)
    df = add_relevance_labels(df)
"""

# Constants
from .constants import (
    ALL_FEATURE_COLS,
    BASE_FEATURE_COLS,
    EARLY_STOPPING_ROUNDS,
    ENGINEERED_FEATURE_COLS,
    FEATURE_COLS,
    LGB_PARAMS,
    MIN_HISTORY_DAYS,
    NUM_BOOST_ROUND,
    RANK_BASE_COLS,
    RANK_FEATURE_COLS,
    RELEVANCE_BINS,
    SEED,
    TARGET_HORIZON,
    TOP_K,
    VALIDATION_DAYS,
    ZS_BASE_COLS,
    ZS_FEATURE_COLS,
)

# Feature engineering functions
from .cross_sectional import add_cross_sectional_ranks, add_cross_sectional_zscores
from .engineered import add_engineered_features
from .target import add_relevance_labels
from .technical import add_technical_indicators, TECHNICAL_FEATURE_COLS
from .macro import MacroDataCollector, MACRO_FEATURE_COLS

__all__ = [
    # Constants
    "BASE_FEATURE_COLS",
    "ENGINEERED_FEATURE_COLS",
    "FEATURE_COLS",
    "ZS_BASE_COLS",
    "ZS_FEATURE_COLS",
    "RANK_BASE_COLS",
    "RANK_FEATURE_COLS",
    "ALL_FEATURE_COLS",
    "LGB_PARAMS",
    "NUM_BOOST_ROUND",
    "EARLY_STOPPING_ROUNDS",
    "TARGET_HORIZON",
    "TOP_K",
    "VALIDATION_DAYS",
    "MIN_HISTORY_DAYS",
    "RELEVANCE_BINS",
    "SEED",
    "TECHNICAL_FEATURE_COLS",
    "MACRO_FEATURE_COLS",
    # Functions
    "add_engineered_features",
    "add_cross_sectional_zscores",
    "add_cross_sectional_ranks",
    "add_relevance_labels",
    "add_technical_indicators",
    "MacroDataCollector",
]
