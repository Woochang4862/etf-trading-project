"""
Feature column definitions and LightGBM parameters from AhnLab LGBM rank model.
Extracted from AhnLab_LGBM_rank_0.19231/train.py
"""

from typing import List

# Base features from raw data
BASE_FEATURE_COLS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dividends",
    "stock_splits",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_63d",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "rsi_28",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "bb_position",
    "atr_14",
    "obv",
    "ema_10",
    "ema_20",
    "ema_50",
    "ema_200",
    "sma_10",
    "sma_20",
    "sma_50",
    "stoch_k",
    "stoch_d",
    "adx",
    "cci",
    "willr",
    "mfi",
    "vwap",
    "volume_sma_20",
    "volume_ratio",
    "vix",
    "fed_funds_rate",
    "unemployment_rate",
    "cpi",
    "treasury_10y",
    "treasury_2y",
    "yield_curve",
    "oil_price",
    "usd_eur",
    "high_yield_spread",
]

# Engineered features computed from base features
ENGINEERED_FEATURE_COLS: List[str] = [
    "ret_10d",
    "ret_30d",
    "vol_20d",
    "vol_63d",
    "price_to_sma_50",
    "price_to_ema_200",
    "volume_trend",
    "close_to_high_52w",
    "ret_5d_20d_ratio",
    "momentum_strength",
    "volume_surge",
    "ret_vol_ratio_20d",
    "ret_vol_ratio_63d",
    "trend_acceleration",
    "close_to_high_20d",
    "close_to_high_63d",
    "close_to_high_126d",
    "ema_5",
    "ema_100",
    "price_to_ema_10",
    "price_to_ema_50",
    "ema_cross_short",
    "ema_cross_long",
    "ema_slope_20",
]

# Combined base + engineered features
FEATURE_COLS: List[str] = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS

# Columns to apply cross-sectional z-score normalization
ZS_BASE_COLS: List[str] = [
    "vol_63d",
    "volume_sma_20",
    "obv",
    "vwap",
    "ema_200",
    "price_to_ema_200",
    "close_to_high_52w",
]
ZS_FEATURE_COLS: List[str] = [f"{col}_zs" for col in ZS_BASE_COLS]

# Columns to apply cross-sectional percentile ranking
RANK_BASE_COLS: List[str] = [
    "ret_20d",
    "ret_63d",
    "vol_20d",
    "momentum_strength",
    "volume_surge",
]
RANK_FEATURE_COLS: List[str] = [f"{col}_rank" for col in RANK_BASE_COLS]

# All feature columns for model input
ALL_FEATURE_COLS: List[str] = FEATURE_COLS + ZS_FEATURE_COLS + RANK_FEATURE_COLS

# LightGBM LambdaRank hyperparameters
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "label_gain": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 30, 33, 36, 39, 43, 47, 52, 56, 61, 66,
        71, 77, 83, 89, 95, 101, 107, 114, 121, 128, 135, 142
    ],
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 45,
    "min_child_samples": 60,
    "subsample": 0.7,
    "subsample_freq": 5,
    "colsample_bytree": 0.65,
    "min_split_gain": 0.00,
    "reg_alpha": 0.8,
    "reg_lambda": 1.2,
    "verbosity": 1,
}

# Training hyperparameters
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 150
TARGET_HORIZON = 63
TOP_K = 100
VALIDATION_DAYS = 90
MIN_HISTORY_DAYS = 126
RELEVANCE_BINS = 50
SEED = 42
