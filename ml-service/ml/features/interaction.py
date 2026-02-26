"""
Feature interaction features for ETF Stock Prediction

Captures non-linear relationships between key features.
"""

import pandas as pd
import numpy as np
from typing import List


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features to DataFrame

    Args:
        df: DataFrame with price, volume, and indicator columns

    Returns:
        DataFrame with added interaction features
    """
    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    close = df["close"]

    # === Momentum × Volatility (Risk-Adjusted Momentum) ===
    if all(col in df.columns for col in ["ret_5d", "volatility_20"]):
        df["mom_vol_adj_5d"] = df["ret_5d"] / (df["volatility_20"] + 1e-10)

    if all(col in df.columns for col in ["ret_20d", "volatility_20"]):
        df["mom_vol_adj_20d"] = df["ret_20d"] / (df["volatility_20"] + 1e-10)

    if all(col in df.columns for col in ["ret_63d", "volatility_63"]):
        df["mom_vol_adj_63d"] = df["ret_63d"] / (df["volatility_63"] + 1e-10)

    # === RSI × Volume (Volume-Confirmed RSI Signals) ===
    if all(col in df.columns for col in ["rsi_14", "volume_ratio"]):
        # RSI extreme with high volume = strong signal
        rsi_extreme_overbought = df["rsi_14"] > 70
        rsi_extreme_oversold = df["rsi_14"] < 30

        df["rsi_vol_confirm_overbought"] = (
            rsi_extreme_overbought & (df["volume_ratio"] > 1.5)
        ).astype(int)

        df["rsi_vol_confirm_oversold"] = (
            rsi_extreme_oversold & (df["volume_ratio"] > 1.5)
        ).astype(int)

        # RSI normalized by volume
        df["rsi_vol_adj"] = (df["rsi_14"] - 50) * np.log1p(df["volume_ratio"])

    # === Price × Volume (Price Movement with Volume Strength) ===
    if all(col in df.columns for col in ["ret_1d", "volume_ratio"]):
        # Positive return with high volume = strong buying
        df["price_vol_bullish"] = (
            (df["ret_1d"] > 0) & (df["volume_ratio"] > 1.2)
        ).astype(int)

        # Negative return with high volume = strong selling
        df["price_vol_bearish"] = (
            (df["ret_1d"] < 0) & (df["volume_ratio"] > 1.2)
        ).astype(int)

        # Volume-weighted return
        df["price_vol_weighted_1d"] = df["ret_1d"] * np.log1p(df["volume_ratio"])

    if "volume" in df.columns:
        # Dollar volume change
        df["dollar_vol_change"] = (close * df["volume"]).pct_change()

    # === MA Crossing × Trend (Crossover Validity) ===
    if all(col in df.columns for col in ["price_to_sma_20", "ma_trend"]):
        # Crossover in trending direction = more reliable
        df["cross_trend_align"] = (
            (df["price_to_sma_20"] > 0) & (df["ma_trend"] > 0)
            | (df["price_to_sma_20"] < 0) & (df["ma_trend"] < 0)
        ).astype(int)

        # Crossover against trend = potential reversal
        df["cross_trend_diverge"] = (
            (df["price_to_sma_20"] > 0) & (df["ma_trend"] < 0)
            | (df["price_to_sma_20"] < 0) & (df["ma_trend"] > 0)
        ).astype(int)

    # === Bollinger Band × Volume ===
    if all(col in df.columns for col in ["bb_position", "volume_ratio"]):
        # BB squeeze with high volume = breakout potential
        bb_near_middle = np.abs(df["bb_position"] - 0.5) < 0.1
        df["bb_squeeze_volume"] = (bb_near_middle & (df["volume_ratio"] > 1.5)).astype(
            int
        )

        # BB extreme with volume = potential reversal
        bb_extreme = np.abs(df["bb_position"] - 0.5) > 0.4
        df["bb_extreme_volume"] = (bb_extreme & (df["volume_ratio"] > 1.5)).astype(int)

    # === ATR × Price Movement ===
    if all(col in df.columns for col in ["atr_ratio", "ret_1d"]):
        # Large move relative to ATR
        df["atr_relative_move"] = df["ret_1d"].abs() / (df["atr_ratio"] + 1e-10)

        # Extreme move (2x ATR)
        df["extreme_atr_move"] = (df["atr_relative_move"] > 2.0).astype(int)

    # === RSI × MACD (Confluence) ===
    if all(col in df.columns for col in ["rsi_14", "macd", "macd_signal"]):
        # Both overbought = strong sell signal
        df["rsi_macd_overbought"] = (
            (df["rsi_14"] > 70) & (df["macd"] < df["macd_signal"])
        ).astype(int)

        # Both oversold = strong buy signal
        df["rsi_macd_oversold"] = (
            (df["rsi_14"] < 30) & (df["macd"] > df["macd_signal"])
        ).astype(int)

    # === Momentum Consistency (Multiple Timeframes) ===
    if all(col in df.columns for col in ["ret_5d", "ret_20d", "ret_63d"]):
        # All positive = strong bullish
        df["mom_all_positive"] = (
            (df["ret_5d"] > 0) & (df["ret_20d"] > 0) & (df["ret_63d"] > 0)
        ).astype(int)

        # All negative = strong bearish
        df["mom_all_negative"] = (
            (df["ret_5d"] < 0) & (df["ret_20d"] < 0) & (df["ret_63d"] < 0)
        ).astype(int)

        # Short positive, long negative = bounce potential
        df["mom_bounce_potential"] = (
            (df["ret_5d"] > 0) & (df["ret_20d"] > 0) & (df["ret_63d"] < 0)
        ).astype(int)

        # Short negative, long positive = pullback opportunity
        df["mom_pullback_opp"] = (
            (df["ret_5d"] < 0) & (df["ret_20d"] < 0) & (df["ret_63d"] > 0)
        ).astype(int)

        # Momentum acceleration
        df["mom_accel"] = df["ret_5d"] - df["ret_20d"] / 4

    # === Volatility × RSI (Volatility-Adjusted Overbought/Oversold) ===
    if all(col in df.columns for col in ["volatility_20", "rsi_14"]):
        # High volatility: RSI extreme is less significant
        vol_adj_rsi = (df["rsi_14"] - 50) / (df["volatility_20"] + 0.1)
        df["vol_adj_rsi"] = np.clip(vol_adj_rsi, -1, 1)

    # === Gap × Volume ===
    if "gap" in df.columns and "volume_ratio" in df.columns:
        # Large gap with high volume = continuation likely
        large_gap = df["gap"].abs() > 0.02
        df["gap_vol_confirm"] = (large_gap & (df["volume_ratio"] > 1.5)).astype(int)

        # Gap fill likelihood (gap against recent trend)
        if "ret_5d" in df.columns:
            df["gap_vs_trend"] = (df["gap"] > 0) & (df["ret_5d"] < 0)

    # === Trend Strength × ADX ===
    if all(col in df.columns for col in ["adx", "ma_trend"]):
        # Strong trend + ADX > 25 = very reliable trend
        df["strong_trend_confirm"] = (
            (df["adx"] > 25) & (np.abs(df["ma_trend"]) > 0.5)
        ).astype(int)

        # Weak ADX despite MA trend = potential reversal
        df["weak_adx_trend"] = (
            (df["adx"] < 20) & (np.abs(df["ma_trend"]) > 0.5)
        ).astype(int)

    # === Stochastic × MACD ===
    if all(col in df.columns for col in ["stoch_k", "stoch_d", "macd", "macd_signal"]):
        # Stochastic oversold + MACD bullish = buy signal
        df["stoch_macd_buy"] = (
            (df["stoch_k"] < 20)
            & (df["stoch_d"] < 20)
            & (df["macd"] > df["macd_signal"])
        ).astype(int)

        # Stochastic overbought + MACD bearish = sell signal
        df["stoch_macd_sell"] = (
            (df["stoch_k"] > 80)
            & (df["stoch_d"] > 80)
            & (df["macd"] < df["macd_signal"])
        ).astype(int)

    # === Support/Resistance × Volume ===
    if "near_52w_high" in df.columns and "volume_ratio" in df.columns:
        # Near 52-week high with volume = breakout
        df["high_vol_breakout"] = (
            (df["near_52w_high"] > 0.98) & (df["volume_ratio"] > 1.3)
        ).astype(int)

    if "near_52w_low" in df.columns and "volume_ratio" in df.columns:
        # Near 52-week low with volume = breakdown
        df["low_vol_breakdown"] = (
            (df["near_52w_low"] < 1.02) & (df["volume_ratio"] > 1.3)
        ).astype(int)

    # === Composite Interaction Score ===
    # Combine multiple interaction signals
    interaction_score = 0

    bullish_interactions = [
        "rsi_vol_confirm_oversold",
        "price_vol_bullish",
        "cross_trend_align",
        "bb_squeeze_volume",
        "rsi_macd_oversold",
        "mom_all_positive",
        "stoch_macd_buy",
        "high_vol_breakout",
    ]

    bearish_interactions = [
        "rsi_vol_confirm_overbought",
        "price_vol_bearish",
        "cross_trend_diverge",
        "bb_extreme_volume",
        "rsi_macd_overbought",
        "mom_all_negative",
        "stoch_macd_sell",
        "low_vol_breakdown",
    ]

    for col in bullish_interactions:
        if col in df.columns:
            interaction_score += df[col]

    for col in bearish_interactions:
        if col in df.columns:
            interaction_score -= df[col]

    df["interaction_net_score"] = interaction_score
    df["interaction_bullish"] = (interaction_score >= 2).astype(int)
    df["interaction_bearish"] = (interaction_score <= -2).astype(int)

    # === Polynomial Features for Key Indicators ===
    if "ret_20d" in df.columns:
        df["ret_20d_sq"] = df["ret_20d"] ** 2
        df["ret_20d_abs"] = np.abs(df["ret_20d"])

    if "rsi_14" in df.columns:
        df["rsi_14_sq"] = ((df["rsi_14"] - 50) ** 2) / 100  # Normalized

    if "volume_ratio" in df.columns:
        df["volume_ratio_sq"] = (df["volume_ratio"] - 1) ** 2

    return df


# List of features added by this module
INTERACTION_FEATURES = [
    # Momentum × Volatility
    "mom_vol_adj_5d",
    "mom_vol_adj_20d",
    "mom_vol_adj_63d",
    # RSI × Volume
    "rsi_vol_confirm_overbought",
    "rsi_vol_confirm_oversold",
    "rsi_vol_adj",
    # Price × Volume
    "price_vol_bullish",
    "price_vol_bearish",
    "price_vol_weighted_1d",
    "dollar_vol_change",
    # MA × Trend
    "cross_trend_align",
    "cross_trend_diverge",
    # BB × Volume
    "bb_squeeze_volume",
    "bb_extreme_volume",
    # ATR × Price
    "atr_relative_move",
    "extreme_atr_move",
    # RSI × MACD
    "rsi_macd_overbought",
    "rsi_macd_oversold",
    # Momentum Consistency
    "mom_all_positive",
    "mom_all_negative",
    "mom_bounce_potential",
    "mom_pullback_opp",
    "mom_accel",
    # Vol × RSI
    "vol_adj_rsi",
    # Gap × Volume
    "gap_vol_confirm",
    "gap_vs_trend",
    # Trend × ADX
    "strong_trend_confirm",
    "weak_adx_trend",
    # Stoch × MACD
    "stoch_macd_buy",
    "stoch_macd_sell",
    # Support/Resistance × Volume
    "high_vol_breakout",
    "low_vol_breakdown",
    # Composite
    "interaction_net_score",
    "interaction_bullish",
    "interaction_bearish",
    # Polynomial
    "ret_20d_sq",
    "ret_20d_abs",
    "rsi_14_sq",
    "volume_ratio_sq",
]
