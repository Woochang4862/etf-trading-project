"""
Market regime features for ETF Stock Prediction

Identifies market state (bull/bear/sideways, high/low volatility)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def detect_volatility_regime(
    volatility: pd.Series, window: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detect volatility regime state

    Args:
        volatility: Rolling volatility series
        window: Window to calculate mean/std

    Returns:
        (vol_regime, vol_above_mean, vol_ratio_to_mean)
        vol_regime: 1=high, 0=medium, -1=low
    """
    vol_mean = volatility.rolling(window=window).mean()
    vol_std = volatility.rolling(window=window).std()

    # Vol ratio to mean
    vol_ratio_to_mean = volatility / (vol_mean + 1e-10)

    # Above/below mean
    vol_above_mean = (volatility > vol_mean).astype(int)

    # Regime classification
    vol_regime = pd.Series(0, index=volatility.index)
    vol_regime[volatility > vol_mean + vol_std] = 1  # High vol
    vol_regime[volatility < vol_mean - vol_std] = -1  # Low vol

    return vol_regime, vol_above_mean, vol_ratio_to_mean


def detect_trend_regime(
    close: pd.Series, adx: pd.Series, adx_threshold: float = 25.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detect trend regime using ADX

    Args:
        close: Price series
        adx: ADX indicator
        adx_threshold: ADX threshold for trend vs ranging

    Returns:
        (trend_regime, trend_direction, trend_strength)
        trend_regime: 1=trending, 0=ranging
        trend_direction: 1=uptrend, -1=downtrend, 0=neutral
        trend_strength: Normalized ADX (0-1)
    """
    # Trend vs ranging (based on ADX)
    trend_regime = (adx > adx_threshold).astype(int)

    # Trend direction (based on short-term vs long-term MA)
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()

    trend_direction = pd.Series(0, index=close.index)
    trend_direction[close > ma_20] = 1
    trend_direction[close < ma_20] = -1

    # Trend strength (normalized ADX)
    trend_strength = (adx - 10) / 50  # Normalize roughly to 0-1
    trend_strength = trend_strength.clip(0, 1)

    return trend_regime, trend_direction, trend_strength


def detect_market_state(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detect overall market state (Bull/Bear/Sideways)

    Returns:
        (market_state, state_bull, state_bear, state_sideways)
        market_state: 1=bull, -1=bear, 0=sideways
    """
    ma_short = close.rolling(10).mean()
    ma_mid = close.rolling(50).mean()
    ma_long = close.rolling(200).mean()

    # Bull market: price above all MAs, short > mid > long
    is_bull = (
        (close > ma_short)
        & (close > ma_mid)
        & (close > ma_long)
        & (ma_short > ma_mid)
        & (ma_mid > ma_long)
    )

    # Bear market: price below all MAs, short < mid < long
    is_bear = (
        (close < ma_short)
        & (close < ma_mid)
        & (close < ma_long)
        & (ma_short < ma_mid)
        & (ma_mid < ma_long)
    )

    # Sideways: not bull nor bear
    is_sideways = ~is_bull & ~is_bear

    market_state = pd.Series(0, index=close.index)
    market_state[is_bull] = 1
    market_state[is_bear] = -1

    return (
        market_state,
        is_bull.astype(int),
        is_bear.astype(int),
        is_sideways.astype(int),
    )


def detect_momentum_regime(
    ret_short: pd.Series, ret_mid: pd.Series, ret_long: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect momentum regime based on multi-period returns

    Args:
        ret_short: Short-term return (e.g., 5d)
        ret_mid: Medium-term return (e.g., 20d)
        ret_long: Long-term return (e.g., 63d)

    Returns:
        (mom_regime, mom_strength)
        mom_regime: 1=positive, -1=negative, 0=mixed
        mom_strength: Average of absolute returns
    """
    # All positive = positive momentum
    all_positive = (ret_short > 0) & (ret_mid > 0) & (ret_long > 0)

    # All negative = negative momentum
    all_negative = (ret_short < 0) & (ret_mid < 0) & (ret_long < 0)

    mom_regime = pd.Series(0, index=ret_short.index)
    mom_regime[all_positive] = 1
    mom_regime[all_negative] = -1

    # Momentum strength (average absolute return)
    mom_strength = (np.abs(ret_short) + np.abs(ret_mid) + np.abs(ret_long)) / 3

    return mom_regime, mom_strength


def detect_regime_transition(
    regime: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detect regime transitions (changes in market state)

    Args:
        regime: Regime series (e.g., market_state, vol_regime)

    Returns:
        (regime_changed, regime_change_mag, regime_stable_periods)
    """
    regime_changed = (regime.diff() != 0).astype(int)

    # Magnitude of change
    regime_change_mag = regime.diff().abs()

    # How long has current regime been stable?
    stable_periods = pd.Series(0, index=regime.index)
    for i in range(1, len(regime)):
        if regime.iloc[i] == regime.iloc[i - 1]:
            stable_periods.iloc[i] = stable_periods.iloc[i - 1] + 1
        else:
            stable_periods.iloc[i] = 0

    return regime_changed, regime_change_mag, stable_periods


def calculate_vol_of_vol(volatility: pd.Series, window: int = 10) -> pd.Series:
    """
    Calculate volatility of volatility (vol-of-vol)

    Measures how stable volatility itself is
    """
    vol_of_vol = volatility.rolling(window=window).std()
    return vol_of_vol


def calculate_vol_skew(volatility: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate skewness of volatility

    Positive skew = more high volatility spikes
    """
    vol_skew = volatility.rolling(window=window).skew()
    return vol_skew


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all regime features to DataFrame

    Args:
        df: DataFrame with price and indicator columns

    Returns:
        DataFrame with added regime features
    """
    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    close = df["close"]

    # === Volatility Regime ===
    if "volatility_20" in df.columns:
        vol_regime, vol_above_mean, vol_ratio = detect_volatility_regime(
            df["volatility_20"]
        )

        df["vol_regime"] = vol_regime
        df["vol_above_mean"] = vol_above_mean
        df["vol_ratio_to_mean"] = vol_ratio

        # Vol-of-vol
        df["vol_of_vol_10"] = calculate_vol_of_vol(df["volatility_20"], 10)
        df["vol_skew_20"] = calculate_vol_skew(df["volatility_20"])

    # === Trend Regime ===
    if "adx" in df.columns:
        trend_regime, trend_direction, trend_strength = detect_trend_regime(
            close, df["adx"]
        )

        df["trend_regime"] = trend_regime
        df["trend_direction"] = trend_direction
        df["trend_strength"] = trend_strength

        # Strong trend indicator
        df["strong_trend"] = (df["trend_regime"] & (df["trend_strength"] > 0.5)).astype(
            int
        )

    # === Market State ===
    market_state, state_bull, state_bear, state_sideways = detect_market_state(close)

    df["market_state"] = market_state
    df["state_bull"] = state_bull
    df["state_bear"] = state_bear
    df["state_sideways"] = state_sideways

    # Market state strength (distance from 50 MA)
    ma_50 = close.rolling(50).mean()
    df["state_strength"] = (close - ma_50) / (ma_50 + 1e-10)

    # === Momentum Regime ===
    if all(col in df.columns for col in ["ret_5d", "ret_20d", "ret_63d"]):
        mom_regime, mom_strength = detect_momentum_regime(
            df["ret_5d"], df["ret_20d"], df["ret_63d"]
        )

        df["mom_regime"] = mom_regime
        df["mom_strength"] = mom_strength

        # Momentum alignment (all same direction)
        df["mom_aligned"] = (mom_regime != 0).astype(int)

    # === Regime Transitions ===
    if "vol_regime" in df.columns:
        vol_changed, vol_change_mag, vol_stable = detect_regime_transition(
            df["vol_regime"]
        )
        df["vol_regime_changed"] = vol_changed
        df["vol_change_mag"] = vol_change_mag
        df["vol_stable_periods"] = vol_stable

    if "market_state" in df.columns:
        state_changed, state_change_mag, state_stable = detect_regime_transition(
            df["market_state"]
        )
        df["market_state_changed"] = state_changed
        df["state_change_mag"] = state_change_mag
        df["state_stable_periods"] = state_stable

    # === Regime Confluence ===
    # Combine multiple regime indicators
    regime_signals = 0

    if "trend_regime" in df.columns:
        regime_signals += (df["trend_regime"] > 0).astype(int)
    if "state_bull" in df.columns:
        regime_signals += df["state_bull"]
    if "vol_above_mean" in df.columns:
        regime_signals += df["vol_above_mean"]
    if "mom_aligned" in df.columns:
        regime_signals += df["mom_aligned"]

    df["regime_confluence_bull"] = (regime_signals >= 3).astype(int)

    # Bearish confluence
    bear_signals = 0
    if "state_bear" in df.columns:
        bear_signals += df["state_bear"]
    if "mom_regime" in df.columns:
        bear_signals += (df["mom_regime"] < 0).astype(int)
    if "vol_above_mean" in df.columns:
        bear_signals += 1 - df["vol_above_mean"]

    df["regime_confluence_bear"] = (bear_signals >= 2).astype(int)

    # === Regime Momentum ===
    # Is the regime strengthening or weakening?
    if "trend_strength" in df.columns:
        df["trend_momentum"] = df["trend_strength"].diff(5)

    if "mom_strength" in df.columns:
        df["mom_momentum"] = df["mom_strength"].diff(5)

    if "vol_ratio_to_mean" in df.columns:
        df["vol_momentum"] = df["vol_ratio_to_mean"].diff(5)

    return df


# List of features added by this module
REGIME_FEATURES = [
    # Volatility regime
    "vol_regime",
    "vol_above_mean",
    "vol_ratio_to_mean",
    "vol_of_vol_10",
    "vol_skew_20",
    # Trend regime
    "trend_regime",
    "trend_direction",
    "trend_strength",
    "strong_trend",
    # Market state
    "market_state",
    "state_bull",
    "state_bear",
    "state_sideways",
    "state_strength",
    # Momentum regime
    "mom_regime",
    "mom_strength",
    "mom_aligned",
    # Regime transitions
    "vol_regime_changed",
    "vol_change_mag",
    "vol_stable_periods",
    "market_state_changed",
    "state_change_mag",
    "state_stable_periods",
    # Regime confluence
    "regime_confluence_bull",
    "regime_confluence_bear",
    # Regime momentum
    "trend_momentum",
    "mom_momentum",
    "vol_momentum",
]
