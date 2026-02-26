"""
Momentum and trend features for ETF Stock Prediction
"""
import pandas as pd
import numpy as np
from typing import List

from ..config import config


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_roc(series: pd.Series, period: int) -> pd.Series:
    """Calculate Rate of Change (%)"""
    return (series / series.shift(period) - 1) * 100


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum and trend features to DataFrame

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added momentum features
    """
    df = df.copy()

    if 'close' not in df.columns:
        raise ValueError("Missing required column: close")

    close = df['close']
    high = df['high'] if 'high' in df.columns else close
    low = df['low'] if 'low' in df.columns else close

    # === Simple Moving Averages ===
    for period in config.features.sma_periods:
        df[f'sma_{period}'] = calculate_sma(close, period)

    # === Exponential Moving Averages ===
    for period in config.features.ema_periods:
        df[f'ema_{period}'] = calculate_ema(close, period)

    # === Price to SMA Ratios (Distance from MA) ===
    for period in [20, 50, 63]:
        sma_col = f'sma_{period}'
        if sma_col in df.columns:
            df[f'price_to_sma_{period}'] = close / df[sma_col] - 1

    # === SMA Slopes (Trend Strength) ===
    if 'sma_20' in df.columns:
        df['sma_20_slope'] = df['sma_20'].pct_change(5)  # 5-day slope
    if 'sma_50' in df.columns:
        df['sma_50_slope'] = df['sma_50'].pct_change(10)  # 10-day slope

    # === Rate of Change (ROC) ===
    for period in config.features.roc_periods:
        df[f'roc_{period}'] = calculate_roc(close, period)

    # === Momentum (Price difference) ===
    df['mom_10'] = close - close.shift(10)
    df['mom_20'] = close - close.shift(20)

    # === Moving Average Crossover Signals ===
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        # Golden Cross: Short MA crosses above Long MA
        df['golden_cross'] = (
            (df['sma_20'] > df['sma_50']) &
            (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
        ).astype(int)

        # Dead Cross: Short MA crosses below Long MA
        df['dead_cross'] = (
            (df['sma_20'] < df['sma_50']) &
            (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
        ).astype(int)

        # MA Trend: 1 if short > long, -1 otherwise
        df['ma_trend'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)

    # === EMA Crossover ===
    if 'ema_20' in df.columns and 'ema_63' in df.columns:
        df['ema_trend'] = np.where(df['ema_20'] > df['ema_63'], 1, -1)

    # === Price Position relative to MAs ===
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        # Price above both MAs = strong uptrend
        df['price_above_mas'] = (
            (close > df['sma_20']) & (close > df['sma_50'])
        ).astype(int)
        # Price below both MAs = strong downtrend
        df['price_below_mas'] = (
            (close < df['sma_20']) & (close < df['sma_50'])
        ).astype(int)

    # === Consecutive Up/Down Days ===
    daily_return = close.pct_change()

    # Up streak
    up_days = (daily_return > 0).astype(int)
    df['up_streak'] = up_days.groupby(
        (up_days != up_days.shift()).cumsum()
    ).cumsum() * up_days

    # Down streak
    down_days = (daily_return < 0).astype(int)
    df['down_streak'] = down_days.groupby(
        (down_days != down_days.shift()).cumsum()
    ).cumsum() * down_days

    # === High-Low Range Position ===
    # Where is current price in recent high-low range?
    for period in [20, 63]:
        rolling_high = high.rolling(window=period).max()
        rolling_low = low.rolling(window=period).min()
        df[f'range_position_{period}'] = (
            (close - rolling_low) / (rolling_high - rolling_low + 1e-10)
        )

    return df


# List of features added by this module
MOMENTUM_FEATURES = [
    # SMAs
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_63',
    # EMAs
    'ema_20', 'ema_63',
    # Price to SMA ratios
    'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_63',
    # SMA slopes
    'sma_20_slope', 'sma_50_slope',
    # ROC
    'roc_5', 'roc_10', 'roc_20',
    # Momentum
    'mom_10', 'mom_20',
    # Crossovers
    'golden_cross', 'dead_cross', 'ma_trend', 'ema_trend',
    # Position
    'price_above_mas', 'price_below_mas',
    # Streaks
    'up_streak', 'down_streak',
    # Range position
    'range_position_20', 'range_position_63'
]
