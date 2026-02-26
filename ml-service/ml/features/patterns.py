"""
Price pattern features for ETF Stock Prediction

Detects candlestick patterns, price action patterns, and support/resistance zones.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def detect_doji(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Detect Doji pattern (open ≈ close)

    Args:
        threshold: Maximum difference between open and close as % of range
    """
    body = np.abs(close - open_price)
    body_range = high - low + 1e-10
    return (body / body_range < threshold).astype(int)


def detect_hammer(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Hammer and Inverted Hammer patterns

    Returns:
        (hammer, inverted_hammer)
    """
    body = np.abs(close - open_price)
    lower_shadow = np.minimum(open_price, close) - low
    upper_shadow = high - np.maximum(open_price, close)
    total_range = high - low + 1e-10

    # Hammer: small body, long lower shadow, little/no upper shadow, downtrend
    is_downtrend = close < close.shift(1)
    hammer = (
        (body / total_range < 0.1)
        & (lower_shadow / total_range > 0.6)
        & (upper_shadow / total_range < 0.1)
        & is_downtrend
    ).astype(int)

    # Inverted Hammer: small body, long upper shadow, little/no lower shadow, downtrend
    inverted_hammer = (
        (body / total_range < 0.1)
        & (upper_shadow / total_range > 0.6)
        & (lower_shadow / total_range < 0.1)
        & is_downtrend
    ).astype(int)

    return hammer, inverted_hammer


def detect_engulfing(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Bullish and Bearish Engulfing patterns

    Returns:
        (bullish_engulfing, bearish_engulfing)
    """
    body = np.abs(close - open_price)
    prev_body = np.abs(close.shift(1) - open_price.shift(1))

    # Bullish: Red (down) candle followed by Green (up) candle that engulfs it
    bullish_engulfing = (
        (close.shift(1) < open_price.shift(1))  # Previous red
        & (close > open_price)  # Current green
        & (open_price < close.shift(1))  # Opens below previous close
        & (close > open_price.shift(1))  # Closes above previous open
    ).astype(int)

    # Bearish: Green (up) candle followed by Red (down) candle that engulfs it
    bearish_engulfing = (
        (close.shift(1) > open_price.shift(1))  # Previous green
        & (close < open_price)  # Current red
        & (open_price > close.shift(1))  # Opens above previous close
        & (close < open_price.shift(1))  # Closes below previous open
    ).astype(int)

    return bullish_engulfing, bearish_engulfing


def detect_morning_star(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Detect Morning Star pattern (bullish reversal)
    """
    body1 = np.abs(close.shift(2) - open_price.shift(2))
    body2 = np.abs(close.shift(1) - open_price.shift(1))
    body3 = np.abs(close - open_price)
    range1 = high.shift(2) - low.shift(2)
    range2 = high.shift(1) - low.shift(1)
    range3 = high - low

    # Day 1: Large bearish candle
    day1_bearish = close.shift(2) < open_price.shift(2)
    day1_large = body1 / (range1 + 1e-10) > 0.5

    # Day 2: Small body (gap down preferred)
    day2_small = body2 / (range2 + 1e-10) < 0.3

    # Day 3: Large bullish candle
    day3_bullish = close > open_price
    day3_large = body3 / (range3 + 1e-10) > 0.5

    morning_star = (
        day1_bearish & day1_large & day2_small & day3_bullish & day3_large
    ).astype(int)
    return morning_star


def detect_evening_star(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Detect Evening Star pattern (bearish reversal)
    """
    body1 = np.abs(close.shift(2) - open_price.shift(2))
    body2 = np.abs(close.shift(1) - open_price.shift(1))
    body3 = np.abs(close - open_price)
    range1 = high.shift(2) - low.shift(2)
    range2 = high.shift(1) - low.shift(1)
    range3 = high - low

    # Day 1: Large bullish candle
    day1_bullish = close.shift(2) > open_price.shift(2)
    day1_large = body1 / (range1 + 1e-10) > 0.5

    # Day 2: Small body (gap up preferred)
    day2_small = body2 / (range2 + 1e-10) < 0.3

    # Day 3: Large bearish candle
    day3_bearish = close < open_price
    day3_large = body3 / (range3 + 1e-10) > 0.5

    evening_star = (
        day1_bullish & day1_large & day2_small & day3_bearish & day3_large
    ).astype(int)
    return evening_star


def detect_three_soldiers(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Detect Three White Soldiers pattern (strong bullish continuation)
    """
    # Three consecutive green candles with small bodies at higher prices
    day1_bullish = close.shift(2) > open_price.shift(2)
    day2_bullish = close.shift(1) > open_price.shift(1)
    day3_bullish = close > open_price

    # Each candle closes higher than previous
    higher_closes = (
        (close.shift(2) > close.shift(3))
        & (close.shift(1) > close.shift(2))
        & (close > close.shift(1))
    )

    # Small upper shadows (strong buying)
    body1 = close.shift(2) - open_price.shift(2)
    body2 = close.shift(1) - open_price.shift(1)
    body3 = close - open_price

    small_shadows = (
        ((high.shift(2) - close.shift(2)) / body1 < 0.3)
        & ((high.shift(1) - close.shift(1)) / body2 < 0.3)
        & ((high - close) / body3 < 0.3)
    )

    three_soldiers = (
        day1_bullish & day2_bullish & day3_bullish & higher_closes & small_shadows
    ).astype(int)
    return three_soldiers


def detect_three_crows(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Detect Three Black Crows pattern (strong bearish continuation)
    """
    # Three consecutive red candles
    day1_bearish = close.shift(2) < open_price.shift(2)
    day2_bearish = close.shift(1) < open_price.shift(1)
    day3_bearish = close < open_price

    # Each candle closes lower than previous
    lower_closes = (
        (close.shift(2) < close.shift(3))
        & (close.shift(1) < close.shift(2))
        & (close < close.shift(1))
    )

    # Small lower shadows (strong selling)
    body1 = open_price.shift(2) - close.shift(2)
    body2 = open_price.shift(1) - close.shift(1)
    body3 = open_price - close

    small_shadows = (
        ((close.shift(2) - low.shift(2)) / body1 < 0.3)
        & ((close.shift(1) - low.shift(1)) / body2 < 0.3)
        & ((close - low) / body3 < 0.3)
    )

    three_crows = (
        day1_bearish & day2_bearish & day3_bearish & lower_closes & small_shadows
    ).astype(int)
    return three_crows


def detect_inside_bar(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Detect Inside Bar pattern (consolidation)
    Current high/low within previous high/low
    """
    inside_bar = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(int)
    return inside_bar


def detect_outside_bar(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Detect Outside Bar pattern (breakout potential)
    Current high/low outside previous high/low
    """
    outside_bar = ((high > high.shift(1)) & (low < low.shift(1))).astype(int)
    return outside_bar


def detect_pin_bar(
    high: pd.Series, low: pd.Series, close: pd.Series, open_price: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Bullish Pin Bar (hammer-like) and Bearish Pin Bar (shooting star-like)

    Returns:
        (bullish_pin_bar, bearish_pin_bar)
    """
    body = np.abs(close - open_price)
    lower_shadow = np.minimum(open_price, close) - low
    upper_shadow = high - np.maximum(open_price, close)
    total_range = high - low + 1e-10

    # Bullish Pin Bar: long lower shadow, small body, small upper shadow
    bullish_pin_bar = (
        (body / total_range < 0.1)
        & (lower_shadow / total_range > 0.5)
        & (upper_shadow / total_range < 0.2)
    ).astype(int)

    # Bearish Pin Bar: long upper shadow, small body, small lower shadow
    bearish_pin_bar = (
        (body / total_range < 0.1)
        & (upper_shadow / total_range > 0.5)
        & (lower_shadow / total_range < 0.2)
    ).astype(int)

    return bullish_pin_bar, bearish_pin_bar


def calculate_pivot_points(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 5
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Pivot Points (S1, S2, R1, R2)

    Args:
        period: Period to aggregate OHLC for pivot calculation

    Returns:
        (pivot, s1, s2, r1, r2)
    """
    # Aggregate over period
    agg_high = high.rolling(window=period).max()
    agg_low = low.rolling(window=period).min()
    agg_close = close.shift(1)

    pivot = (agg_high + agg_low + agg_close) / 3

    # Support and Resistance levels
    r1 = (2 * pivot) - agg_low
    r2 = pivot + (agg_high - agg_low)
    s1 = (2 * pivot) - agg_high
    s2 = pivot - (agg_high - agg_low)

    return pivot, s1, s2, r1, r2


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all price pattern features to DataFrame

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added pattern features
    """
    df = df.copy()

    if not all(col in df.columns for col in ["open", "high", "low", "close"]):
        raise ValueError("Missing required OHLC columns")

    open_price = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # === Single Candlestick Patterns ===
    df["pattern_doji"] = detect_doji(open_price, high, low, close)

    hammer, inverted_hammer = detect_hammer(open_price, high, low, close)
    df["pattern_hammer"] = hammer
    df["pattern_inverted_hammer"] = inverted_hammer

    bullish_pin, bearish_pin = detect_pin_bar(high, low, close, open_price)
    df["pattern_bullish_pin"] = bullish_pin
    df["pattern_bearish_pin"] = bearish_pin

    # === Multi-Candle Patterns ===
    bullish_engulfing, bearish_engulfing = detect_engulfing(
        open_price, high, low, close
    )
    df["pattern_bullish_engulfing"] = bullish_engulfing
    df["pattern_bearish_engulfing"] = bearish_engulfing

    df["pattern_morning_star"] = detect_morning_star(open_price, high, low, close)
    df["pattern_evening_star"] = detect_evening_star(open_price, high, low, close)
    df["pattern_three_soldiers"] = detect_three_soldiers(open_price, high, low, close)
    df["pattern_three_crows"] = detect_three_crows(open_price, high, low, close)

    # === Price Action Patterns ===
    df["pattern_inside_bar"] = detect_inside_bar(high, low, close)
    df["pattern_outside_bar"] = detect_outside_bar(high, low, close)

    # === Gap Patterns ===
    if "volume" in df.columns:
        gap_up = (open_price / close.shift(1) - 1) > 0.02
        gap_down = (open_price / close.shift(1) - 1) < -0.02
        volume_spike = df["volume"] > df["volume"].rolling(20).mean() * 1.5

        df["pattern_gap_up_volume"] = (gap_up & volume_spike).astype(int)
        df["pattern_gap_down_volume"] = (gap_down & volume_spike).astype(int)
        df["pattern_gap_up"] = gap_up.astype(int)
        df["pattern_gap_down"] = gap_down.astype(int)

        # Gap fill detection
        gap_filled_up = gap_up.shift(1) & (low <= close.shift(2))
        gap_filled_down = gap_down.shift(1) & (high >= close.shift(2))
        df["pattern_gap_filled"] = (gap_filled_up | gap_filled_down).astype(int)

    # === Pivot Points ===
    pivot, s1, s2, r1, r2 = calculate_pivot_points(high, low, close)

    df["pivot_point"] = pivot
    df["pivot_s1"] = s1
    df["pivot_s2"] = s2
    df["pivot_r1"] = r1
    df["pivot_r2"] = r2

    # Position relative to pivots
    df["pivot_position"] = (close - pivot) / (pivot + 1e-10)
    df["near_pivot_r1"] = (np.abs(close - r1) / r1 < 0.01).astype(int)
    df["near_pivot_r2"] = (np.abs(close - r2) / r2 < 0.01).astype(int)
    df["near_pivot_s1"] = (np.abs(close - s1) / s1 < 0.01).astype(int)
    df["near_pivot_s2"] = (np.abs(close - s2) / s2 < 0.01).astype(int)

    # === Price Zones (20-day high/low clusters) ===
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()

    df["price_zone"] = (close - low_20) / (high_20 - low_20 + 1e-10)
    df["in_upper_zone"] = (df["price_zone"] > 0.8).astype(int)
    df["in_lower_zone"] = (df["price_zone"] < 0.2).astype(int)
    df["in_middle_zone"] = (
        (df["price_zone"] >= 0.4) & (df["price_zone"] <= 0.6)
    ).astype(int)

    # === Pattern Confluence ===
    bullish_signals = (
        df["pattern_bullish_engulfing"]
        + df["pattern_morning_star"]
        + df["pattern_three_soldiers"]
        + df["pattern_hammer"]
        + df["pattern_bullish_pin"]
    )

    bearish_signals = (
        df["pattern_bearish_engulfing"]
        + df["pattern_evening_star"]
        + df["pattern_three_crows"]
        + df["pattern_inverted_hammer"]
        + df["pattern_bearish_pin"]
    )

    df["pattern_bullish_score"] = bullish_signals
    df["pattern_bearish_score"] = bearish_signals
    df["pattern_net_score"] = bullish_signals - bearish_signals

    return df


# List of features added by this module
PATTERN_FEATURES = [
    # Single candlestick
    "pattern_doji",
    "pattern_hammer",
    "pattern_inverted_hammer",
    "pattern_bullish_pin",
    "pattern_bearish_pin",
    # Multi-candle
    "pattern_bullish_engulfing",
    "pattern_bearish_engulfing",
    "pattern_morning_star",
    "pattern_evening_star",
    "pattern_three_soldiers",
    "pattern_three_crows",
    # Price action
    "pattern_inside_bar",
    "pattern_outside_bar",
    # Gaps
    "pattern_gap_up",
    "pattern_gap_down",
    "pattern_gap_up_volume",
    "pattern_gap_down_volume",
    "pattern_gap_filled",
    # Pivots
    "pivot_point",
    "pivot_s1",
    "pivot_s2",
    "pivot_r1",
    "pivot_r2",
    "pivot_position",
    "near_pivot_r1",
    "near_pivot_r2",
    "near_pivot_s1",
    "near_pivot_s2",
    # Zones
    "price_zone",
    "in_upper_zone",
    "in_lower_zone",
    "in_middle_zone",
    # Confluence
    "pattern_bullish_score",
    "pattern_bearish_score",
    "pattern_net_score",
]
