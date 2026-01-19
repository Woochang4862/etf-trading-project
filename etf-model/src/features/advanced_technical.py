"""
Advanced Technical Indicators for ETF Stock Prediction

Includes:
- Ichimoku Cloud components
- Keltner Channels
- Donchian Channels
- Chandelier Exit
- Parabolic SAR
- SuperTrend
- VWAP (Volume Weighted Average Price)
- Elder Ray (Bull/Bear Power)
- Coppock Curve
- Ultimate Oscillator
"""

import numpy as np
import pandas as pd
from typing import List


# Feature names for external reference
ADVANCED_TECHNICAL_FEATURES: List[str] = [
    # Ichimoku Cloud
    "ichimoku_tenkan",
    "ichimoku_kijun",
    "ichimoku_senkou_a",
    "ichimoku_senkou_b",
    "ichimoku_chikou",
    "ichimoku_cloud_thickness",
    "ichimoku_price_vs_cloud",
    "ichimoku_tk_cross",
    # Keltner Channels
    "keltner_upper",
    "keltner_middle",
    "keltner_lower",
    "keltner_position",
    "keltner_width",
    # Donchian Channels
    "donchian_upper",
    "donchian_middle",
    "donchian_lower",
    "donchian_position",
    "donchian_width",
    # Chandelier Exit
    "chandelier_long",
    "chandelier_short",
    "chandelier_signal",
    # Parabolic SAR
    "psar_value",
    "psar_trend",
    "psar_distance",
    # SuperTrend
    "supertrend_value",
    "supertrend_direction",
    "supertrend_distance",
    # VWAP
    "vwap",
    "vwap_distance",
    "vwap_std_band_upper",
    "vwap_std_band_lower",
    # Elder Ray
    "elder_bull_power",
    "elder_bear_power",
    "elder_ray_signal",
    # Coppock Curve
    "coppock_curve",
    "coppock_signal",
    # Ultimate Oscillator
    "ultimate_oscillator",
    # Mass Index
    "mass_index",
    "mass_index_signal",
    # Ease of Movement
    "eom",
    "eom_sma",
    # Chaikin Money Flow
    "cmf",
    # Force Index
    "force_index",
    "force_index_ema",
]


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period, min_periods=1).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate Average True Range"""
    tr = _true_range(high, low, close)
    return tr.rolling(window=period, min_periods=1).mean()


def add_ichimoku_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Ichimoku Cloud indicators

    - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods forward
    - Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods
    - Chikou Span (Lagging Span): Close, shifted 26 periods backward
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Tenkan-sen (9-period)
    tenkan_high = high.rolling(window=9, min_periods=1).max()
    tenkan_low = low.rolling(window=9, min_periods=1).min()
    df["ichimoku_tenkan"] = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (26-period)
    kijun_high = high.rolling(window=26, min_periods=1).max()
    kijun_low = low.rolling(window=26, min_periods=1).min()
    df["ichimoku_kijun"] = (kijun_high + kijun_low) / 2

    # Senkou Span A (shifted forward 26 periods - but we use current for features)
    df["ichimoku_senkou_a"] = (df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2

    # Senkou Span B (52-period)
    senkou_high = high.rolling(window=52, min_periods=1).max()
    senkou_low = low.rolling(window=52, min_periods=1).min()
    df["ichimoku_senkou_b"] = (senkou_high + senkou_low) / 2

    # Chikou Span (close shifted backward - use current close for features)
    df["ichimoku_chikou"] = close

    # Derived features
    df["ichimoku_cloud_thickness"] = (
        df["ichimoku_senkou_a"] - df["ichimoku_senkou_b"]
    ) / close

    # Price position vs cloud
    cloud_top = df[["ichimoku_senkou_a", "ichimoku_senkou_b"]].max(axis=1)
    cloud_bottom = df[["ichimoku_senkou_a", "ichimoku_senkou_b"]].min(axis=1)
    df["ichimoku_price_vs_cloud"] = np.where(
        close > cloud_top, 1, np.where(close < cloud_bottom, -1, 0)
    )

    # Tenkan/Kijun cross signal
    df["ichimoku_tk_cross"] = np.where(
        df["ichimoku_tenkan"] > df["ichimoku_kijun"],
        1,
        np.where(df["ichimoku_tenkan"] < df["ichimoku_kijun"], -1, 0),
    )

    return df


def add_keltner_channel_features(
    df: pd.DataFrame, period: int = 20, multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Add Keltner Channel indicators

    - Middle: EMA of close
    - Upper: Middle + multiplier * ATR
    - Lower: Middle - multiplier * ATR
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Middle band (EMA)
    df["keltner_middle"] = _ema(close, period)

    # ATR
    atr = _atr(high, low, close, period)

    # Upper and lower bands
    df["keltner_upper"] = df["keltner_middle"] + multiplier * atr
    df["keltner_lower"] = df["keltner_middle"] - multiplier * atr

    # Position within channel (0 = lower, 1 = upper)
    channel_width = df["keltner_upper"] - df["keltner_lower"]
    df["keltner_position"] = np.where(
        channel_width > 0, (close - df["keltner_lower"]) / channel_width, 0.5
    )

    # Channel width as percentage of price
    df["keltner_width"] = channel_width / close

    return df


def add_donchian_channel_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Add Donchian Channel indicators

    - Upper: Highest high over period
    - Lower: Lowest low over period
    - Middle: Average of upper and lower
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    df["donchian_upper"] = high.rolling(window=period, min_periods=1).max()
    df["donchian_lower"] = low.rolling(window=period, min_periods=1).min()
    df["donchian_middle"] = (df["donchian_upper"] + df["donchian_lower"]) / 2

    # Position within channel
    channel_width = df["donchian_upper"] - df["donchian_lower"]
    df["donchian_position"] = np.where(
        channel_width > 0, (close - df["donchian_lower"]) / channel_width, 0.5
    )

    # Channel width as percentage
    df["donchian_width"] = channel_width / close

    return df


def add_chandelier_exit_features(
    df: pd.DataFrame, period: int = 22, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Add Chandelier Exit indicators

    - Long: Highest high - multiplier * ATR
    - Short: Lowest low + multiplier * ATR
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    atr = _atr(high, low, close, period)
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()

    df["chandelier_long"] = highest_high - multiplier * atr
    df["chandelier_short"] = lowest_low + multiplier * atr

    # Signal: 1 if close > long exit (bullish), -1 if close < short exit (bearish)
    df["chandelier_signal"] = np.where(
        close > df["chandelier_long"],
        1,
        np.where(close < df["chandelier_short"], -1, 0),
    )

    return df


def add_parabolic_sar_features(
    df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2
) -> pd.DataFrame:
    """
    Add Parabolic SAR indicator (simplified version)

    Uses a simplified calculation for efficiency
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    psar = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    af = np.full(n, af_start)
    ep = np.zeros(n)  # Extreme point

    # Initialize
    psar[0] = low[0]
    ep[0] = high[0]

    for i in range(1, n):
        if trend[i - 1] == 1:  # Uptrend
            psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])
            psar[i] = min(psar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])

            if low[i] < psar[i]:  # Trend reversal
                trend[i] = -1
                psar[i] = ep[i - 1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af[i] = min(af[i - 1] + af_step, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
        else:  # Downtrend
            psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])
            psar[i] = max(psar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])

            if high[i] > psar[i]:  # Trend reversal
                trend[i] = 1
                psar[i] = ep[i - 1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + af_step, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]

    df["psar_value"] = psar
    df["psar_trend"] = trend
    df["psar_distance"] = (close - psar) / close

    return df


def add_supertrend_features(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Add SuperTrend indicator

    SuperTrend = (High + Low) / 2 ± multiplier * ATR
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # SuperTrend calculation
    supertrend = np.zeros(len(df))
    direction = np.ones(len(df))  # 1 = uptrend, -1 = downtrend

    close_vals = close.values
    upper_vals = upper_band.values
    lower_vals = lower_band.values

    for i in range(1, len(df)):
        if close_vals[i] > upper_vals[i - 1]:
            direction[i] = 1
        elif close_vals[i] < lower_vals[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        if direction[i] == 1:
            supertrend[i] = lower_vals[i]
        else:
            supertrend[i] = upper_vals[i]

    df["supertrend_value"] = supertrend
    df["supertrend_direction"] = direction
    df["supertrend_distance"] = (close_vals - supertrend) / close_vals

    return df


def add_vwap_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Add VWAP (Volume Weighted Average Price) features

    Rolling VWAP for intraday-like analysis
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"]

    # Rolling VWAP
    tp_vol = typical_price * volume
    cum_tp_vol = tp_vol.rolling(window=period, min_periods=1).sum()
    cum_vol = volume.rolling(window=period, min_periods=1).sum()

    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

    # VWAP standard deviation bands
    squared_diff = ((typical_price - df["vwap"]) ** 2) * volume
    variance = squared_diff.rolling(
        window=period, min_periods=1
    ).sum() / cum_vol.replace(0, np.nan)
    vwap_std = np.sqrt(variance)

    df["vwap_std_band_upper"] = (df["vwap"] + 2 * vwap_std - df["close"]) / df["close"]
    df["vwap_std_band_lower"] = (df["close"] - df["vwap"] + 2 * vwap_std) / df["close"]

    return df


def add_elder_ray_features(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """
    Add Elder Ray (Bull/Bear Power) indicators

    - Bull Power = High - EMA
    - Bear Power = Low - EMA
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema = _ema(close, period)

    df["elder_bull_power"] = (high - ema) / close
    df["elder_bear_power"] = (low - ema) / close

    # Signal: Bull power positive and bear power increasing
    df["elder_ray_signal"] = np.where(
        (df["elder_bull_power"] > 0)
        & (df["elder_bear_power"] > df["elder_bear_power"].shift(1)),
        1,
        np.where(
            (df["elder_bear_power"] < 0)
            & (df["elder_bull_power"] < df["elder_bull_power"].shift(1)),
            -1,
            0,
        ),
    )

    return df


def add_coppock_curve_features(
    df: pd.DataFrame, roc1: int = 14, roc2: int = 11, wma_period: int = 10
) -> pd.DataFrame:
    """
    Add Coppock Curve indicator

    Coppock = WMA(ROC(14) + ROC(11), 10)
    """
    close = df["close"]

    # Rate of Change
    roc_14 = (close - close.shift(roc1)) / close.shift(roc1) * 100
    roc_11 = (close - close.shift(roc2)) / close.shift(roc2) * 100

    # Weighted Moving Average
    weights = np.arange(1, wma_period + 1)
    combined_roc = roc_14 + roc_11

    df["coppock_curve"] = combined_roc.rolling(window=wma_period, min_periods=1).apply(
        lambda x: np.dot(x, weights[-len(x) :]) / weights[-len(x) :].sum(), raw=True
    )

    # Signal: Coppock crosses above zero
    df["coppock_signal"] = np.where(
        (df["coppock_curve"] > 0) & (df["coppock_curve"].shift(1) <= 0),
        1,
        np.where(
            (df["coppock_curve"] < 0) & (df["coppock_curve"].shift(1) >= 0), -1, 0
        ),
    )

    return df


def add_ultimate_oscillator_features(
    df: pd.DataFrame, short: int = 7, medium: int = 14, long: int = 28
) -> pd.DataFrame:
    """
    Add Ultimate Oscillator

    Combines short, medium, and long-term price momentum
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    # Buying Pressure
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)

    # True Range
    tr = _true_range(high, low, close)

    # Averages
    bp_sum_short = bp.rolling(window=short, min_periods=1).sum()
    tr_sum_short = tr.rolling(window=short, min_periods=1).sum()

    bp_sum_medium = bp.rolling(window=medium, min_periods=1).sum()
    tr_sum_medium = tr.rolling(window=medium, min_periods=1).sum()

    bp_sum_long = bp.rolling(window=long, min_periods=1).sum()
    tr_sum_long = tr.rolling(window=long, min_periods=1).sum()

    # Ultimate Oscillator
    avg_short = bp_sum_short / tr_sum_short.replace(0, np.nan)
    avg_medium = bp_sum_medium / tr_sum_medium.replace(0, np.nan)
    avg_long = bp_sum_long / tr_sum_long.replace(0, np.nan)

    df["ultimate_oscillator"] = 100 * (4 * avg_short + 2 * avg_medium + avg_long) / 7

    return df


def add_mass_index_features(
    df: pd.DataFrame, ema_period: int = 9, sum_period: int = 25
) -> pd.DataFrame:
    """
    Add Mass Index indicator

    Identifies trend reversals based on range expansion
    """
    high = df["high"]
    low = df["low"]

    hl_range = high - low
    ema1 = _ema(hl_range, ema_period)
    ema2 = _ema(ema1, ema_period)

    ratio = ema1 / ema2.replace(0, np.nan)
    df["mass_index"] = ratio.rolling(window=sum_period, min_periods=1).sum()

    # Reversal bulge signal (>27 then drops below 26.5)
    df["mass_index_signal"] = np.where(
        (df["mass_index"].shift(1) > 27) & (df["mass_index"] < 26.5), 1, 0
    )

    return df


def add_eom_features(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Ease of Movement indicator

    Measures relationship between price change and volume
    """
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume / 1e8) / (high - low).replace(0, np.nan)

    df["eom"] = distance / box_ratio.replace(0, np.nan)
    df["eom_sma"] = _sma(df["eom"], period)

    return df


def add_cmf_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Add Chaikin Money Flow indicator

    Measures buying/selling pressure over a period
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)

    # Money Flow Volume
    mfv = mfm * volume

    # Chaikin Money Flow
    df["cmf"] = mfv.rolling(window=period, min_periods=1).sum() / volume.rolling(
        window=period, min_periods=1
    ).sum().replace(0, np.nan)

    return df


def add_force_index_features(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """
    Add Force Index indicator

    Measures the force behind price movements
    """
    close = df["close"]
    volume = df["volume"]

    df["force_index"] = (close - close.shift(1)) * volume
    df["force_index_ema"] = _ema(df["force_index"], period) / close  # Normalize

    return df


def add_advanced_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all advanced technical features

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with additional advanced technical features
    """
    df = add_ichimoku_features(df)
    df = add_keltner_channel_features(df)
    df = add_donchian_channel_features(df)
    df = add_chandelier_exit_features(df)
    df = add_parabolic_sar_features(df)
    df = add_supertrend_features(df)
    df = add_vwap_features(df)
    df = add_elder_ray_features(df)
    df = add_coppock_curve_features(df)
    df = add_ultimate_oscillator_features(df)
    df = add_mass_index_features(df)
    df = add_eom_features(df)
    df = add_cmf_features(df)
    df = add_force_index_features(df)

    return df
