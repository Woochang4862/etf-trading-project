"""
Volume-based features for ETF Stock Prediction
"""
import pandas as pd
import numpy as np

from ..config import config


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV)"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0

    obv = (volume * direction).cumsum()
    return obv


def calculate_ad(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """Calculate Accumulation/Distribution Line"""
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    ad = (clv * volume).cumsum()
    return ad


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Money Flow Index (MFI)"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    # Positive and negative money flow
    tp_diff = typical_price.diff()
    pos_flow = raw_money_flow.where(tp_diff > 0, 0)
    neg_flow = raw_money_flow.where(tp_diff < 0, 0)

    # Rolling sums
    pos_sum = pos_flow.rolling(window=period).sum()
    neg_sum = neg_flow.rolling(window=period).sum()

    # Money ratio and MFI
    money_ratio = pos_sum / (neg_sum + 1e-10)
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def calculate_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """Calculate Chaikin Money Flow (CMF)"""
    mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * volume
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """Calculate Volume Weighted Average Price (rolling)"""
    typical_price = (high + low + close) / 3
    vwap = (
        (typical_price * volume).rolling(window=period).sum() /
        volume.rolling(window=period).sum()
    )
    return vwap


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features to DataFrame

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added volume features
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ['high', 'low', 'close', 'volume']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # === On-Balance Volume (OBV) ===
    df['obv'] = calculate_obv(close, volume)

    # OBV trend (OBV relative to its MA)
    df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
    df['obv_trend'] = df['obv'] / (df['obv_sma_20'] + 1e-10) - 1

    # OBV slope
    df['obv_slope'] = df['obv'].diff(5) / (df['obv'].shift(5).abs() + 1e-10)

    # === Accumulation/Distribution ===
    df['ad'] = calculate_ad(high, low, close, volume)

    # AD trend
    df['ad_sma_20'] = df['ad'].rolling(window=20).mean()
    df['ad_trend'] = df['ad'] / (df['ad_sma_20'].abs() + 1e-10) - 1

    # === Chaikin A/D Oscillator (ADOSC) ===
    # Fast EMA - Slow EMA of AD
    ad_ema_3 = df['ad'].ewm(span=3, adjust=False).mean()
    ad_ema_10 = df['ad'].ewm(span=10, adjust=False).mean()
    df['adosc'] = ad_ema_3 - ad_ema_10

    # === Volume Moving Averages ===
    for period in config.features.volume_ma_periods:
        df[f'volume_sma_{period}'] = volume.rolling(window=period).mean()

    # Volume ratio (current vs average)
    df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-10)

    # Volume spike detection
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)

    # === Money Flow Index (MFI) ===
    df['mfi'] = calculate_mfi(
        high, low, close, volume,
        config.features.mfi_period
    )

    # === Chaikin Money Flow (CMF) ===
    df['cmf'] = calculate_cmf(high, low, close, volume, 20)

    # === VWAP ===
    df['vwap_20'] = calculate_vwap(high, low, close, volume, 20)
    df['price_to_vwap'] = close / df['vwap_20'] - 1

    # === Volume Trend ===
    # Is volume increasing with price?
    price_up = close.diff() > 0
    volume_up = volume.diff() > 0

    # Healthy uptrend: price up + volume up
    df['vol_price_confirm'] = (price_up & volume_up).astype(int)

    # Distribution: price up + volume down (weakness)
    df['vol_price_diverge'] = (price_up & ~volume_up).astype(int)

    # === Volume Relative Strength ===
    # Compare volume to price movement
    price_change = close.pct_change().abs()
    volume_change = volume.pct_change().abs()
    df['vol_price_strength'] = volume_change / (price_change + 1e-10)

    # === Average Dollar Volume ===
    dollar_volume = close * volume
    df['dollar_volume_20'] = dollar_volume.rolling(window=20).mean()

    # Normalized dollar volume
    df['dollar_volume_ratio'] = dollar_volume / (df['dollar_volume_20'] + 1e-10)

    # === Volume Volatility ===
    df['volume_std_20'] = volume.rolling(window=20).std()
    df['volume_cv'] = df['volume_std_20'] / (df['volume_sma_20'] + 1e-10)  # Coefficient of variation

    return df


# List of features added by this module
VOLUME_FEATURES = [
    # OBV
    'obv', 'obv_sma_20', 'obv_trend', 'obv_slope',
    # AD
    'ad', 'ad_sma_20', 'ad_trend', 'adosc',
    # Volume MA
    'volume_sma_10', 'volume_sma_20', 'volume_ratio', 'volume_spike',
    # Money flow
    'mfi', 'cmf',
    # VWAP
    'vwap_20', 'price_to_vwap',
    # Volume-Price
    'vol_price_confirm', 'vol_price_diverge', 'vol_price_strength',
    # Dollar volume
    'dollar_volume_20', 'dollar_volume_ratio',
    # Volume volatility
    'volume_std_20', 'volume_cv'
]
