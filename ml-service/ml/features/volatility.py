"""
Volatility features for ETF Stock Prediction
"""
import pandas as pd
import numpy as np

from ..config import config


def calculate_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """Calculate True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range"""
    tr = calculate_true_range(high, low, close)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple:
    """
    Calculate Bollinger Bands

    Returns:
        Tuple of (upper, middle, lower, width, position)
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    # Band width (volatility measure)
    width = (upper - lower) / middle

    # Position within bands (0 = at lower, 1 = at upper)
    position = (close - lower) / (upper - lower + 1e-10)

    return upper, middle, lower, width, position


def calculate_keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    multiplier: float = 2.0
) -> tuple:
    """
    Calculate Keltner Channel

    Returns:
        Tuple of (upper, middle, lower)
    """
    typical_price = (high + low + close) / 3
    middle = typical_price.ewm(span=period, adjust=False).mean()

    atr = calculate_atr(high, low, close, period)

    upper = middle + multiplier * atr
    lower = middle - multiplier * atr

    return upper, middle, lower


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility features to DataFrame

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added volatility features
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ['high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    close = df['close']
    high = df['high']
    low = df['low']

    # === Bollinger Bands ===
    bb_upper, bb_middle, bb_lower, bb_width, bb_position = calculate_bollinger_bands(
        close,
        config.features.bb_period,
        config.features.bb_std
    )
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = bb_width
    df['bb_position'] = bb_position

    # BB squeeze (width relative to recent width)
    df['bb_squeeze'] = bb_width / bb_width.rolling(window=50).mean()

    # === ATR (Average True Range) ===
    atr = calculate_atr(high, low, close, config.features.atr_period)
    df['atr_14'] = atr

    # ATR as percentage of price (normalized volatility)
    df['atr_ratio'] = atr / close

    # === Historical Volatility ===
    # Annualized volatility of returns
    daily_return = close.pct_change()

    for period in config.features.volatility_periods:
        df[f'volatility_{period}'] = (
            daily_return.rolling(window=period).std() * np.sqrt(252)
        )

    # === Volatility ratio (short-term vs long-term) ===
    if 'volatility_10' in df.columns and 'volatility_63' in df.columns:
        df['vol_ratio'] = df['volatility_10'] / (df['volatility_63'] + 1e-10)

    # === Keltner Channel ===
    kc_upper, kc_middle, kc_lower = calculate_keltner_channel(
        high, low, close, 20, 2.0
    )
    df['kc_upper'] = kc_upper
    df['kc_middle'] = kc_middle
    df['kc_lower'] = kc_lower

    # Keltner position
    df['kc_position'] = (close - kc_lower) / (kc_upper - kc_lower + 1e-10)

    # === Intraday Volatility ===
    # High-Low range as % of close
    df['intraday_range'] = (high - low) / close

    # Average intraday range
    df['avg_intraday_range_10'] = df['intraday_range'].rolling(window=10).mean()
    df['avg_intraday_range_20'] = df['intraday_range'].rolling(window=20).mean()

    # === Gap (Open vs Previous Close) ===
    if 'open' in df.columns:
        df['gap'] = df['open'] / close.shift(1) - 1

        # Average absolute gap
        df['avg_gap_20'] = df['gap'].abs().rolling(window=20).mean()

    # === Parkinson Volatility (uses high-low range) ===
    # More efficient estimator than close-to-close
    log_hl = np.log(high / low)
    df['parkinson_vol_20'] = np.sqrt(
        (1 / (4 * np.log(2))) *
        (log_hl ** 2).rolling(window=20).mean()
    ) * np.sqrt(252)

    # === Yang-Zhang Volatility (more accurate) ===
    # Combines overnight and intraday volatility
    if 'open' in df.columns:
        log_co = np.log(close / df['open'])
        log_oc = np.log(df['open'] / close.shift(1))

        # Overnight variance
        overnight_var = log_oc.rolling(window=20).var()
        # Open-to-close variance
        oc_var = log_co.rolling(window=20).var()
        # Rogers-Satchell variance
        log_ho = np.log(high / df['open'])
        log_lo = np.log(low / df['open'])
        rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window=20).mean()

        k = 0.34 / (1.34 + (20 + 1) / (20 - 1))
        yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
        df['yz_vol_20'] = np.sqrt(yz_var) * np.sqrt(252)

    return df


# List of features added by this module
VOLATILITY_FEATURES = [
    # Bollinger Bands
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position', 'bb_squeeze',
    # ATR
    'atr_14', 'atr_ratio',
    # Historical volatility
    'volatility_10', 'volatility_20', 'volatility_63', 'vol_ratio',
    # Keltner Channel
    'kc_upper', 'kc_middle', 'kc_lower', 'kc_position',
    # Intraday volatility
    'intraday_range', 'avg_intraday_range_10', 'avg_intraday_range_20',
    # Gap
    'gap', 'avg_gap_20',
    # Advanced volatility
    'parkinson_vol_20', 'yz_vol_20'
]
