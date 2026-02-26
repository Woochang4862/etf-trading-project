"""
Technical indicator features for ETF Stock Prediction
Using 'ta' library (pure Python, no C dependencies)
"""
import pandas as pd
import numpy as np
from typing import Optional

try:
    import ta
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.trend import MACD, ADXIndicator, AroonIndicator
    HAS_TA = True
except ImportError:
    HAS_TA = False

from ..config import config


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index)

    Manual implementation as fallback
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use EMA for smoother RSI
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Returns:
        Tuple of (macd, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> tuple:
    """
    Calculate Stochastic Oscillator

    Returns:
        Tuple of (stoch_k, stoch_d)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return willr


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> tuple:
    """
    Calculate ADX (Average Directional Index)

    Returns:
        Tuple of (adx, plus_di, minus_di)
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smoothed averages
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx, plus_di, minus_di


def calculate_aroon(
    high: pd.Series,
    low: pd.Series,
    period: int = 25
) -> tuple:
    """
    Calculate Aroon Indicator

    Returns:
        Tuple of (aroon_up, aroon_down, aroon_osc)
    """
    aroon_up = high.rolling(window=period + 1).apply(
        lambda x: 100 * (period - (period - x.argmax())) / period,
        raw=True
    )
    aroon_down = low.rolling(window=period + 1).apply(
        lambda x: 100 * (period - (period - x.argmin())) / period,
        raw=True
    )
    aroon_osc = aroon_up - aroon_down

    return aroon_up, aroon_down, aroon_osc


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator features to DataFrame

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added technical features
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    close = df['close']
    high = df['high']
    low = df['low']

    # === RSI ===
    for period in config.features.rsi_periods:
        if HAS_TA:
            indicator = RSIIndicator(close=close, window=period)
            df[f'rsi_{period}'] = indicator.rsi()
        else:
            df[f'rsi_{period}'] = calculate_rsi(close, period)

    # === MACD ===
    if HAS_TA:
        macd_indicator = MACD(
            close=close,
            window_slow=config.features.macd_slow,
            window_fast=config.features.macd_fast,
            window_sign=config.features.macd_signal
        )
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
    else:
        macd, macd_sig, macd_hist = calculate_macd(
            close,
            config.features.macd_fast,
            config.features.macd_slow,
            config.features.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = macd_sig
        df['macd_hist'] = macd_hist

    # === Stochastic Oscillator ===
    if HAS_TA:
        stoch = StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=config.features.stoch_k_period,
            smooth_window=config.features.stoch_d_period
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
    else:
        stoch_k, stoch_d = calculate_stochastic(
            high, low, close,
            config.features.stoch_k_period,
            config.features.stoch_d_period
        )
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

    # === Williams %R ===
    if HAS_TA:
        willr = WilliamsRIndicator(high=high, low=low, close=close, lbp=14)
        df['willr'] = willr.williams_r()
    else:
        df['willr'] = calculate_williams_r(high, low, close, 14)

    # === ADX ===
    if HAS_TA:
        adx_indicator = ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=config.features.adx_period
        )
        df['adx'] = adx_indicator.adx()
        df['plus_di'] = adx_indicator.adx_pos()
        df['minus_di'] = adx_indicator.adx_neg()
    else:
        adx, plus_di, minus_di = calculate_adx(
            high, low, close,
            config.features.adx_period
        )
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

    # === Aroon ===
    if HAS_TA:
        aroon = AroonIndicator(
            high=high,
            low=low,
            window=config.features.aroon_period
        )
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_osc'] = aroon.aroon_indicator()
    else:
        aroon_up, aroon_down, aroon_osc = calculate_aroon(
            high, low,
            config.features.aroon_period
        )
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        df['aroon_osc'] = aroon_osc

    return df


# List of features added by this module
TECHNICAL_FEATURES = [
    'rsi_7', 'rsi_14', 'rsi_21',
    'macd', 'macd_signal', 'macd_hist',
    'stoch_k', 'stoch_d',
    'willr',
    'adx', 'plus_di', 'minus_di',
    'aroon_up', 'aroon_down', 'aroon_osc'
]
