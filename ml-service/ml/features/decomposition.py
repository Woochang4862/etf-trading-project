"""
Time-series decomposition features for ETF Stock Prediction

Extracts trend, cycle, and seasonal components using various decomposition methods.
"""

import pandas as pd
import numpy as np
from typing import Tuple

# Optional scipy imports
try:
    from scipy import signal
    from scipy.signal import savgol_filter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def hp_filter(series: pd.Series, lamb: float = 100000) -> Tuple[pd.Series, pd.Series]:
    """
    Hodrick-Prescott filter for trend and cycle decomposition

    Args:
        series: Time series
        lamb: Smoothing parameter (higher = smoother trend)

    Returns:
        (trend, cycle)
    """
    n = len(series)
    # Second difference matrix: D2 is (n-2) x n
    # Create second difference operator properly
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1
    D2T = D2.T

    # Solve (I + lambda*D2'D2)y = trend
    # D2T @ D2 gives (n x n) matrix
    trend = np.linalg.inv(np.eye(n) + lamb * D2T @ D2) @ series.values
    cycle = series.values - trend

    return pd.Series(trend, index=series.index), pd.Series(cycle, index=series.index)


def detect_dominant_cycle(
    series: pd.Series, min_period: int = 20, max_period: int = 126
) -> Tuple[float, float]:
    """
    Detect dominant cycle using FFT (Fast Fourier Transform)

    Args:
        series: Time series
        min_period: Minimum cycle period to detect
        max_period: Maximum cycle period to detect

    Returns:
        (dominant_period, cycle_strength)
    """
    # Remove NaN and center
    clean_series = series.dropna()
    centered = clean_series - clean_series.mean()

    # FFT
    n = len(centered)
    fft_values = np.fft.fft(centered.values)
    fft_freq = np.fft.fftfreq(n)

    # Power spectrum
    power = np.abs(fft_values) ** 2

    # Find dominant frequency in valid range
    freq_mask = (np.abs(fft_freq) >= 1 / max_period) & (
        np.abs(fft_freq) <= 1 / min_period
    )
    valid_power = power * freq_mask

    if np.sum(valid_power) == 0:
        return np.nan, np.nan

    dominant_idx = np.argmax(valid_power)
    dominant_freq = np.abs(fft_freq[dominant_idx])

    # Convert frequency to period
    dominant_period = 1 / dominant_freq if dominant_freq != 0 else np.nan
    cycle_strength = valid_power[dominant_idx] / np.sum(power)

    return dominant_period, cycle_strength


def add_decomposition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add decomposition features to DataFrame

    Args:
        df: DataFrame with price columns

    Returns:
        DataFrame with added decomposition features
    """
    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    close = df["close"]
    daily_ret = close.pct_change()

    # === HP Filter Decomposition ===
    # Trend component
    hp_trend, hp_cycle = hp_filter(close, lamb=10000)

    df["hp_trend"] = hp_trend
    df["hp_cycle"] = hp_cycle

    # Cycle amplitude (normalized)
    cycle_range = hp_cycle.rolling(63).max() - hp_cycle.rolling(63).min()
    df["cycle_amplitude"] = np.abs(hp_cycle) / (cycle_range + 1e-10)

    # Position in cycle (0-1)
    cycle_min = hp_cycle.rolling(63).min()
    cycle_max = hp_cycle.rolling(63).max()
    df["cycle_position"] = (hp_cycle - cycle_min) / (cycle_max - cycle_min + 1e-10)

    # Cycle phase (sine approximation)
    df["cycle_phase"] = np.sin(2 * np.pi * df["cycle_position"])

    # === Dominant Cycle Detection (scipy optional) ===
    if HAS_SCIPY:
        # Rolling window for cycle detection
        window_size = 252
        for i in range(0, len(close) - window_size + 1, 63):
            window = close.iloc[i : i + window_size]
            period, strength = detect_dominant_cycle(window)

            if i == 0:
                periods = [np.nan] * i + [period]
                strengths = [np.nan] * i + [strength]
            else:
                periods.append(period)
                strengths.append(strength)

        periods += [np.nan] * (len(close) - len(periods))
        strengths += [np.nan] * (len(close) - len(strengths))

        df["dominant_cycle_period"] = pd.Series(periods, index=close.index)
        df["cycle_strength"] = pd.Series(strengths, index=close.index)
    else:
        # Fallback: Use rolling momentum as proxy for cycle
        df["dominant_cycle_period"] = (
            df["ret_20d"].rolling(126).std() * 4
        )  # Approximate
        df["cycle_strength"] = 0.5

    # === Detrended Returns ===
    # Returns minus trend component
    if "hp_trend" in df.columns:
        trend_ret = df["hp_trend"].pct_change()
        df["detrended_ret_5d"] = (
            df["ret_5d"] - trend_ret * 5 if "ret_5d" in df.columns else np.nan
        )
        df["detrended_ret_20d"] = (
            df["ret_20d"] - trend_ret * 20 if "ret_20d" in df.columns else np.nan
        )

    # === Smoothed Trend (Savitzky-Golay, scipy optional) ===
    if HAS_SCIPY:
        window_length = min(21, len(close) // 10)
        if window_length >= 3 and window_length % 2 == 1:
            try:
                smooth_trend = savgol_filter(close.values, window_length, 3)
                df["smooth_trend"] = pd.Series(smooth_trend, index=close.index)

                # Price relative to smooth trend
                df["price_vs_smooth_trend"] = close / df["smooth_trend"] - 1

                # Smooth trend slope
                df["smooth_trend_slope"] = df["smooth_trend"].diff(5) / (
                    df["smooth_trend"].shift(5) + 1e-10
                )
            except Exception:
                pass
    else:
        # Fallback: Use rolling average
        df["smooth_trend"] = close.rolling(10).mean()
        df["price_vs_smooth_trend"] = close / df["smooth_trend"] - 1
        df["smooth_trend_slope"] = df["smooth_trend"].diff(5) / (
            df["smooth_trend"].shift(5) + 1e-10
        )

    # === Residual Decomposition ===
    # Remove polynomial trend
    x = np.arange(len(close))
    z = np.polyfit(x, close.values, 2)
    p = np.poly1d(z)
    polynomial_trend = p(x)
    df["poly_trend"] = pd.Series(polynomial_trend, index=close.index)
    df["poly_residual"] = close - df["poly_trend"]

    # Residual Z-score (is price above/below polynomial trend?)
    residual_std = df["poly_residual"].rolling(63).std()
    df["residual_zscore"] = df["poly_residual"] / (residual_std + 1e-10)

    # === Component Analysis ===
    # Are we in trend or cycle-dominant regime?
    if "hp_trend" in df.columns and "hp_cycle" in df.columns:
        trend_var = df["hp_trend"].rolling(63).var()
        cycle_var = df["hp_cycle"].rolling(63).var()

        df["trend_dominant"] = (trend_var > cycle_var).astype(int)
        df["cycle_dominant"] = (cycle_var > trend_var).astype(int)

        # Ratio of variance
        df["trend_cycle_ratio"] = trend_var / (cycle_var + 1e-10)

    # === Acceleration Analysis ===
    if "smooth_trend" in df.columns:
        # Second derivative of price (curvature)
        df["price_curvature"] = df["smooth_trend"].diff().diff()

        # Positive curvature = acceleration
        df["price_accelerating"] = (df["price_curvature"] > 0).astype(int)
        df["price_decelerating"] = (df["price_curvature"] < 0).astype(int)

    # === Seasonality Effects ===
    if isinstance(df.index, pd.DatetimeIndex):
        # Day of week effect
        df["day_of_week"] = df.index.dayofweek

        # Month effect
        df["month"] = df.index.month

        # Rolling day-of-week return
        for dow in range(5):
            dow_mask = df["day_of_week"] == dow
            dow_ret_20 = daily_ret.where(dow_mask).rolling(20).mean()
            df[f"dow_{dow}_ret_20"] = dow_ret_20

        # Rolling month return
        for month in range(1, 13):
            month_mask = df["month"] == month
            month_ret = daily_ret.where(month_mask).rolling(60).mean()
            df[f"month_{month}_ret_60"] = month_ret

    # === Cycle-Trend Interaction ===
    if all(col in df.columns for col in ["cycle_position", "trend_dominant"]):
        # Up phase + trend dominant = strong buy
        df["cycle_trend_bullish"] = (
            (df["cycle_position"] > 0.7) & (df["trend_dominant"] > 0.5)
        ).astype(int)

        # Down phase + trend dominant = strong sell
        df["cycle_trend_bearish"] = (
            (df["cycle_position"] < 0.3) & (df["trend_dominant"] > 0.5)
        ).astype(int)

    # === Momentum from Components ===
    # Is momentum coming from trend or cycle?
    if all(col in df.columns for col in ["hp_trend", "hp_cycle"]):
        trend_mom = df["hp_trend"].pct_change(20)
        cycle_mom = df["hp_cycle"].diff(20) / (df["hp_cycle"].shift(20) + 1e-10)

        df["momentum_from_trend"] = trend_mom
        df["momentum_from_cycle"] = cycle_mom

        # Dominant momentum source
        df["trend_mom_dominant"] = (np.abs(trend_mom) > np.abs(cycle_mom)).astype(int)
        df["cycle_mom_dominant"] = (np.abs(cycle_mom) > np.abs(trend_mom)).astype(int)

    return df


# List of features added by this module
DECOMPOSITION_FEATURES = [
    # HP filter
    "hp_trend",
    "hp_cycle",
    "cycle_amplitude",
    "cycle_position",
    "cycle_phase",
    # Dominant cycle
    "dominant_cycle_period",
    "cycle_strength",
    # Detrended
    "detrended_ret_5d",
    "detrended_ret_20d",
    # Smoothed trend
    "smooth_trend",
    "price_vs_smooth_trend",
    "smooth_trend_slope",
    # Polynomial
    "poly_trend",
    "poly_residual",
    "residual_zscore",
    # Component analysis
    "trend_dominant",
    "cycle_dominant",
    "trend_cycle_ratio",
    # Curvature
    "price_curvature",
    "price_accelerating",
    "price_decelerating",
    # Cycle-Trend interaction
    "cycle_trend_bullish",
    "cycle_trend_bearish",
    # Momentum from components
    "momentum_from_trend",
    "momentum_from_cycle",
    "trend_mom_dominant",
    "cycle_mom_dominant",
    # Seasonality (dynamic, depends on index)
    "day_of_week",
    "month",
    # Note: dow_X_ret_20 and month_X_ret_60 are added dynamically
]
