"""
Autocorrelation features for ETF Stock Prediction

Measures return persistence and mean reversion potential.
"""

import pandas as pd
import numpy as np
from typing import List


def calculate_autocorrelation(series: pd.Series, max_lag: int = 20) -> pd.DataFrame:
    """
    Calculate rolling autocorrelation for multiple lags

    Args:
        series: Time series (e.g., daily returns)
        max_lag: Maximum lag to calculate

    Returns:
        DataFrame with autocorrelation for each lag
    """
    autocorrs = {}

    for lag in range(1, max_lag + 1):
        autocorr = series.rolling(window=126).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan, raw=False
        )
        autocorrs[f"autocorr_lag{lag}"] = autocorr

    return pd.DataFrame(autocorrs)


def add_autocorr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add autocorrelation features to DataFrame

    Args:
        df: DataFrame with price columns

    Returns:
        DataFrame with added autocorrelation features
    """
    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    close = df["close"]
    daily_ret = close.pct_change()

    # === Lagged Returns ===
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_lag{lag}"] = daily_ret.shift(lag)

    # === Lagged Return Autocorrelation ===
    # Correlation between current return and past returns
    for lag in [1, 5, 10, 20]:
        if f"ret_{lag}d" in df.columns:
            df[f"ret_autocorr_{lag}"] = daily_ret.rolling(window=60).corr(
                df[f"ret_{lag}d"].shift(lag)
            )

    # === Momentum Persistence ===
    # If momentum (positive return) tends to continue
    if "ret_5d" in df.columns and "ret_20d" in df.columns:
        # Correlation of 5d return with previous 5d returns
        mom_pers_5d = df["ret_5d"].rolling(window=126).corr(df["ret_5d"].shift(5))
        df["mom_persistence_5d"] = mom_pers_5d

        # Correlation of 20d return with previous 20d returns
        mom_pers_20d = df["ret_20d"].rolling(window=126).corr(df["ret_20d"].shift(20))
        df["mom_persistence_20d"] = mom_pers_20d

        # Overall momentum persistence score
        df["mom_persistence_score"] = (mom_pers_5d + mom_pers_20d) / 2

    # === Mean Reversion Potential ===
    # Negative autocorrelation = mean reversion
    if f"autocorr_lag1" in df.columns or "ret_autocorr_5" in df.columns:
        # Use autocorr_lag5 if available
        if "ret_autocorr_5" in df.columns:
            autocorr_short = df["ret_autocorr_5"]
        else:
            autocorr_short = daily_ret.rolling(window=60).corr(daily_ret.shift(5))

        # Negative autocorr = mean reversion opportunity
        df["mean_rev_potential"] = -autocorr_short

        # Strong mean reversion signal
        df["strong_mean_rev"] = (autocorr_short < -0.2).astype(int)

    # === Serial Correlation ===
    # Rolling correlation of returns with lagged returns
    for window in [20, 60, 126]:
        serial_corr = daily_ret.rolling(window=window).corr(daily_ret.shift(1))
        df[f"serial_corr_{window}"] = serial_corr

        # Positive serial correlation = trending
        df[f"trending_{window}"] = (serial_corr > 0.1).astype(int)

        # Negative serial correlation = mean reverting
        df[f"reverting_{window}"] = (serial_corr < -0.1).astype(int)

    # === Return Continuation ===
    # Does a positive (negative) return tend to be followed by positive (negative)?
    if "ret_1d" in df.columns:
        ret_sign = np.sign(df["ret_1d"].shift(1))
        current_sign = np.sign(df["ret_1d"])

        # Same sign as previous = continuation
        df["return_continuation"] = (ret_sign == current_sign).astype(int)

        # Rolling continuation rate
        df["continuation_rate_20"] = df["return_continuation"].rolling(20).mean()
        df["continuation_rate_60"] = df["return_continuation"].rolling(60).mean()

    # === Momentum Reversal Signal ===
    # If momentum was strong and reverses, could be turning point
    if all(col in df.columns for col in ["ret_5d", "ret_20d"]):
        # Recent strong positive momentum followed by negative
        strong_mom_up = df["ret_20d"].shift(1) > 0.1
        mom_reversed_down = df["ret_5d"] < -0.02
        df["mom_reversal_down"] = (strong_mom_up & mom_reversed_down).astype(int)

        # Recent strong negative momentum followed by positive
        strong_mom_down = df["ret_20d"].shift(1) < -0.1
        mom_reversed_up = df["ret_5d"] > 0.02
        df["mom_reversal_up"] = (strong_mom_down & mom_reversed_up).astype(int)

        # Any reversal
        df["mom_reversal_any"] = (
            df["mom_reversal_down"] | df["mom_reversal_up"]
        ).astype(int)

    # === Autocorrelation Decay ===
    # How fast does autocorrelation decay with lag?
    if all(
        col in df.columns
        for col in ["ret_autocorr_1", "ret_autocorr_5", "ret_autocorr_10"]
    ):
        # Rate of decay (1-5 vs 5-10)
        decay_short = df["ret_autocorr_1"] - df["ret_autocorr_5"]
        decay_long = df["ret_autocorr_5"] - df["ret_autocorr_10"]

        df["autocorr_decay_short"] = decay_short
        df["autocorr_decay_long"] = decay_long

        # Fast decay = noisy market
        df["noisy_market"] = (decay_short > 0.2).astype(int)

        # Slow decay = trending market
        df["trending_market"] = (decay_short < 0.05).astype(int)

    # === Volatility Autocorrelation ===
    # Does high volatility cluster?
    if "volatility_20" in df.columns:
        vol_autocorr = (
            df["volatility_20"].rolling(window=126).corr(df["volatility_20"].shift(10))
        )
        df["vol_autocorr"] = vol_autocorr

        # Volatility clustering
        df["vol_clustering"] = (vol_autocorr > 0.3).astype(int)

        # Volatility mean reversion
        df["vol_mean_rev"] = (vol_autocorr < 0).astype(int)

    # === Cross-Lag Correlation ===
    # Correlation between different lag periods
    if all(col in df.columns for col in ["ret_5d", "ret_20d"]):
        cross_corr = df["ret_5d"].rolling(window=126).corr(df["ret_20d"].shift(15))
        df["cross_lag_corr_5_20"] = cross_corr

        # High cross-lag correlation = consistent momentum
        df["consistent_momentum"] = (cross_corr > 0.3).astype(int)

    # === Return Predictability Score ===
    # How predictable are returns based on past returns?
    if all(col in df.columns for col in ["serial_corr_20", "serial_corr_60"]):
        # Higher average serial correlation = more predictable
        df["return_predictability"] = (
            np.abs(df["serial_corr_20"]) + np.abs(df["serial_corr_60"])
        ) / 2

        # Very high predictability regime
        df["high_predictability"] = (df["return_predictability"] > 0.15).astype(int)

    # === Regime-Specific Autocorrelation ===
    # Does autocorrelation change based on market state?
    if all(col in df.columns for col in ["serial_corr_20", "market_state"]):
        # Filter autocorrelation by market state
        bull_autocorr = df[df["market_state"] > 0]["serial_corr_20"]
        bear_autocorr = df[df["market_state"] < 0]["serial_corr_20"]
        sideways_autocorr = df[df["market_state"] == 0]["serial_corr_20"]

        # Difference in autocorrelation by regime
        df["autocorr_bull_vs_bear"] = bull_autocorr - bear_autocorr

        # Market state affects predictability
        df["predictability_by_regime"] = np.abs(bull_autocorr) - np.abs(
            sideways_autocorr
        )

    return df


# List of features added by this module
AUTOCORR_FEATURES = [
    # Lagged returns
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag5",
    "ret_lag10",
    "ret_lag20",
    # Lagged autocorrelation
    "ret_autocorr_1",
    "ret_autocorr_5",
    "ret_autocorr_10",
    "ret_autocorr_20",
    # Momentum persistence
    "mom_persistence_5d",
    "mom_persistence_20d",
    "mom_persistence_score",
    # Mean reversion
    "mean_rev_potential",
    "strong_mean_rev",
    # Serial correlation
    "serial_corr_20",
    "serial_corr_60",
    "serial_corr_126",
    "trending_20",
    "trending_60",
    "trending_126",
    "reverting_20",
    "reverting_60",
    "reverting_126",
    # Return continuation
    "return_continuation",
    "continuation_rate_20",
    "continuation_rate_60",
    # Momentum reversal
    "mom_reversal_down",
    "mom_reversal_up",
    "mom_reversal_any",
    # Autocorrelation decay
    "autocorr_decay_short",
    "autocorr_decay_long",
    "noisy_market",
    "trending_market",
    # Volatility autocorrelation
    "vol_autocorr",
    "vol_clustering",
    "vol_mean_rev",
    # Cross-lag correlation
    "cross_lag_corr_5_20",
    "consistent_momentum",
    # Predictability
    "return_predictability",
    "high_predictability",
    # Regime-specific
    "autocorr_bull_vs_bear",
    "predictability_by_regime",
]
