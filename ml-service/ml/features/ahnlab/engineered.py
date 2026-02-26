"""
Engineered feature generation functions.
Extracted from AhnLab_LGBM_rank_0.19231/train.py lines 220-282.
"""

import pandas as pd


def add_engineered_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the panel data.

    Features include:
    - Additional return periods (10d, 30d)
    - Volatility measures (20d, 63d)
    - Price-to-moving-average ratios
    - Volume trend and surge indicators
    - Momentum and trend acceleration
    - EMA crosses and slopes

    Args:
        panel: DataFrame with columns ['ticker', 'date', 'close', 'high', 'volume',
               'ret_1d', 'ret_5d', 'ret_20d', 'ret_63d', 'sma_50', 'ema_10',
               'ema_20', 'ema_50', 'ema_200', 'volume_sma_20']

    Returns:
        DataFrame with additional engineered feature columns.
    """
    panel = panel.copy()

    grouped = panel.groupby("ticker")

    # Additional return periods
    panel["ret_10d"] = grouped["close"].pct_change(10)
    panel["ret_30d"] = grouped["close"].pct_change(30)

    # Rolling volatility
    panel["vol_20d"] = grouped["ret_1d"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    panel["vol_63d"] = grouped["ret_1d"].transform(
        lambda x: x.rolling(63, min_periods=10).std()
    )

    # Price to moving average ratios
    panel["price_to_sma_50"] = panel["close"] / (panel["sma_50"] + 1e-8)
    panel["price_to_ema_200"] = panel["close"] / (panel["ema_200"] + 1e-8)

    # Volume trend
    panel["volume_trend"] = panel["volume"] / (panel["volume_sma_20"] + 1e-8)

    # 52-week high ratio
    panel["high_252d"] = grouped["high"].transform(
        lambda x: x.rolling(252, min_periods=50).max()
    )
    panel["close_to_high_52w"] = panel["close"] / (panel["high_252d"] + 1e-8)

    # Return ratio
    panel["ret_5d_20d_ratio"] = panel["ret_5d"] / (panel["ret_20d"].abs() + 1e-8)

    # Momentum strength (weighted combination)
    panel["momentum_strength"] = (
        panel["ret_5d"] * 0.2 +
        panel["ret_20d"] * 0.3 +
        panel["ret_63d"] * 0.5
    )

    # Volume surge relative to 63-day average
    panel["volume_surge"] = (
        panel.groupby("ticker")["volume"]
        .transform(lambda x: x / (x.rolling(63, min_periods=20).mean() + 1e-8))
    )

    # Risk-adjusted returns
    panel["ret_vol_ratio_20d"] = panel["ret_20d"] / (panel["vol_20d"] + 1e-8)
    panel["ret_vol_ratio_63d"] = panel["ret_63d"] / (panel["vol_63d"] + 1e-8)

    # Trend acceleration (change in 5-day return)
    panel["trend_acceleration"] = (
        grouped["ret_5d"].transform(lambda x: x.diff(5))
    )

    # Distance from rolling highs (20d, 63d, 126d)
    panel["high_20d"] = grouped["high"].transform(
        lambda x: x.rolling(20, min_periods=10).max()
    )
    panel["close_to_high_20d"] = panel["close"] / (panel["high_20d"] + 1e-8)

    panel["high_63d"] = grouped["high"].transform(
        lambda x: x.rolling(63, min_periods=20).max()
    )
    panel["close_to_high_63d"] = panel["close"] / (panel["high_63d"] + 1e-8)

    panel["high_126d"] = grouped["high"].transform(
        lambda x: x.rolling(126, min_periods=30).max()
    )
    panel["close_to_high_126d"] = panel["close"] / (panel["high_126d"] + 1e-8)

    # Additional EMAs
    panel["ema_5"] = grouped["close"].transform(
        lambda x: x.ewm(span=5, adjust=False, min_periods=3).mean()
    )
    panel["ema_100"] = grouped["close"].transform(
        lambda x: x.ewm(span=100, adjust=False, min_periods=30).mean()
    )

    # Price to EMA ratios
    panel["price_to_ema_10"] = panel["close"] / (panel["ema_10"] + 1e-8)
    panel["price_to_ema_50"] = panel["close"] / (panel["ema_50"] + 1e-8)

    # EMA crossover signals
    panel["ema_cross_short"] = (
        (panel["ema_10"] - panel["ema_20"]) / (panel["ema_20"] + 1e-8)
    )
    panel["ema_cross_long"] = (
        (panel["ema_50"] - panel["ema_200"]) / (panel["ema_200"] + 1e-8)
    )

    # EMA slope (5-day change rate)
    panel["ema_slope_20"] = grouped["ema_20"].pct_change(5)

    # Drop temporary columns
    panel.drop(
        columns=["high_252d", "high_20d", "high_63d", "high_126d"],
        inplace=True,
        errors="ignore"
    )

    return panel
