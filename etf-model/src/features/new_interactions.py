"""
New Interaction Features for ETF Stock Prediction

Creates interaction features by combining existing features:
- Return × Volatility (risk-adjusted metrics)
- Volume × Price change (volume-price dynamics)
- Momentum × Technical indicators
- Cross-feature ratios and products
"""

import numpy as np
import pandas as pd
from typing import List


NEW_INTERACTION_FEATURES: List[str] = [
    "ret_vol_ratio_5d",
    "ret_vol_ratio_20d",
    "ret_vol_ratio_63d",
    "sharpe_like_5d",
    "sharpe_like_20d",
    "sharpe_like_63d",
    "sortino_like_20d",
    "calmar_like_63d",
    "vol_adj_momentum_20d",
    "vol_adj_momentum_63d",
    "volume_price_trend",
    "volume_return_corr_20d",
    "volume_breakout_score",
    "price_volume_divergence",
    "obv_momentum",
    "volume_weighted_return",
    "momentum_rsi_interaction",
    "momentum_macd_interaction",
    "trend_strength_composite",
    "mean_reversion_score",
    "breakout_score",
    "continuation_score",
    "volatility_regime_interaction",
    "volume_volatility_ratio",
    "price_efficiency_ratio",
    "hurst_proxy",
    "return_skew_20d",
    "return_kurt_20d",
    "tail_ratio",
    "upside_potential_ratio",
    "gain_loss_ratio",
    "win_rate_20d",
    "profit_factor_20d",
    "recovery_factor",
    "ulcer_index",
    "pain_index",
]


def _safe_divide(
    numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0
) -> pd.Series:
    result = np.where(denominator != 0, numerator / denominator, fill_value)
    return pd.Series(result, index=numerator.index)


def add_risk_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Risk-adjusted return metrics"""
    close = df["close"]
    ret_1d = close.pct_change()

    for period in [5, 20, 63]:
        ret_period = close / close.shift(period) - 1
        vol_period = ret_1d.rolling(window=period, min_periods=1).std() * np.sqrt(252)

        df[f"ret_vol_ratio_{period}d"] = _safe_divide(ret_period, vol_period)

        mean_ret = ret_1d.rolling(window=period, min_periods=1).mean() * 252
        df[f"sharpe_like_{period}d"] = _safe_divide(mean_ret, vol_period)

    downside_ret = ret_1d.copy()
    downside_ret[downside_ret > 0] = 0
    downside_vol = downside_ret.rolling(window=20, min_periods=1).std() * np.sqrt(252)
    mean_ret_20 = ret_1d.rolling(window=20, min_periods=1).mean() * 252
    df["sortino_like_20d"] = _safe_divide(mean_ret_20, downside_vol)

    ret_63d = close / close.shift(63) - 1
    max_dd = (
        (close / close.rolling(window=63, min_periods=1).max() - 1)
        .rolling(window=63, min_periods=1)
        .min()
        .abs()
    )
    df["calmar_like_63d"] = _safe_divide(ret_63d, max_dd)

    return df


def add_volatility_adjusted_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum normalized by volatility"""
    close = df["close"]
    ret_1d = close.pct_change()

    for period in [20, 63]:
        momentum = close / close.shift(period) - 1
        vol = ret_1d.rolling(window=period, min_periods=1).std()
        df[f"vol_adj_momentum_{period}d"] = _safe_divide(momentum, vol)

    return df


def add_volume_price_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-price interaction features"""
    close = df["close"]
    volume = df["volume"]
    ret_1d = close.pct_change()

    vol_change = volume.pct_change()
    df["volume_price_trend"] = ret_1d * np.sign(vol_change)

    df["volume_return_corr_20d"] = ret_1d.rolling(window=20, min_periods=5).corr(
        vol_change
    )

    vol_sma = volume.rolling(window=20, min_periods=1).mean()
    vol_std = volume.rolling(window=20, min_periods=1).std()
    vol_zscore = _safe_divide(volume - vol_sma, vol_std)
    price_breakout = (
        close > close.rolling(window=20, min_periods=1).max().shift(1)
    ).astype(float)
    df["volume_breakout_score"] = vol_zscore * price_breakout

    price_trend = np.sign(close - close.shift(5))
    vol_trend = np.sign(
        volume.rolling(window=5, min_periods=1).mean()
        - volume.rolling(window=20, min_periods=1).mean()
    )
    df["price_volume_divergence"] = (price_trend != vol_trend).astype(
        float
    ) * price_trend

    obv = (np.sign(ret_1d) * volume).cumsum()
    df["obv_momentum"] = (obv / obv.shift(20) - 1).clip(-5, 5)

    vol_weight = volume / volume.rolling(window=20, min_periods=1).sum()
    df["volume_weighted_return"] = (
        (ret_1d * vol_weight).rolling(window=20, min_periods=1).sum()
    )

    return df


def add_momentum_indicator_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum × Technical indicator interactions"""
    close = df["close"]

    momentum_20d = close / close.shift(20) - 1

    if "rsi_14" in df.columns:
        rsi_normalized = (df["rsi_14"] - 50) / 50
        df["momentum_rsi_interaction"] = momentum_20d * rsi_normalized
    else:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = _safe_divide(gain, loss, fill_value=1)
        rsi = 100 - (100 / (1 + rs))
        rsi_normalized = (rsi - 50) / 50
        df["momentum_rsi_interaction"] = momentum_20d * rsi_normalized

    if "macd_histogram" in df.columns:
        macd_norm = df["macd_histogram"] / close
        df["momentum_macd_interaction"] = momentum_20d * np.sign(macd_norm)
    else:
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        df["momentum_macd_interaction"] = momentum_20d * np.sign(histogram / close)

    return df


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Composite scores combining multiple signals"""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    sma_20 = close.rolling(window=20, min_periods=1).mean()
    sma_50 = close.rolling(window=50, min_periods=1).mean()

    price_above_sma20 = (close > sma_20).astype(float)
    price_above_sma50 = (close > sma_50).astype(float)
    sma20_above_sma50 = (sma_20 > sma_50).astype(float)
    higher_highs = (high > high.shift(1)).astype(float)
    higher_lows = (low > low.shift(1)).astype(float)

    df["trend_strength_composite"] = (
        price_above_sma20
        + price_above_sma50
        + sma20_above_sma50
        + higher_highs
        + higher_lows
    ) / 5

    sma_20_dist = (close - sma_20) / sma_20
    bb_mid = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std()
    bb_position = _safe_divide(close - bb_mid, 2 * bb_std)
    df["mean_reversion_score"] = -sma_20_dist * (1 - bb_position.abs())

    high_20 = high.rolling(window=20, min_periods=1).max()
    low_20 = low.rolling(window=20, min_periods=1).min()
    breakout_up = (
        (close > high_20.shift(1)) & (close.shift(1) <= high_20.shift(2))
    ).astype(float)
    breakout_down = (
        (close < low_20.shift(1)) & (close.shift(1) >= low_20.shift(2))
    ).astype(float)
    df["breakout_score"] = breakout_up - breakout_down

    ret_5d = close / close.shift(5) - 1
    ret_20d = close / close.shift(20) - 1
    same_direction = (np.sign(ret_5d) == np.sign(ret_20d)).astype(float)
    df["continuation_score"] = same_direction * np.sign(ret_20d) * ret_5d.abs()

    return df


def add_regime_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility regime interaction features"""
    close = df["close"]
    volume = df["volume"]
    ret_1d = close.pct_change()

    vol_short = ret_1d.rolling(window=10, min_periods=1).std()
    vol_long = ret_1d.rolling(window=60, min_periods=1).std()
    vol_regime = _safe_divide(vol_short, vol_long)

    momentum = close / close.shift(20) - 1
    df["volatility_regime_interaction"] = momentum * vol_regime

    vol_ma = volume.rolling(window=20, min_periods=1).mean()
    price_vol = ret_1d.rolling(window=20, min_periods=1).std()
    df["volume_volatility_ratio"] = _safe_divide(vol_ma, price_vol * close)

    net_move = abs(close - close.shift(20))
    gross_move = ret_1d.abs().rolling(window=20, min_periods=1).sum() * close.shift(20)
    df["price_efficiency_ratio"] = _safe_divide(net_move, gross_move)

    ret_20d = ret_1d.rolling(window=20, min_periods=1)
    max_ret = ret_20d.max()
    min_ret = ret_20d.min()
    range_ret = max_ret - min_ret
    df["hurst_proxy"] = _safe_divide(
        range_ret, ret_1d.rolling(window=20, min_periods=1).std() * np.sqrt(20)
    )

    return df


def add_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return distribution features"""
    close = df["close"]
    ret_1d = close.pct_change()

    df["return_skew_20d"] = ret_1d.rolling(window=20, min_periods=5).skew()
    df["return_kurt_20d"] = ret_1d.rolling(window=20, min_periods=5).kurt()

    def tail_ratio(x):
        if len(x) < 10:
            return np.nan
        sorted_x = np.sort(x)
        n = len(sorted_x)
        upper_5pct = sorted_x[int(0.95 * n) :]
        lower_5pct = sorted_x[: int(0.05 * n) + 1]
        if len(lower_5pct) == 0 or np.mean(np.abs(lower_5pct)) == 0:
            return np.nan
        return np.mean(upper_5pct) / np.mean(np.abs(lower_5pct))

    df["tail_ratio"] = ret_1d.rolling(window=60, min_periods=20).apply(
        tail_ratio, raw=True
    )

    positive_ret = ret_1d.copy()
    positive_ret[positive_ret < 0] = 0
    expected_shortfall = ret_1d.rolling(window=20, min_periods=5).quantile(0.05).abs()
    df["upside_potential_ratio"] = _safe_divide(
        positive_ret.rolling(window=20, min_periods=5).mean(), expected_shortfall
    )

    return df


def add_trading_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Trading-style performance metrics"""
    close = df["close"]
    ret_1d = close.pct_change()

    gains = ret_1d.copy()
    losses = ret_1d.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0

    avg_gain = gains.rolling(window=20, min_periods=5).mean()
    avg_loss = losses.rolling(window=20, min_periods=5).mean().abs()
    df["gain_loss_ratio"] = _safe_divide(avg_gain, avg_loss, fill_value=1)

    df["win_rate_20d"] = (ret_1d > 0).rolling(window=20, min_periods=5).mean()

    gross_profit = gains.rolling(window=20, min_periods=5).sum()
    gross_loss = losses.rolling(window=20, min_periods=5).sum().abs()
    df["profit_factor_20d"] = _safe_divide(gross_profit, gross_loss, fill_value=1)

    cum_ret = (1 + ret_1d).cumprod()
    rolling_max = cum_ret.rolling(window=252, min_periods=20).max()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.rolling(window=63, min_periods=20).min().abs()
    ret_63d = cum_ret / cum_ret.shift(63) - 1
    df["recovery_factor"] = _safe_divide(ret_63d, max_dd)

    squared_dd = drawdown**2
    df["ulcer_index"] = np.sqrt(squared_dd.rolling(window=14, min_periods=5).mean())

    df["pain_index"] = drawdown.abs().rolling(window=20, min_periods=5).mean()

    return df


def add_new_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all new interaction features

    Args:
        df: DataFrame with OHLCV and existing feature columns

    Returns:
        DataFrame with additional interaction features
    """
    df = add_risk_adjusted_features(df)
    df = add_volatility_adjusted_momentum(df)
    df = add_volume_price_interactions(df)
    df = add_momentum_indicator_interactions(df)
    df = add_composite_scores(df)
    df = add_regime_interactions(df)
    df = add_distribution_features(df)
    df = add_trading_metrics(df)

    return df
