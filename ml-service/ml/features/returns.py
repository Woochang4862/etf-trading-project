"""
Return-based features for ETF Stock Prediction
"""
import pandas as pd
import numpy as np

from ..config import config


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return-based features to DataFrame

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added return features
    """
    df = df.copy()

    if 'close' not in df.columns:
        raise ValueError("Missing required column: close")

    close = df['close']
    high = df['high'] if 'high' in df.columns else close
    low = df['low'] if 'low' in df.columns else close

    # === Simple Returns (Percentage) ===
    for period in config.features.return_periods:
        df[f'ret_{period}d'] = close.pct_change(period)

    # === Log Returns (More normally distributed) ===
    for period in [5, 20, 63]:
        df[f'log_ret_{period}d'] = np.log(close / close.shift(period))

    # === Return Volatility (Rolling Std of Returns) ===
    daily_return = close.pct_change()

    for period in [10, 20, 63]:
        df[f'ret_std_{period}'] = daily_return.rolling(window=period).std()

    # === Return Statistics ===
    # Skewness (asymmetry of returns)
    df['ret_skew_20'] = daily_return.rolling(window=20).skew()
    df['ret_skew_63'] = daily_return.rolling(window=63).skew()

    # Kurtosis (tail heaviness)
    df['ret_kurt_20'] = daily_return.rolling(window=20).kurt()

    # === Max Drawdown ===
    # Rolling maximum
    rolling_max = close.rolling(window=63, min_periods=1).max()
    df['drawdown_63'] = (close - rolling_max) / rolling_max

    # Rolling min drawdown (worst)
    df['max_drawdown_63'] = df['drawdown_63'].rolling(window=63).min()

    # === From High/Low Ratios ===
    for period in [20, 63, 252]:
        # From rolling high
        rolling_high = high.rolling(window=period).max()
        df[f'from_high_{period}'] = close / rolling_high - 1

        # From rolling low
        rolling_low = low.rolling(window=period).min()
        df[f'from_low_{period}'] = close / rolling_low - 1

    # === Sharpe Ratio (Rolling) ===
    # Annualized Sharpe (assuming 0 risk-free rate for simplicity)
    df['sharpe_20'] = (
        daily_return.rolling(window=20).mean() /
        (daily_return.rolling(window=20).std() + 1e-10)
    ) * np.sqrt(252)

    df['sharpe_63'] = (
        daily_return.rolling(window=63).mean() /
        (daily_return.rolling(window=63).std() + 1e-10)
    ) * np.sqrt(252)

    # === Sortino Ratio (Downside volatility only) ===
    downside_returns = daily_return.where(daily_return < 0, 0)

    df['sortino_20'] = (
        daily_return.rolling(window=20).mean() /
        (downside_returns.rolling(window=20).std() + 1e-10)
    ) * np.sqrt(252)

    # === Calmar Ratio (Return / Max Drawdown) ===
    annual_return = daily_return.rolling(window=252).sum()
    df['calmar_252'] = annual_return / (df['max_drawdown_63'].abs() + 1e-10)

    # === Positive Return Ratio ===
    # Percentage of positive days in period
    df['pos_ret_ratio_20'] = (daily_return > 0).rolling(window=20).mean()
    df['pos_ret_ratio_63'] = (daily_return > 0).rolling(window=63).mean()

    # === Average Win/Loss Ratio ===
    def avg_win_loss_ratio(returns, window):
        """Calculate average win to average loss ratio"""
        wins = returns.where(returns > 0, np.nan)
        losses = returns.where(returns < 0, np.nan).abs()

        avg_win = wins.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()

        return avg_win / (avg_loss + 1e-10)

    df['win_loss_ratio_20'] = avg_win_loss_ratio(daily_return, 20)

    # === Momentum Acceleration ===
    # Is momentum accelerating or decelerating?
    if 'ret_5d' in df.columns and 'ret_20d' in df.columns:
        df['mom_acceleration'] = df['ret_5d'] - df['ret_5d'].shift(5)

    # === Return Reversals ===
    # Short-term reversal signal (mean reversion)
    df['reversal_5d'] = -df['ret_5d'].shift(1)  # Negative of past return

    # === Cumulative Returns ===
    df['cum_ret_ytd'] = (1 + daily_return).cumprod() - 1

    # Reset cumulative at year start (approximate)
    # Note: This is a simplified version
    year = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex)
                          else df['date'] if 'date' in df.columns
                          else df.index).year
    year_change = year != np.roll(year, 1)
    year_change[0] = True

    cum_ret = []
    current_cum = 1.0
    for i, (ret, is_new_year) in enumerate(zip(daily_return.fillna(0), year_change)):
        if is_new_year:
            current_cum = 1.0
        current_cum *= (1 + ret)
        cum_ret.append(current_cum - 1)
    df['cum_ret_ytd'] = cum_ret

    # === Target Variable ===
    # 3-month forward return (63 trading days)
    target_horizon = config.data.target_horizon
    df['target_3m'] = close.shift(-target_horizon) / close - 1

    return df


# List of features added by this module
RETURN_FEATURES = [
    # Simple returns
    'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d',
    'ret_20d', 'ret_40d', 'ret_63d', 'ret_126d', 'ret_252d',
    # Log returns
    'log_ret_5d', 'log_ret_20d', 'log_ret_63d',
    # Return volatility
    'ret_std_10', 'ret_std_20', 'ret_std_63',
    # Return statistics
    'ret_skew_20', 'ret_skew_63', 'ret_kurt_20',
    # Drawdown
    'drawdown_63', 'max_drawdown_63',
    # From high/low
    'from_high_20', 'from_high_63', 'from_high_252',
    'from_low_20', 'from_low_63', 'from_low_252',
    # Risk-adjusted returns
    'sharpe_20', 'sharpe_63', 'sortino_20', 'calmar_252',
    # Win ratio
    'pos_ret_ratio_20', 'pos_ret_ratio_63', 'win_loss_ratio_20',
    # Momentum
    'mom_acceleration', 'reversal_5d',
    # Cumulative
    'cum_ret_ytd',
    # Target
    'target_3m'
]
