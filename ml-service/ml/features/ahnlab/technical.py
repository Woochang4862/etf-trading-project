"""
Technical indicator calculations using pandas-ta.

This module provides functions to calculate technical indicators for stock price data.
Based on the AhnLab LGBM implementation.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List

# List of all technical indicator columns that will be generated
TECHNICAL_FEATURE_COLS: List[str] = [
    # Returns
    'ret_1d', 'ret_5d', 'ret_20d', 'ret_63d',
    # MACD
    'macd', 'macd_signal', 'macd_hist',
    # RSI
    'rsi_14', 'rsi_28',
    # Bollinger Bands
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    # ATR
    'atr_14',
    # OBV
    'obv',
    # EMA
    'ema_10', 'ema_20', 'ema_50', 'ema_200',
    # SMA
    'sma_10', 'sma_20', 'sma_50',
    # Stochastic
    'stoch_k', 'stoch_d',
    # ADX
    'adx',
    # CCI
    'cci',
    # Williams %R
    'willr',
    # MFI
    'mfi',
    # VWAP
    'vwap',
    # Volume
    'volume_sma_20', 'volume_ratio'
]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV dataframe.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with additional technical indicator columns

    Notes:
        - Input dataframe is modified in place
        - Missing values (NaN) are expected for early periods due to lookback windows
        - All calculations use pandas-ta library
    """
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Returns (simple percentage change)
        df['ret_1d'] = close.pct_change(1)
        df['ret_5d'] = close.pct_change(5)
        df['ret_20d'] = close.pct_change(20)
        df['ret_63d'] = close.pct_change(63)

        # MACD (Moving Average Convergence Divergence)
        # pandas_ta returns: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        # Note: Requires at least 35 data points (slow=26 + signal=9)
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan

        macd_result = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_result is not None and not macd_result.empty:
            cols = macd_result.columns.tolist()
            for c in cols:
                if c.startswith('MACD_') and not c.startswith('MACDs_') and not c.startswith('MACDh_'):
                    df['macd'] = macd_result[c]
                elif c.startswith('MACDs_'):
                    df['macd_signal'] = macd_result[c]
                elif c.startswith('MACDh_'):
                    df['macd_hist'] = macd_result[c]

        # RSI (Relative Strength Index)
        df['rsi_14'] = ta.rsi(close, length=14)
        df['rsi_28'] = ta.rsi(close, length=28)

        # Bollinger Bands
        bbands = ta.bbands(close, length=20, std=2)
        if bbands is not None and not bbands.empty:
            bb_cols = bbands.columns.tolist()
            upper_col = [c for c in bb_cols if 'BBU' in c][0] if any('BBU' in c for c in bb_cols) else None
            middle_col = [c for c in bb_cols if 'BBM' in c][0] if any('BBM' in c for c in bb_cols) else None
            lower_col = [c for c in bb_cols if 'BBL' in c][0] if any('BBL' in c for c in bb_cols) else None

            if upper_col and middle_col and lower_col:
                df['bb_upper'] = bbands[upper_col]
                df['bb_middle'] = bbands[middle_col]
                df['bb_lower'] = bbands[lower_col]
                # Width: normalized by middle band to prevent division by zero
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
                # Position: where price is within the bands (0 = lower, 1 = upper)
                df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

        # ATR (Average True Range)
        df['atr_14'] = ta.atr(high, low, close, length=14)

        # OBV (On Balance Volume)
        df['obv'] = ta.obv(close, volume)

        # EMA (Exponential Moving Average)
        df['ema_10'] = ta.ema(close, length=10)
        df['ema_20'] = ta.ema(close, length=20)
        df['ema_50'] = ta.ema(close, length=50)
        df['ema_200'] = ta.ema(close, length=200)

        # SMA (Simple Moving Average)
        df['sma_10'] = ta.sma(close, length=10)
        df['sma_20'] = ta.sma(close, length=20)
        df['sma_50'] = ta.sma(close, length=50)

        # Stochastic Oscillator
        stoch = ta.stoch(high, low, close, k=14, d=3)
        if stoch is not None and not stoch.empty:
            stoch_cols = stoch.columns.tolist()
            k_col = [c for c in stoch_cols if 'STOCHk' in c or 'K_' in c]
            d_col = [c for c in stoch_cols if 'STOCHd' in c or 'D_' in c]
            if k_col:
                df['stoch_k'] = stoch[k_col[0]]
            if d_col:
                df['stoch_d'] = stoch[d_col[0]]

        # ADX (Average Directional Index)
        adx_result = ta.adx(high, low, close, length=14)
        if adx_result is not None and not adx_result.empty:
            adx_cols = [c for c in adx_result.columns if 'ADX' in c and 'DMP' not in c and 'DMN' not in c]
            if adx_cols:
                df['adx'] = adx_result[adx_cols[0]]

        # CCI (Commodity Channel Index)
        df['cci'] = ta.cci(high, low, close, length=20)

        # Williams %R
        df['willr'] = ta.willr(high, low, close, length=14)

        # MFI (Money Flow Index)
        df['mfi'] = ta.mfi(high, low, close, volume, length=14)

        # VWAP (Volume Weighted Average Price)
        # VWAP requires an ordered DatetimeIndex
        if 'date' in df.columns:
            original_index = df.index.copy()
            df_temp = df.set_index(pd.to_datetime(df['date']))
            vwap_result = ta.vwap(df_temp['high'], df_temp['low'], df_temp['close'], df_temp['volume'])
            df['vwap'] = vwap_result.values if vwap_result is not None else None
        else:
            df['vwap'] = ta.vwap(high, low, close, volume)

        # Volume indicators
        df['volume_sma_20'] = ta.sma(volume, length=20)
        # Volume ratio: current volume relative to 20-day average
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-8)

    except Exception as e:
        print(f"Technical indicator calculation failed: {e}")
        raise

    return df
