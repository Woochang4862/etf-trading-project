"""
Processed Data Service - Read ML features from etf2_db_processed.

Reads the 85 AhnLab feature columns from per-symbol tables
(e.g., AAPL_D, NVDA_D) in the etf2_db_processed database.
"""
import logging
from typing import List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# 85 feature columns (from etf-model/src/models/ahnlab_constants.py)
BASE_FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "dividends", "stock_splits",
    "ret_1d", "ret_5d", "ret_20d", "ret_63d",
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "rsi_28",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
    "atr_14", "obv",
    "ema_10", "ema_20", "ema_50", "ema_200",
    "sma_10", "sma_20", "sma_50",
    "stoch_k", "stoch_d", "adx", "cci", "willr", "mfi", "vwap",
    "volume_sma_20", "volume_ratio",
    "vix", "fed_funds_rate", "unemployment_rate", "cpi",
    "treasury_10y", "treasury_2y", "yield_curve",
    "oil_price", "usd_eur", "high_yield_spread",
]

ENGINEERED_FEATURE_COLS = [
    "ret_10d", "ret_30d", "vol_20d", "vol_63d",
    "price_to_sma_50", "price_to_ema_200", "volume_trend",
    "close_to_high_52w", "ret_5d_20d_ratio", "momentum_strength",
    "volume_surge", "ret_vol_ratio_20d", "ret_vol_ratio_63d",
    "trend_acceleration", "close_to_high_20d", "close_to_high_63d",
    "close_to_high_126d",
    "ema_5", "ema_100", "price_to_ema_10", "price_to_ema_50",
    "ema_cross_short", "ema_cross_long", "ema_slope_20",
]

ZS_BASE_COLS = [
    "vol_63d", "volume_sma_20", "obv", "vwap",
    "ema_200", "price_to_ema_200", "close_to_high_52w",
]
ZS_FEATURE_COLS = [f"{col}_zs" for col in ZS_BASE_COLS]

RANK_BASE_COLS = [
    "ret_20d", "ret_63d", "vol_20d", "momentum_strength", "volume_surge",
]
RANK_FEATURE_COLS = [f"{col}_rank" for col in RANK_BASE_COLS]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS + ZS_FEATURE_COLS + RANK_FEATURE_COLS


class ProcessedDataService:
    """Read ML features from etf2_db_processed."""

    def __init__(self, db: Session):
        self.db = db

    def list_symbols(self, timeframe: str = "D") -> List[str]:
        """List available symbols by checking which tables exist."""
        result = self.db.execute(
            text(
                "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_SCHEMA = 'etf2_db_processed' "
                "AND TABLE_NAME LIKE :pattern"
            ),
            {"pattern": f"%_{timeframe}"},
        )
        symbols = []
        suffix = f"_{timeframe}"
        for (table_name,) in result:
            if table_name.endswith(suffix):
                symbols.append(table_name[: -len(suffix)])
        return sorted(symbols)

    def get_features(
        self,
        symbol: str,
        timeframe: str = "D",
        limit: int = 252,
    ) -> pd.DataFrame:
        """Get feature data for a single symbol.

        Args:
            symbol: Stock ticker
            timeframe: Timeframe (D, 1h, etc.)
            limit: Number of recent rows to fetch

        Returns:
            DataFrame with feature columns
        """
        table_name = f"{symbol}_{timeframe}"
        cols = ", ".join(f"`{c}`" for c in ALL_FEATURE_COLS if c in ALL_FEATURE_COLS)

        query = text(
            f"SELECT `time`, {cols} FROM `{table_name}` "
            f"ORDER BY `time` DESC LIMIT :limit"
        )

        try:
            result = self.db.execute(query, {"limit": limit})
            rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=["time"] + ALL_FEATURE_COLS)
            df = df.sort_values("time").reset_index(drop=True)
            return df
        except Exception as e:
            logger.warning(f"Failed to read features for {symbol}: {e}")
            return pd.DataFrame()

    def get_all_latest_features(self, timeframe: str = "D") -> pd.DataFrame:
        """Get the latest-date features for ALL symbols (for ranking prediction).

        Returns a DataFrame with one row per symbol, containing the most recent
        feature values. Includes a 'symbol' column.
        """
        symbols = self.list_symbols(timeframe)
        if not symbols:
            logger.warning("No symbols found in processed DB")
            return pd.DataFrame()

        all_rows = []
        cols_str = ", ".join(f"`{c}`" for c in ALL_FEATURE_COLS)

        for symbol in symbols:
            table_name = f"{symbol}_{timeframe}"
            query = text(
                f"SELECT `time`, {cols_str} FROM `{table_name}` "
                f"ORDER BY `time` DESC LIMIT 1"
            )
            try:
                result = self.db.execute(query, {})
                row = result.fetchone()
                if row:
                    row_dict = dict(zip(["time"] + ALL_FEATURE_COLS, row))
                    row_dict["symbol"] = symbol
                    all_rows.append(row_dict)
            except Exception as e:
                logger.warning(f"Failed to read latest for {symbol}: {e}")
                continue

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        logger.info(f"Loaded latest features for {len(df)} symbols")
        return df
