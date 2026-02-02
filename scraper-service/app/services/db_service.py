#!/usr/bin/env python3
"""
Database Service for TradingView Data Upload (Container Version)

기존 data-scraping/db_service.py 코드를 그대로 가져와서
컨테이너 환경에 맞게 최소한의 수정만 적용한 버전.

변경사항:
- SSH 터널 제거 (host.docker.internal 사용)
- volume NULL 값 처리 추가
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for uploading TradingView CSV data to remote MySQL database"""

    def __init__(
        self,
        db_url: Optional[str] = None,
        db_user: str = "ahnbi2",
        db_password: str = "bigdata",
        db_host: str = "host.docker.internal",
        db_port: int = 3306,
        db_name: str = "etf2_db",
    ):
        """
        Initialize database service.

        Args:
            db_url: Full database URL (overrides other params if provided)
            db_user: MySQL username
            db_password: MySQL password
            db_host: MySQL host (default: host.docker.internal for container)
            db_port: MySQL port
            db_name: MySQL database name
        """
        # Use environment variable or construct URL
        self.db_url = db_url or os.getenv(
            "DB_URL",
            f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        self.db_name = db_name
        self.engine = None
        self.Session = None

    def connect(self):
        """Establish connection to database"""
        try:
            logger.info(f"Connecting to database...")

            self.engine = create_engine(
                self.db_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            self.Session = sessionmaker(bind=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("Database connection successful")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.close()
            raise

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed")

    @contextmanager
    def get_session(self):
        """Get database session context manager"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_table_name(self, symbol: str, timeframe: str) -> str:
        """
        Get table name for a symbol and timeframe.

        Args:
            symbol: Stock symbol (e.g., AAPL, NVDA)
            timeframe: Time period (e.g., D, 1h, 10m)

        Returns:
            Table name in format {symbol}_{timeframe}
        """
        # Map common period names to DB conventions (기존 코드와 동일)
        timeframe_map = {
            "12개월": "D",
            "12달": "D",
            "1개월": "D",
            "1달": "D",
            "1주": "D",
            "1일": "1h",
            "1시간": "10m",
            "10분": "1m",
        }

        tf = timeframe_map.get(timeframe, timeframe)
        return f"{symbol}_{tf}"

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = :db_name AND TABLE_NAME = :table_name
            """),
                {"db_name": self.db_name, "table_name": table_name},
            )
            count = result.scalar()
            return count > 0

    def create_table_if_not_exists(self, table_name: str):
        """Create table for stock data if it doesn't exist"""
        if self.table_exists(table_name):
            logger.debug(f"Table {table_name} already exists")
            return

        create_sql = f"""
        CREATE TABLE `{table_name}` (
            `time` DATETIME NOT NULL,
            `symbol` VARCHAR(32) NOT NULL,
            `timeframe` VARCHAR(16) NOT NULL,
            `open` DOUBLE,
            `high` DOUBLE,
            `low` DOUBLE,
            `close` DOUBLE,
            `volume` BIGINT,
            `rsi` DOUBLE,
            `macd` DOUBLE,
            PRIMARY KEY (`time`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """

        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()
        logger.info(f"Created table {table_name}")

    def parse_tradingview_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Parse TradingView exported CSV file. (기존 코드와 동일)

        Expected columns: time, open, high, low, close, volume
        Optional: Any indicator columns

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with parsed data
        """
        df = pd.read_csv(csv_path)

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Parse time column
        if "time" in df.columns:
            if pd.api.types.is_numeric_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], unit="s")
            else:
                df["time"] = pd.to_datetime(df["time"])
        elif "date" in df.columns:
            if pd.api.types.is_numeric_dtype(df["date"]):
                df["time"] = pd.to_datetime(df["date"], unit="s")
            else:
                df["time"] = pd.to_datetime(df["date"])
            df.drop("date", axis=1, inplace=True)

        # Ensure required columns exist
        required_cols = ["time", "open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add volume if missing
        if "volume" not in df.columns:
            df["volume"] = 0

        # Add RSI and MACD as null if not present
        if "rsi" not in df.columns:
            df["rsi"] = np.nan
        if "macd" not in df.columns:
            df["macd"] = np.nan

        # Select only the columns we need
        df = df[["time", "open", "high", "low", "close", "volume", "rsi", "macd"]]

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume", "rsi", "macd"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with null time
        df = df.dropna(subset=["time"])

        # Sort by time
        df = df.sort_values("time").reset_index(drop=True)

        return df

    def upload_csv(
        self,
        csv_path: Path,
        symbol: str,
        timeframe: str,
        replace_existing: bool = True,
    ) -> int:
        """
        Upload CSV data to database. (기존 코드 기반 + volume NULL 처리)

        Args:
            csv_path: Path to CSV file
            symbol: Stock symbol
            timeframe: Time period
            replace_existing: If True, replace existing data

        Returns:
            Number of rows inserted
        """
        table_name = self.get_table_name(symbol, timeframe)

        # Parse CSV
        df = self.parse_tradingview_csv(csv_path)
        if df.empty:
            logger.warning(f"No data in CSV: {csv_path}")
            return 0

        logger.info(f"Parsed {len(df)} rows from {csv_path.name}")

        # Create table if needed
        self.create_table_if_not_exists(table_name)

        # Upload data
        with self.engine.connect() as conn:
            if replace_existing:
                min_time = df["time"].min()
                max_time = df["time"].max()

                delete_sql = text(f"""
                    DELETE FROM `{table_name}`
                    WHERE `time` >= :min_time AND `time` <= :max_time
                """)
                result = conn.execute(
                    delete_sql, {"min_time": min_time, "max_time": max_time}
                )
                deleted = result.rowcount
                if deleted > 0:
                    logger.info(f"Deleted {deleted} existing rows in time range")

            # Convert DataFrame to records
            records = df.replace({np.nan: None}).to_dict("records")

            for record in records:
                record["symbol"] = symbol
                record["timeframe"] = timeframe
                # volume 컬럼이 NULL을 허용하지 않는 기존 테이블 호환
                if record["volume"] is None:
                    record["volume"] = 0

            insert_sql = text(f"""
                INSERT INTO `{table_name}` (`time`, `symbol`, `timeframe`, `open`, `high`, `low`, `close`, `volume`, `rsi`, `macd`)
                VALUES (:time, :symbol, :timeframe, :open, :high, :low, :close, :volume, :rsi, :macd)
                ON DUPLICATE KEY UPDATE
                    `open` = VALUES(`open`),
                    `high` = VALUES(`high`),
                    `low` = VALUES(`low`),
                    `close` = VALUES(`close`),
                    `volume` = VALUES(`volume`),
                    `rsi` = VALUES(`rsi`),
                    `macd` = VALUES(`macd`)
            """)

            for record in records:
                conn.execute(insert_sql, record)
            conn.commit()

        logger.info(f"Uploaded {len(records)} rows to {table_name}")
        return len(records)
