"""Abstract base class for data providers."""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional


class BaseDataProvider(ABC):
    @abstractmethod
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a single ticker."""
        ...

    @abstractmethod
    def fetch_batch(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data for multiple tickers."""
        ...

    def supports_dividends(self) -> bool:
        return False
