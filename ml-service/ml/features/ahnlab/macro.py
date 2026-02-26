"""
Macro economic data collection from FRED API.

This module handles downloading and processing macro-economic indicators
from the Federal Reserve Economic Data (FRED) API.
"""
import os
import pandas as pd
from fredapi import Fred
from typing import Optional, Dict, List
import warnings

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Look for .env in etf-model directory
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars

warnings.filterwarnings("ignore")

# List of macro feature columns expected in the final dataset
MACRO_FEATURE_COLS: List[str] = [
    "vix",
    "fed_funds_rate",
    "unemployment_rate",
    "cpi",
    "treasury_10y",
    "treasury_2y",
    "yield_curve",
    "oil_price",
    "usd_eur",
    "high_yield_spread",
]

# Mapping of FRED indicator codes to feature names
FRED_INDICATORS: Dict[str, str] = {
    'VIXCLS': 'vix',                        # VIX Index
    'DFF': 'fed_funds_rate',                # Federal Funds Rate
    'UNRATE': 'unemployment_rate',          # Unemployment Rate
    'CPIAUCSL': 'cpi',                      # Consumer Price Index
    'DGS10': 'treasury_10y',                # 10-Year Treasury Yield
    'DGS2': 'treasury_2y',                  # 2-Year Treasury Yield
    'T10Y2Y': 'yield_curve',                # Yield Curve (10Y-2Y)
    'DCOILWTICO': 'oil_price',              # WTI Oil Price
    'DEXUSEU': 'usd_eur',                   # USD/EUR Exchange Rate
    'BAMLH0A0HYM2': 'high_yield_spread',    # High Yield Spread
}


class MacroDataCollector:
    """
    Collector for macro-economic data from FRED API.

    Usage:
        collector = MacroDataCollector(api_key="your_api_key")
        macro_df = collector.fetch_macro_data("2010-01-01", "2024-12-31")
        panel = collector.merge_with_panel(stock_panel, macro_df)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize MacroDataCollector.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key not provided. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.fred = Fred(api_key=self.api_key)

    def fetch_macro_data(
        self,
        start_date: str,
        end_date: str,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Fetch all macro indicators from FRED.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            verbose: Whether to print download progress

        Returns:
            DataFrame with date index and macro indicator columns
        """
        if verbose:
            print("\nDownloading macro-economic data from FRED...")

        fred_data = {}

        for fred_code, col_name in FRED_INDICATORS.items():
            try:
                series = self.fred.get_series(
                    fred_code,
                    observation_start=start_date,
                    observation_end=end_date
                )
                fred_data[col_name] = series
                if verbose:
                    print(f"  ✓ {col_name} ({fred_code}): {len(series)} observations")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {col_name} ({fred_code}): {e}")
                # Continue with other indicators even if one fails

        if not fred_data:
            raise RuntimeError("Failed to download any macro indicators from FRED")

        # Combine into DataFrame
        macro_df = pd.DataFrame(fred_data)
        macro_df.index.name = 'date'

        # Forward-fill missing values (common for indicators with different frequencies)
        macro_df = macro_df.ffill()

        if verbose:
            print(f"\nMacro data shape: {macro_df.shape}")
            print(f"Date range: {macro_df.index.min()} to {macro_df.index.max()}")
            print(f"Missing values after ffill:\n{macro_df.isnull().sum()}")

        return macro_df

    def merge_with_panel(
        self,
        panel: pd.DataFrame,
        macro_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Merge macro data with stock panel, forward-filling missing values.

        Args:
            panel: Stock panel DataFrame with a date column
            macro_df: Macro DataFrame with date index
            date_col: Name of date column in panel

        Returns:
            Merged panel with macro features
        """
        # Reset index to make date a column
        macro_data = macro_df.reset_index()
        macro_data['date'] = pd.to_datetime(macro_data['date'])

        # Ensure panel date column is datetime
        panel = panel.copy()
        panel[date_col] = pd.to_datetime(panel[date_col])

        # Remove timezone info if present (to avoid merge issues)
        if panel[date_col].dt.tz is not None:
            panel[date_col] = panel[date_col].dt.tz_localize(None)
        if macro_data['date'].dt.tz is not None:
            macro_data['date'] = macro_data['date'].dt.tz_localize(None)

        # Left join: keep all panel rows, add macro data where available
        merged = pd.merge(panel, macro_data, on=date_col, how='left')

        # Sort by ticker and date for proper forward-fill within groups
        if 'ticker' in merged.columns:
            merged = merged.sort_values(['ticker', date_col]).reset_index(drop=True)
        else:
            merged = merged.sort_values(date_col).reset_index(drop=True)

        # Forward-fill macro columns (grouped by ticker if available)
        macro_cols = [col for col in macro_data.columns if col != 'date']
        if 'ticker' in merged.columns:
            for col in macro_cols:
                # Forward-fill within each ticker group, then backward-fill any remaining
                merged[col] = merged.groupby('ticker')[col].ffill().bfill()
        else:
            # Global forward-fill and backward-fill
            for col in macro_cols:
                merged[col] = merged[col].ffill().bfill()

        return merged


def fetch_macro_data(
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch macro data.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: FRED API key (optional, reads from env if None)
        verbose: Whether to print progress

    Returns:
        DataFrame with macro indicators, or None if API key not available
    """
    try:
        collector = MacroDataCollector(api_key=api_key)
        return collector.fetch_macro_data(start_date, end_date, verbose=verbose)
    except ValueError as e:
        if verbose:
            print(f"Warning: {e}")
            print("Continuing without macro data.")
        return None
    except Exception as e:
        if verbose:
            print(f"Error fetching macro data: {e}")
            print("Continuing without macro data.")
        return None


# Example usage
if __name__ == "__main__":
    # Fetch macro data
    collector = MacroDataCollector()
    macro_df = collector.fetch_macro_data("2010-01-01", "2024-12-31")

    print("\nMacro Data Sample:")
    print(macro_df.head())
    print(f"\nColumns: {macro_df.columns.tolist()}")
    print(f"\nData types:\n{macro_df.dtypes}")
