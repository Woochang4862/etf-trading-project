"""
Feature Engineering Pipeline - Orchestrates all feature modules to create 85-feature panel.

This pipeline integrates:
- Technical indicators (MACD, RSI, Bollinger Bands, etc.)
- Engineered features (momentum, volatility, price ratios)
- Cross-sectional features (z-scores, percentile ranks)
- Macro-economic indicators (VIX, interest rates, etc.)

Output: Panel with 85 features ready for LightGBM training.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Literal
import warnings

from .ahnlab.constants import ALL_FEATURE_COLS, BASE_FEATURE_COLS, ENGINEERED_FEATURE_COLS
from .ahnlab.technical import add_technical_indicators
from .ahnlab.engineered import add_engineered_features
from .ahnlab.cross_sectional import add_cross_sectional_zscores, add_cross_sectional_ranks
from .ahnlab.macro import MacroDataCollector
from .data_providers.yfinance_provider import YFinanceProvider
from .data_providers.mysql_provider import MySQLProvider

warnings.filterwarnings("ignore")


class FeaturePipeline:
    """
    Main orchestrator for feature engineering pipeline.

    Creates 85-feature panel from raw OHLCV data:
    - 49 base features (OHLCV + technical indicators + macro)
    - 24 engineered features
    - 12 cross-sectional features (7 z-scores + 5 ranks)

    Usage:
        # Using YFinance (default)
        pipeline = FeaturePipeline(data_provider="yfinance")

        # Using MySQL (TradingView scraped data)
        pipeline = FeaturePipeline(data_provider="mysql", mysql_url="mysql+pymysql://...")

        panel = pipeline.create_panel(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date="2020-01-01",
            end_date="2024-12-31",
            shift_features=True
        )
    """

    def __init__(
        self,
        data_provider: Literal["yfinance", "mysql"] = "yfinance",
        fred_api_key: Optional[str] = None,
        include_macro: bool = True,
        include_target: bool = True,
        target_horizon: int = 63,
        mysql_url: Optional[str] = None,
    ):
        """
        Initialize FeaturePipeline.

        Args:
            data_provider: Data source ("yfinance" or "mysql")
            fred_api_key: FRED API key for macro data (reads from env if None)
            include_macro: Whether to include macro-economic features
            include_target: Whether to include target variable (target_3m)
            target_horizon: Number of days for forward return (default 63 = ~3 months)
            mysql_url: MySQL connection URL (for mysql provider, reads from env if None)
        """
        # Initialize data provider
        if data_provider == "mysql":
            self.data_provider = MySQLProvider(db_url=mysql_url)
            self.provider_name = "mysql"
        else:
            self.data_provider = YFinanceProvider(include_dividends=True)
            self.provider_name = "yfinance"

        self.macro_collector = MacroDataCollector(api_key=fred_api_key) if include_macro else None
        self.include_macro = include_macro
        self.include_target = include_target
        self.target_horizon = target_horizon

    def create_panel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        shift_features: bool = True,
        convert_to_float32: bool = True,
        validate_features: bool = True,
    ) -> pd.DataFrame:
        """
        Create full feature panel with all 85 features + target variable.

        Processing steps:
        1. Fetch OHLCV data from provider
        2. Add technical indicators (per ticker)
        3. Add macro data (if enabled)
        4. Add engineered features
        5. Add cross-sectional features (per date)
        6. Add target variable (target_3m = forward return, if enabled)
        7. Shift features by 1 day (if enabled, for prediction leakage prevention)
        8. Clean and validate

        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            shift_features: Apply 1-day lag to prevent data leakage
            convert_to_float32: Convert numeric columns to float32 to save memory
            validate_features: Check that all expected features are present

        Returns:
            DataFrame with columns: ['ticker', 'date'] + ALL_FEATURE_COLS

        Raises:
            ValueError: If validation fails or no data fetched
        """
        # Step 1: Fetch raw data
        print(f"\n{'='*60}")
        print(f"Feature Pipeline: Creating 85-feature panel")
        print(f"{'='*60}")
        print(f"Data provider: {self.provider_name}")
        print(f"Tickers: {len(tickers)}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Include macro: {self.include_macro}")
        print(f"Include target: {self.include_target} (horizon={self.target_horizon} days)")
        print(f"Shift features: {shift_features}")

        print(f"\n[1/7] Fetching OHLCV data for {len(tickers)} tickers...")
        panel = self.data_provider.fetch_batch(tickers, start_date, end_date)

        if panel.empty:
            raise ValueError("No data fetched from provider. Check tickers and date range.")

        print(f"  ✓ Fetched {panel.shape[0]:,} rows for {panel['ticker'].nunique()} tickers")

        # Step 2: Add technical indicators per ticker
        print(f"\n[2/7] Adding technical indicators...")
        try:
            panel = panel.groupby('ticker', group_keys=False).apply(
                lambda df: add_technical_indicators(df.sort_values('date'))
            ).reset_index(drop=True)
            print(f"  ✓ Technical indicators added")
        except Exception as e:
            print(f"  ✗ Error adding technical indicators: {e}")
            raise

        # Step 3: Add macro data
        if self.include_macro and self.macro_collector:
            print(f"\n[3/7] Adding macro-economic data...")
            try:
                macro_df = self.macro_collector.fetch_macro_data(
                    start_date, end_date, verbose=False
                )
                panel = self.macro_collector.merge_with_panel(panel, macro_df)
                print(f"  ✓ Macro data merged ({len(macro_df.columns)} indicators)")
            except Exception as e:
                print(f"  ⚠ Warning: Macro data collection failed: {e}")
                print(f"  → Continuing without macro features")
                self.include_macro = False
        else:
            print(f"\n[3/7] Skipping macro data (disabled)")

        # Step 4: Add engineered features
        print(f"\n[4/7] Adding engineered features...")
        try:
            panel = add_engineered_features(panel)
            print(f"  ✓ Engineered features added")
        except Exception as e:
            print(f"  ✗ Error adding engineered features: {e}")
            raise

        # Step 5: Add cross-sectional features
        print(f"\n[5/7] Adding cross-sectional features...")
        try:
            panel = add_cross_sectional_zscores(panel)
            panel = add_cross_sectional_ranks(panel)
            print(f"  ✓ Cross-sectional features added (z-scores + ranks)")
        except Exception as e:
            print(f"  ✗ Error adding cross-sectional features: {e}")
            raise

        # Step 6: Add target variable (before feature shift - target should NOT be shifted)
        if self.include_target:
            print(f"\n[6/7] Adding target variable (target_3m = {self.target_horizon}-day forward return)...")
            try:
                panel = self._add_target(panel)
                print(f"  ✓ Target variable added")
            except Exception as e:
                print(f"  ✗ Error adding target variable: {e}")
                raise
        else:
            print(f"\n[6/7] Skipping target variable (disabled)")

        # Step 7: Shift features (1-day lag for prediction)
        if shift_features:
            print(f"\n[7/7] Shifting features by 1 day...")
            try:
                panel = self._shift_features(panel)
                print(f"  ✓ Features shifted to prevent data leakage")
            except Exception as e:
                print(f"  ✗ Error shifting features: {e}")
                raise
        else:
            print(f"\n[7/7] Skipping feature shift (disabled)")

        # Clean up
        print(f"\nCleaning and validating panel...")
        panel = self._clean_data(panel, convert_to_float32)

        # Validate features
        if validate_features:
            self._validate_features(panel)

        print(f"\n{'='*60}")
        print(f"✓ Panel created successfully")
        print(f"  Shape: {panel.shape[0]:,} rows × {panel.shape[1]} columns")
        print(f"  Tickers: {panel['ticker'].nunique()}")
        print(f"  Date range: {panel['date'].min()} to {panel['date'].max()}")
        # Only check columns that exist in the panel
        existing_features = [c for c in ALL_FEATURE_COLS if c in panel.columns]
        print(f"  Features: {len(existing_features)}/{len(ALL_FEATURE_COLS)}")
        if self.include_target and 'target_3m' in panel.columns:
            target_valid = panel['target_3m'].notna().sum()
            print(f"  Target (target_3m): {target_valid:,} valid rows")
        print(f"  Missing values: {panel[existing_features].isnull().sum().sum():,}")
        print(f"{'='*60}\n")

        return panel

    def _add_target(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Add target variable: forward return over target_horizon days.

        target_3m = (close[t+horizon] - close[t]) / close[t]

        Note: This creates NaN values for the last `target_horizon` days
        of each ticker (no future data available).
        """
        panel = panel.copy()

        # Calculate forward return per ticker
        # shift(-horizon) gives us the future close price
        panel['target_3m'] = panel.groupby('ticker')['close'].transform(
            lambda x: x.pct_change(self.target_horizon).shift(-self.target_horizon)
        )

        # Also add the target date for reference (optional, useful for debugging)
        panel['target_date'] = panel.groupby('ticker')['date'].shift(-self.target_horizon)

        return panel

    def _shift_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Shift all features by 1 day per ticker to prevent data leakage.

        This ensures that features at time t only use information available
        up to time t-1, preventing look-ahead bias.

        Note: target_3m is NOT shifted - it's a label, not a feature.
        """
        panel = panel.copy()
        feature_cols = [c for c in ALL_FEATURE_COLS if c in panel.columns]

        # Exclude target columns from shifting
        feature_cols = [c for c in feature_cols if c not in ['target_3m', 'target_date']]

        # Shift within each ticker group
        for col in feature_cols:
            panel[col] = panel.groupby('ticker')[col].shift(1)

        return panel

    def _clean_data(self, panel: pd.DataFrame, convert_to_float32: bool) -> pd.DataFrame:
        """
        Clean panel data: replace inf, optionally convert to float32.

        Note: NaN values are kept - they'll be handled during model training.
        """
        # Replace infinite values with NaN
        panel = panel.replace([np.inf, -np.inf], np.nan)

        # Convert to float32 to save memory
        if convert_to_float32:
            numeric_cols = panel.select_dtypes(include=[np.number]).columns
            # Exclude 'date' column from conversion
            numeric_cols = [c for c in numeric_cols if c != 'date']
            for col in numeric_cols:
                panel[col] = panel[col].astype(np.float32)

        return panel

    def _validate_features(self, panel: pd.DataFrame) -> None:
        """
        Validate that all expected features are present in the panel.

        Raises:
            ValueError: If required features are missing
        """
        missing_features = [col for col in ALL_FEATURE_COLS if col not in panel.columns]

        if missing_features:
            print(f"\n⚠ WARNING: Missing {len(missing_features)} expected features:")
            for feat in missing_features[:10]:  # Show first 10
                print(f"  - {feat}")
            if len(missing_features) > 10:
                print(f"  ... and {len(missing_features) - 10} more")

            # Don't raise error if only macro features are missing
            non_macro_missing = [f for f in missing_features
                                 if f not in ['vix', 'fed_funds_rate', 'unemployment_rate',
                                             'cpi', 'treasury_10y', 'treasury_2y',
                                             'yield_curve', 'oil_price', 'usd_eur',
                                             'high_yield_spread']]

            if non_macro_missing:
                raise ValueError(
                    f"Critical features missing: {non_macro_missing[:5]}. "
                    f"Check feature generation modules."
                )
            else:
                print(f"  → Only macro features missing (OK if macro disabled)")
        else:
            print(f"  ✓ All {len(ALL_FEATURE_COLS)} expected features present")


# Convenience function
def create_feature_panel(
    tickers: List[str],
    start_date: str,
    end_date: str,
    data_provider: Literal["yfinance", "mysql"] = "yfinance",
    include_macro: bool = True,
    include_target: bool = True,
    target_horizon: int = 63,
    fred_api_key: Optional[str] = None,
    mysql_url: Optional[str] = None,
    shift_features: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to create feature panel with default settings.

    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_provider: Data source ("yfinance" or "mysql")
        include_macro: Whether to include macro-economic features
        include_target: Whether to include target variable (target_3m)
        target_horizon: Number of days for forward return (default 63 = ~3 months)
        fred_api_key: FRED API key (reads from env if None)
        mysql_url: MySQL connection URL (for mysql provider)
        shift_features: Apply 1-day lag to prevent data leakage

    Returns:
        DataFrame with 85 features + target_3m (if include_target=True)
    """
    pipeline = FeaturePipeline(
        data_provider=data_provider,
        include_macro=include_macro,
        include_target=include_target,
        target_horizon=target_horizon,
        fred_api_key=fred_api_key,
        mysql_url=mysql_url,
    )

    return pipeline.create_panel(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        shift_features=shift_features,
    )


# Example usage
if __name__ == "__main__":
    # Test with a few tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]

    print("Testing Feature Pipeline...")
    panel = create_feature_panel(
        tickers=test_tickers,
        start_date="2023-01-01",
        end_date="2023-12-31",
        include_macro=False,  # Skip macro for quick test
        shift_features=True,
    )

    print("\nPanel Sample:")
    print(panel.head())
    print(f"\nColumns ({len(panel.columns)}):")
    print(panel.columns.tolist())
    print(f"\nData types:")
    print(panel.dtypes.value_counts())
    print(f"\nMissing values per feature:")
    print(panel.isnull().sum().sort_values(ascending=False).head(10))
