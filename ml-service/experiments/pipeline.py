"""
Main pipeline for ETF Stock Prediction Competition
Memory-optimized version: processes year by year
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import warnings
import time
import gc

from .config import config, DATA_DIR, SUBMISSIONS_DIR
from .data.loader import DataLoader
from .data.preprocessor import Preprocessor, optimize_memory
from .features.technical import add_technical_features, TECHNICAL_FEATURES
from .features.momentum import add_momentum_features, MOMENTUM_FEATURES
from .features.volatility import add_volatility_features, VOLATILITY_FEATURES
from .features.volume import add_volume_features, VOLUME_FEATURES
from .features.returns import add_return_features, RETURN_FEATURES
from .features.cross_sectional import add_cross_sectional_features, CROSS_SECTIONAL_FEATURES
from .features.enhanced import add_enhanced_features, add_enhanced_cross_sectional, ENHANCED_FEATURES, ENHANCED_CROSS_SECTIONAL_FEATURES
from .models.lightgbm_model import ETFRankingModel
from .utils.evaluation import validate_submission


class CompetitionPipeline:
    """
    Memory-efficient pipeline for competition

    Processes each prediction year separately to reduce memory usage.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        output_dir: Path = SUBMISSIONS_DIR
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        self.loader = DataLoader(data_dir)
        self.preprocessor = Preprocessor(method='robust')
        self.feature_cols: List[str] = []
        self.results: Dict[int, pd.DataFrame] = {}

    def _get_feature_cols(self) -> List[str]:
        """Get list of feature columns (excluding target and identifiers)"""
        all_features = (
            TECHNICAL_FEATURES +
            MOMENTUM_FEATURES +
            VOLATILITY_FEATURES +
            VOLUME_FEATURES +
            RETURN_FEATURES +
            CROSS_SECTIONAL_FEATURES +
            ENHANCED_FEATURES +
            ENHANCED_CROSS_SECTIONAL_FEATURES
        )
        # Remove target and duplicates
        features = list(dict.fromkeys([f for f in all_features if f != 'target_3m']))
        return features

    def _create_features_for_ticker(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Create features for a single ticker"""
        try:
            if len(df) < config.data.min_history_days:
                return None

            tmp = df.copy()
            tmp = add_technical_features(tmp)
            tmp = add_momentum_features(tmp)
            tmp = add_volatility_features(tmp)
            tmp = add_volume_features(tmp)
            tmp = add_return_features(tmp)
            tmp = add_enhanced_features(tmp)  # 새로운 피처 추가

            tmp['date'] = tmp.index
            tmp['ticker'] = ticker
            tmp = tmp.replace([np.inf, -np.inf], np.nan)

            return tmp

        except Exception as e:
            return None

    def _get_required_dates(self, pred_year: int) -> List[str]:
        """Get required prediction dates from sample submission"""
        sample_path = self.data_dir / f"{pred_year}_sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            dates = sample['date'].unique().tolist()
            return sorted(dates)
        return []

    def process_year(
        self,
        pred_year: int,
        train_years: int = 10,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Process a single prediction year

        Args:
            pred_year: Year to predict (2020-2024)
            train_years: Years of training data
            verbose: Print progress

        Returns:
            Submission DataFrame
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {pred_year}")
            print(f"{'='*60}")

        # Get required dates from sample submission
        required_dates = self._get_required_dates(pred_year)
        if verbose and required_dates:
            print(f"Required dates: {len(required_dates)} days")

        # Define date ranges
        train_start = f"{pred_year - train_years}-01-01"
        train_end = f"{pred_year - 1}-12-31"
        pred_start = f"{pred_year}-01-01"
        pred_end = f"{pred_year}-12-31"

        # Load universe
        universe = self.loader.load_universe(pred_year)
        if verbose:
            print(f"Universe: {len(universe)} tickers")

        # Load and process stock data
        if verbose:
            print(f"Loading data from {train_start} to {pred_end}...")

        frames = []
        failed = 0
        iterator = tqdm(universe, desc="Loading & features") if verbose else universe

        for ticker in iterator:
            df = self.loader.load_stock_data(ticker, train_start, pred_end)
            if df is None:
                failed += 1
                continue

            featured = self._create_features_for_ticker(df, ticker)
            if featured is not None:
                frames.append(featured)

            # Clear memory periodically
            if len(frames) % 100 == 0:
                gc.collect()

        if verbose:
            print(f"Loaded {len(frames)}/{len(universe)} tickers (failed: {failed})")

        if not frames:
            raise ValueError(f"No valid data for {pred_year}")

        # Create panel
        panel = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

        panel['date'] = pd.to_datetime(panel['date'])

        # Optimize memory
        panel = optimize_memory(panel)
        if verbose:
            mem_mb = panel.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"Panel: {len(panel):,} rows, {mem_mb:.1f} MB")

        # Add cross-sectional features
        if verbose:
            print("Adding cross-sectional features...")
        panel = add_cross_sectional_features(panel)
        panel = add_enhanced_cross_sectional(panel)  # 향상된 횡단면 피처

        # Get feature columns
        if not self.feature_cols:
            self.feature_cols = self._get_feature_cols()
            self.feature_cols = [c for c in self.feature_cols if c in panel.columns]
            if verbose:
                print(f"Using {len(self.feature_cols)} features")

        # Split train and prediction data
        train_mask = (panel['date'] <= train_end) & panel['target_3m'].notna()
        pred_mask = (panel['date'] >= pred_start) & (panel['date'] <= pred_end)

        train_df = panel[train_mask].copy()
        pred_df = panel[pred_mask].copy()

        del panel
        gc.collect()

        if verbose:
            print(f"Training rows: {len(train_df):,}")
            print(f"Prediction rows: {len(pred_df):,}")

        # Prepare training data
        X_train = train_df[self.feature_cols].fillna(0).astype('float32')
        y_train = train_df['target_3m'].astype('float32')

        # Use last year as validation
        valid_start = f"{pred_year - 1}-01-01"
        valid_mask = train_df['date'] >= valid_start

        # LambdaRank를 위한 그룹 정보 계산 (날짜별 종목 수)
        train_dates = train_df[~valid_mask]['date']
        valid_dates = train_df[valid_mask]['date']
        train_groups = train_dates.value_counts().sort_index().tolist()
        valid_groups = valid_dates.value_counts().sort_index().tolist()

        X_valid = X_train[valid_mask]
        y_valid = y_train[valid_mask]
        X_train_final = X_train[~valid_mask]
        y_train_final = y_train[~valid_mask]

        del train_df
        gc.collect()

        # Train model with LambdaRank
        if verbose:
            print(f"\nTraining model with LambdaRank...")
            print(f"Training groups: {len(train_groups)} dates")
            print(f"Validation groups: {len(valid_groups)} dates")

        model = ETFRankingModel(use_lambdarank=True)
        model.fit(
            X_train_final, y_train_final,
            X_valid, y_valid,
            self.feature_cols,
            train_groups=train_groups,
            valid_groups=valid_groups
        )

        del X_train, X_train_final, y_train, y_train_final, X_valid, y_valid
        gc.collect()

        if verbose:
            print(f"\nTop 10 features:")
            print(model.get_feature_importance(10).to_string(index=False))

        # Generate predictions
        if verbose:
            print(f"\nGenerating predictions...")

        # Use required dates from sample submission if available
        if required_dates:
            target_dates = [pd.to_datetime(d) for d in required_dates]
        else:
            target_dates = sorted(pred_df['date'].unique())

        available_dates = set(pred_df['date'].unique())
        results = []
        last_predictions = None  # Cache for missing dates

        for date in tqdm(target_dates, desc="Predicting") if verbose else target_dates:
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')

            if date in available_dates:
                day_df = pred_df[pred_df['date'] == date]
                X_day = day_df[self.feature_cols].fillna(0).astype('float32')
                tickers = day_df['ticker'].tolist()
                top_tickers, _ = model.predict_top_k(X_day, tickers, k=100)
                last_predictions = top_tickers  # Cache for potential missing dates
            else:
                # Use last available predictions for missing date
                if last_predictions is not None:
                    top_tickers = last_predictions
                else:
                    # Fallback: use universe tickers
                    top_tickers = universe[:100]

            for rank, ticker in enumerate(top_tickers, 1):
                results.append({
                    'date': date_str,
                    'rank': rank,
                    'ticker': ticker
                })

        del pred_df, model
        gc.collect()

        submission = pd.DataFrame(results)
        if verbose:
            print(f"Generated {len(submission):,} predictions ({len(target_dates)} days)")

        return submission

    def run(
        self,
        pred_years: List[int] = None,
        train_years: int = 10,
        verbose: bool = True
    ) -> Dict[int, Path]:
        """
        Run pipeline for all prediction years

        Args:
            pred_years: Years to predict
            train_years: Years of training data
            verbose: Print progress

        Returns:
            Dictionary mapping year to submission file path
        """
        if pred_years is None:
            pred_years = config.data.pred_years

        start_time = time.time()

        print("\n" + "="*60)
        print("ETF STOCK PREDICTION COMPETITION PIPELINE")
        print("Memory-optimized: processing year by year")
        print("="*60)

        paths = {}

        for year in pred_years:
            try:
                submission = self.process_year(year, train_years, verbose)
                self.results[year] = submission

                # Save immediately
                filepath = self.output_dir / f"{year}.submission.csv"
                submission.to_csv(filepath, index=False)
                paths[year] = filepath

                if verbose:
                    print(f"Saved: {filepath}")

                # Clear memory
                gc.collect()

            except Exception as e:
                print(f"Error processing {year}: {e}")
                continue

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Pipeline completed in {elapsed/60:.1f} minutes")
        print(f"{'='*60}")

        # Validate
        self._validate_all()

        return paths

    def _validate_all(self):
        """Validate all submissions"""
        print("\nValidation:")
        expected = {2020: 25300, 2021: 25200, 2022: 25100, 2023: 25000, 2024: 25200}

        for year, sub in self.results.items():
            exp = expected.get(year, 25000)
            actual = len(sub)
            status = "OK" if abs(actual - exp) <= 100 else "CHECK"
            print(f"  {year}: {actual:,} rows (expected ~{exp:,}) [{status}]")


def main():
    """Main entry point"""
    pipeline = CompetitionPipeline()
    paths = pipeline.run()

    print("\nSubmission files:")
    for year, path in paths.items():
        print(f"  {year}: {path}")


if __name__ == "__main__":
    main()
