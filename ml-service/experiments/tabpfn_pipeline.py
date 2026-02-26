"""
TabPFN Pipeline for ETF Stock Prediction Competition

Key strategy: Date-by-date model training
- For each prediction date, train a new TabPFN model on recent data
- Leverages TabPFN's few-shot learning capability
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
from .models.tabpfn_model import TabPFNRankingModel, DateByDateTrainer
from .utils.evaluation import validate_submission


class TabPFNPipeline:
    """
    TabPFN-based pipeline for ETF stock prediction

    Uses date-by-date training strategy:
    - Each prediction date trains its own model on recent cross-sectional data
    - Ideal for TabPFN's few-shot learning characteristics
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        output_dir: Path = SUBMISSIONS_DIR,
        lookback_days: int = None,
        max_features: int = None,
        device: str = None
    ):
        """
        Initialize pipeline

        Args:
            data_dir: Data directory
            output_dir: Output directory for submissions
            lookback_days: Days of data for training each model
            max_features: Maximum features to use
            device: Computing device
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Use config defaults if not specified
        self.lookback_days = lookback_days or config.tabpfn.lookback_days
        self.max_features = max_features or config.tabpfn.max_features
        self.device = device or config.tabpfn.device

        self.loader = DataLoader(data_dir)
        self.preprocessor = Preprocessor(method='robust')
        self.feature_cols: List[str] = []
        self.selected_features: List[str] = []
        self.results: Dict[int, pd.DataFrame] = {}

    def _get_all_feature_cols(self) -> List[str]:
        """Get list of all feature columns (excluding target and identifiers)"""
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

    def _select_features(
        self,
        panel: pd.DataFrame,
        max_features: int = 100
    ) -> List[str]:
        """
        Select top features for TabPFN

        TabPFN works better with fewer, high-quality features.
        Uses correlation with target to select features.

        Args:
            panel: Panel data with features and target
            max_features: Maximum number of features

        Returns:
            List of selected feature names
        """
        all_features = [c for c in self.feature_cols if c in panel.columns]

        # Filter to rows with valid target
        valid_panel = panel[panel['target_3m'].notna()].copy()

        if len(valid_panel) == 0:
            return all_features[:max_features]

        # Calculate correlation with target
        correlations = {}
        for col in all_features:
            if col in valid_panel.columns:
                try:
                    # Sample for speed if panel is large
                    if len(valid_panel) > 50000:
                        sample = valid_panel.sample(n=50000, random_state=42)
                    else:
                        sample = valid_panel

                    corr = sample[col].corr(sample['target_3m'])
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
                except Exception:
                    continue

        # Sort by absolute correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        # Select top features
        selected = [f for f, _ in sorted_features[:max_features]]

        # Ensure we have at least some features
        if len(selected) < 10:
            selected = all_features[:max_features]

        return selected

    def _create_features_for_ticker(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> Optional[pd.DataFrame]:
        """
        Create features for a single ticker
        (Identical to CompetitionPipeline method)
        """
        try:
            if len(df) < config.data.min_history_days:
                return None

            tmp = df.copy()
            tmp = add_technical_features(tmp)
            tmp = add_momentum_features(tmp)
            tmp = add_volatility_features(tmp)
            tmp = add_volume_features(tmp)
            tmp = add_return_features(tmp)
            tmp = add_enhanced_features(tmp)

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
        Process a single prediction year using TabPFN

        Steps:
        1. Load universe and stock data
        2. Generate features (reusing existing code)
        3. Select top features for TabPFN
        4. For each prediction date:
           a. Collect training data from past lookback_days
           b. Train TabPFN model
           c. Predict on current date's stocks
           d. Select Top-100
        5. Generate submission file

        Args:
            pred_year: Year to predict (2020-2024)
            train_years: Years of data to load for feature calculation
            verbose: Print progress

        Returns:
            Submission DataFrame
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TabPFN Processing {pred_year}")
            print(f"  Device: {self.device}")
            print(f"  Lookback days: {self.lookback_days}")
            print(f"  Max features: {self.max_features}")
            print(f"{'='*60}")

        # Get required dates from sample submission
        required_dates = self._get_required_dates(pred_year)
        if verbose and required_dates:
            print(f"Required dates: {len(required_dates)} days")

        # Define date ranges - need more history for feature calculation
        train_start = f"{pred_year - train_years}-01-01"
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
        panel = add_enhanced_cross_sectional(panel)

        # Get all feature columns
        if not self.feature_cols:
            self.feature_cols = self._get_all_feature_cols()
            self.feature_cols = [c for c in self.feature_cols if c in panel.columns]
            if verbose:
                print(f"Total features available: {len(self.feature_cols)}")

        # Select top features for TabPFN
        if verbose:
            print(f"Selecting top {self.max_features} features by correlation...")
        self.selected_features = self._select_features(panel, self.max_features)
        if verbose:
            print(f"Selected {len(self.selected_features)} features")
            print(f"Top 5: {self.selected_features[:5]}")

        # Initialize date-by-date trainer
        trainer = DateByDateTrainer(
            feature_cols=self.selected_features,
            target_col='target_3m',
            lookback_days=self.lookback_days,
            min_train_samples=config.tabpfn.min_train_samples,
            max_train_samples=config.tabpfn.max_train_samples,
            model_params={
                'n_estimators': config.tabpfn.n_estimators,
                'device': self.device,
                'ignore_pretraining_limits': config.tabpfn.ignore_pretraining_limits,
                'memory_saving_mode': config.tabpfn.memory_saving_mode,
                'random_state': config.seed,
            }
        )

        # Filter to prediction period
        pred_start = f"{pred_year}-01-01"
        if required_dates:
            target_dates = [pd.to_datetime(d) for d in required_dates]
        else:
            pred_panel = panel[(panel['date'] >= pred_start) & (panel['date'] <= pred_end)]
            target_dates = sorted(pred_panel['date'].unique())

        # Generate predictions for each date
        if verbose:
            print(f"\nGenerating predictions for {len(target_dates)} dates...")

        results = []
        failed_dates = 0

        for date in tqdm(target_dates, desc="TabPFN Predicting") if verbose else target_dates:
            date_ts = pd.Timestamp(date)
            date_str = date_ts.strftime('%Y-%m-%d')

            try:
                top_tickers, predictions = trainer.predict_date(
                    panel, date_ts, verbose=False
                )

                for rank, ticker in enumerate(top_tickers, 1):
                    results.append({
                        'date': date_str,
                        'rank': rank,
                        'ticker': ticker
                    })

            except Exception as e:
                failed_dates += 1
                if verbose and failed_dates <= 5:
                    print(f"  Warning: Failed {date_str}: {e}")

                # Use fallback
                try:
                    top_tickers, _ = trainer._fallback_prediction(panel, date_ts)
                    for rank, ticker in enumerate(top_tickers[:100], 1):
                        results.append({
                            'date': date_str,
                            'rank': rank,
                            'ticker': ticker
                        })
                except Exception:
                    # Last resort: use universe
                    for rank, ticker in enumerate(universe[:100], 1):
                        results.append({
                            'date': date_str,
                            'rank': rank,
                            'ticker': ticker
                        })

        del panel
        gc.collect()

        if verbose and failed_dates > 0:
            print(f"Failed dates: {failed_dates}/{len(target_dates)}")

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
        Run TabPFN pipeline for all prediction years

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
        print("TABPFN ETF STOCK PREDICTION PIPELINE")
        print("Date-by-date training strategy")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Lookback days: {self.lookback_days}")
        print(f"Max features: {self.max_features}")

        paths = {}

        for year in pred_years:
            try:
                submission = self.process_year(year, train_years, verbose)
                self.results[year] = submission

                # Save with tabpfn suffix
                filepath = self.output_dir / f"{year}.tabpfn.submission.csv"
                submission.to_csv(filepath, index=False)
                paths[year] = filepath

                if verbose:
                    print(f"Saved: {filepath}")

                # Clear memory
                gc.collect()

            except Exception as e:
                print(f"Error processing {year}: {e}")
                import traceback
                traceback.print_exc()
                continue

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"TabPFN Pipeline completed in {elapsed/60:.1f} minutes")
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
    """Main entry point for TabPFN pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Run TabPFN experiment')
    parser.add_argument('--year', type=int, nargs='+', default=None,
                       help='Prediction years (default: all)')
    parser.add_argument('--lookback', type=int, default=None,
                       help='Lookback days for training')
    parser.add_argument('--features', type=int, default=None,
                       help='Max features to use')
    parser.add_argument('--device', type=str, default=None,
                       choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')

    args = parser.parse_args()

    pred_years = args.year if args.year else None

    pipeline = TabPFNPipeline(
        lookback_days=args.lookback,
        max_features=args.features,
        device=args.device
    )

    paths = pipeline.run(
        pred_years=pred_years,
        verbose=not args.quiet
    )

    print("\nTabPFN Submission files:")
    for year, path in paths.items():
        print(f"  {year}: {path}")


if __name__ == "__main__":
    main()
