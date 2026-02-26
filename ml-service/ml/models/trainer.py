"""
Walk-forward training strategy for ETF Stock Prediction Competition
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime

from .lightgbm_model import ETFRankingModel
from ..config import config, SUBMISSIONS_DIR


class WalkForwardTrainer:
    """
    Walk-forward validation trainer

    Trains models using rolling window approach:
    - 2020 prediction: Train on 2010-2019
    - 2021 prediction: Train on 2011-2020 (or 2010-2020)
    - etc.
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = 'target_3m',
        train_years: int = None,
        expanding_window: bool = True
    ):
        """
        Initialize trainer

        Args:
            feature_cols: List of feature column names
            target_col: Target column name
            train_years: Number of years for training window
            expanding_window: If True, use expanding window (2010-2019, 2010-2020, ...)
                            If False, use rolling window (2010-2019, 2011-2020, ...)
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.train_years = train_years or config.data.train_years
        self.expanding_window = expanding_window

        self.models: Dict[int, ETFRankingModel] = {}
        self.results: Dict[int, pd.DataFrame] = {}

    def get_train_valid_dates(
        self,
        pred_year: int
    ) -> Tuple[str, str, str, str]:
        """
        Get training and validation date ranges

        Args:
            pred_year: Year to predict

        Returns:
            Tuple of (train_start, train_end, valid_start, valid_end)
        """
        if self.expanding_window:
            # Expanding window: always start from base year
            train_start = f"{config.data.train_start_year}-01-01"
        else:
            # Rolling window
            train_start = f"{pred_year - self.train_years}-01-01"

        # Training ends at validation start
        valid_start = f"{pred_year - 1}-01-01"
        train_end = f"{pred_year - 2}-12-31"  # 1 year before pred for validation
        valid_end = f"{pred_year - 1}-12-31"

        return train_start, train_end, valid_start, valid_end

    def prepare_data(
        self,
        panel: pd.DataFrame,
        pred_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training, validation, and prediction data

        Args:
            panel: Full panel data
            pred_year: Year to predict

        Returns:
            Tuple of (train_df, valid_df, pred_df)
        """
        train_start, train_end, valid_start, valid_end = self.get_train_valid_dates(pred_year)

        # Convert date column to datetime if needed
        panel = panel.copy()
        panel['date'] = pd.to_datetime(panel['date'])

        # Filter data
        train_mask = (
            (panel['date'] >= train_start) &
            (panel['date'] <= train_end) &
            panel[self.target_col].notna()
        )
        valid_mask = (
            (panel['date'] >= valid_start) &
            (panel['date'] <= valid_end) &
            panel[self.target_col].notna()
        )
        pred_mask = (
            (panel['date'] >= f"{pred_year}-01-01") &
            (panel['date'] <= f"{pred_year}-12-31")
        )

        train_df = panel[train_mask].copy()
        valid_df = panel[valid_mask].copy()
        pred_df = panel[pred_mask].copy()

        return train_df, valid_df, pred_df

    def train_year(
        self,
        panel: pd.DataFrame,
        pred_year: int,
        verbose: bool = True
    ) -> ETFRankingModel:
        """
        Train model for a specific prediction year

        Args:
            panel: Full panel data
            pred_year: Year to predict
            verbose: Print progress

        Returns:
            Trained model
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training for {pred_year} predictions")
            print(f"{'='*60}")

        # Prepare data
        train_df, valid_df, pred_df = self.prepare_data(panel, pred_year)

        if verbose:
            train_start, train_end, valid_start, valid_end = self.get_train_valid_dates(pred_year)
            print(f"Training period: {train_start} to {train_end} ({len(train_df):,} rows)")
            print(f"Validation period: {valid_start} to {valid_end} ({len(valid_df):,} rows)")
            print(f"Prediction period: {pred_year} ({len(pred_df):,} rows)")

        # Prepare features and targets
        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df[self.target_col]

        X_valid = valid_df[self.feature_cols].fillna(0)
        y_valid = valid_df[self.target_col]

        # Train model
        model = ETFRankingModel()
        model.fit(X_train, y_train, X_valid, y_valid, self.feature_cols)

        self.models[pred_year] = model

        if verbose:
            print(f"\nTop 10 features:")
            print(model.get_feature_importance(10).to_string(index=False))

        return model

    def generate_submission(
        self,
        panel: pd.DataFrame,
        pred_year: int,
        model: Optional[ETFRankingModel] = None,
        top_k: int = 100,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Generate submission file for a prediction year

        Args:
            panel: Full panel data
            pred_year: Year to predict
            model: Trained model (uses stored model if None)
            top_k: Number of top stocks to select per day
            verbose: Print progress

        Returns:
            Submission DataFrame with columns: date, rank, ticker
        """
        if model is None:
            if pred_year not in self.models:
                raise ValueError(f"No trained model for {pred_year}. Call train_year() first.")
            model = self.models[pred_year]

        # Get prediction data
        _, _, pred_df = self.prepare_data(panel, pred_year)

        # Get unique prediction dates
        pred_dates = pred_df['date'].unique()
        pred_dates = sorted(pred_dates)

        if verbose:
            print(f"\nGenerating predictions for {len(pred_dates)} days in {pred_year}")

        results = []
        iterator = tqdm(pred_dates, desc=f"Predicting {pred_year}") if verbose else pred_dates

        for date in iterator:
            # Get data for this date
            day_df = pred_df[pred_df['date'] == date].copy()

            if len(day_df) == 0:
                continue

            # Prepare features
            X_day = day_df[self.feature_cols].fillna(0)
            tickers = day_df['ticker'].tolist()

            # Get top-k predictions
            top_tickers, _ = model.predict_top_k(X_day, tickers, k=top_k)

            # Create submission rows
            for rank, ticker in enumerate(top_tickers, 1):
                results.append({
                    'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                    'rank': rank,
                    'ticker': ticker
                })

        submission = pd.DataFrame(results)
        self.results[pred_year] = submission

        if verbose:
            print(f"Generated {len(submission):,} predictions for {pred_year}")

        return submission

    def train_and_predict_all(
        self,
        panel: pd.DataFrame,
        pred_years: List[int] = None,
        verbose: bool = True
    ) -> Dict[int, pd.DataFrame]:
        """
        Train and predict for all years

        Args:
            panel: Full panel data
            pred_years: Years to predict (default from config)
            verbose: Print progress

        Returns:
            Dictionary mapping year to submission DataFrame
        """
        if pred_years is None:
            pred_years = config.data.pred_years

        for year in pred_years:
            # Train
            model = self.train_year(panel, year, verbose=verbose)

            # Predict
            submission = self.generate_submission(
                panel, year, model,
                top_k=config.model.top_k,
                verbose=verbose
            )

        return self.results

    def save_submissions(
        self,
        output_dir: str = None,
        prefix: str = ""
    ):
        """
        Save all submission files

        Args:
            output_dir: Output directory (default from config)
            prefix: Filename prefix
        """
        if output_dir is None:
            output_dir = SUBMISSIONS_DIR

        for year, submission in self.results.items():
            filename = f"{prefix}{year}.submission.csv" if prefix else f"{year}.submission.csv"
            filepath = output_dir / filename
            submission.to_csv(filepath, index=False)
            print(f"Saved: {filepath} ({len(submission):,} rows)")

    def validate_submissions(self) -> Dict[int, Dict]:
        """
        Validate submission files against expected format

        Returns:
            Dictionary of validation results per year
        """
        expected_rows = {
            2020: 25300,  # 253 days * 100
            2021: 25200,  # 252 days * 100
            2022: 25100,  # 251 days * 100
            2023: 25000,  # 250 days * 100
            2024: 25200,  # 252 days * 100
        }

        results = {}
        for year, submission in self.results.items():
            expected = expected_rows.get(year, 25000)
            actual = len(submission)

            # Check columns
            has_correct_cols = set(submission.columns) == {'date', 'rank', 'ticker'}

            # Check ranks
            ranks_per_date = submission.groupby('date')['rank'].apply(
                lambda x: list(sorted(x)) == list(range(1, 101))
            )
            all_ranks_correct = ranks_per_date.all()

            # Check for duplicates
            has_duplicates = submission.duplicated(subset=['date', 'ticker']).any()

            results[year] = {
                'expected_rows': expected,
                'actual_rows': actual,
                'rows_match': abs(actual - expected) <= 100,  # Allow small variance
                'has_correct_columns': has_correct_cols,
                'all_ranks_correct': all_ranks_correct,
                'has_no_duplicates': not has_duplicates,
                'is_valid': (
                    abs(actual - expected) <= 100 and
                    has_correct_cols and
                    all_ranks_correct and
                    not has_duplicates
                )
            }

            status = "VALID" if results[year]['is_valid'] else "INVALID"
            print(f"{year}: {status} ({actual:,} rows)")

        return results
