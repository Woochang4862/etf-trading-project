"""
Evaluation utilities for ETF Stock Prediction Competition
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def calculate_top100_accuracy(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    date_col: str = 'date',
    ticker_col: str = 'ticker'
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Top-100 prediction accuracy

    For each date, computes the overlap between predicted Top-100
    and actual Top-100 stocks.

    Args:
        pred_df: Predicted top stocks (date, rank, ticker)
        actual_df: Actual top stocks (date, rank, ticker) or with returns
        date_col: Date column name
        ticker_col: Ticker column name

    Returns:
        Tuple of (overall_accuracy, daily_accuracy_df)
    """
    # Ensure date format consistency
    pred_df = pred_df.copy()
    actual_df = actual_df.copy()
    pred_df[date_col] = pd.to_datetime(pred_df[date_col])
    actual_df[date_col] = pd.to_datetime(actual_df[date_col])

    # Get unique dates
    pred_dates = set(pred_df[date_col].unique())
    actual_dates = set(actual_df[date_col].unique())
    common_dates = pred_dates & actual_dates

    daily_results = []

    for date in sorted(common_dates):
        # Get predicted top-100
        pred_tickers = set(
            pred_df[pred_df[date_col] == date][ticker_col].values
        )

        # Get actual top-100
        actual_tickers = set(
            actual_df[actual_df[date_col] == date][ticker_col].values
        )

        # Calculate overlap
        overlap = len(pred_tickers & actual_tickers)
        accuracy = overlap / 100  # As percentage

        daily_results.append({
            'date': date,
            'accuracy': accuracy,
            'overlap': overlap,
            'pred_count': len(pred_tickers),
            'actual_count': len(actual_tickers)
        })

    daily_df = pd.DataFrame(daily_results)
    overall_accuracy = daily_df['accuracy'].mean() if len(daily_df) > 0 else 0

    return overall_accuracy, daily_df


def calculate_actual_top100(
    panel: pd.DataFrame,
    target_col: str = 'target_3m',
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    top_k: int = 100
) -> pd.DataFrame:
    """
    Calculate actual Top-100 stocks based on realized returns

    Args:
        panel: Panel data with realized returns
        target_col: Column with realized returns
        date_col: Date column
        ticker_col: Ticker column
        top_k: Number of top stocks

    Returns:
        DataFrame with actual top stocks (date, rank, ticker)
    """
    results = []

    for date in panel[date_col].unique():
        day_df = panel[panel[date_col] == date].copy()

        # Skip if no target data
        if day_df[target_col].isna().all():
            continue

        # Get top-k by actual return
        top_df = day_df.nlargest(top_k, target_col)

        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            results.append({
                date_col: date,
                'rank': rank,
                ticker_col: row[ticker_col],
                'actual_return': row[target_col]
            })

    return pd.DataFrame(results)


def validate_submission(
    submission: pd.DataFrame,
    year: int,
    universe: Optional[List[str]] = None
) -> Dict:
    """
    Validate submission file format

    Args:
        submission: Submission DataFrame
        year: Prediction year
        universe: Valid ticker universe

    Returns:
        Dictionary with validation results
    """
    expected_days = {
        2020: 253,
        2021: 252,
        2022: 251,
        2023: 250,
        2024: 252
    }

    results = {
        'year': year,
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check required columns
    required_cols = {'date', 'rank', 'ticker'}
    if not required_cols.issubset(submission.columns):
        results['valid'] = False
        results['errors'].append(
            f"Missing columns: {required_cols - set(submission.columns)}"
        )
        return results

    # Check row count
    expected_rows = expected_days.get(year, 250) * 100
    if len(submission) != expected_rows:
        results['warnings'].append(
            f"Row count mismatch: expected {expected_rows}, got {len(submission)}"
        )

    # Check ranks per date
    for date in submission['date'].unique():
        day_ranks = submission[submission['date'] == date]['rank'].values
        expected_ranks = list(range(1, 101))

        if sorted(day_ranks) != expected_ranks:
            results['errors'].append(f"Invalid ranks on {date}")
            results['valid'] = False

    # Check for duplicate (date, ticker) pairs
    duplicates = submission.duplicated(subset=['date', 'ticker'])
    if duplicates.any():
        results['errors'].append(
            f"Duplicate (date, ticker) pairs: {duplicates.sum()}"
        )
        results['valid'] = False

    # Check tickers in universe
    if universe is not None:
        invalid_tickers = set(submission['ticker']) - set(universe)
        if invalid_tickers:
            results['warnings'].append(
                f"Tickers not in universe: {len(invalid_tickers)}"
            )

    return results


def backtest_strategy(
    submission: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str = 'target_3m',
    date_col: str = 'date',
    ticker_col: str = 'ticker'
) -> Dict:
    """
    Backtest the prediction strategy

    Calculates actual returns if we invested equally in Top-100 stocks

    Args:
        submission: Submission with predictions
        panel: Panel data with actual returns
        target_col: Actual return column
        date_col: Date column
        ticker_col: Ticker column

    Returns:
        Dictionary with backtest results
    """
    submission = submission.copy()
    panel = panel.copy()

    submission[date_col] = pd.to_datetime(submission[date_col])
    panel[date_col] = pd.to_datetime(panel[date_col])

    # Merge to get actual returns
    merged = submission.merge(
        panel[[date_col, ticker_col, target_col]],
        on=[date_col, ticker_col],
        how='left'
    )

    # Calculate daily portfolio return (equal weight)
    daily_returns = merged.groupby(date_col)[target_col].mean()

    # Calculate metrics
    results = {
        'mean_return': daily_returns.mean(),
        'median_return': daily_returns.median(),
        'std_return': daily_returns.std(),
        'sharpe_ratio': daily_returns.mean() / (daily_returns.std() + 1e-10),
        'min_return': daily_returns.min(),
        'max_return': daily_returns.max(),
        'positive_days': (daily_returns > 0).sum(),
        'negative_days': (daily_returns < 0).sum(),
        'total_days': len(daily_returns),
        'hit_rate': (daily_returns > 0).mean()
    }

    return results


def compare_to_baseline(
    submission: pd.DataFrame,
    baseline_submission: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str = 'target_3m'
) -> Dict:
    """
    Compare submission to baseline

    Args:
        submission: New submission
        baseline_submission: Baseline submission
        panel: Panel data with actual returns

    Returns:
        Dictionary with comparison results
    """
    # Backtest both
    new_results = backtest_strategy(submission, panel, target_col)
    baseline_results = backtest_strategy(baseline_submission, panel, target_col)

    comparison = {
        'new_mean_return': new_results['mean_return'],
        'baseline_mean_return': baseline_results['mean_return'],
        'improvement': new_results['mean_return'] - baseline_results['mean_return'],
        'improvement_pct': (
            (new_results['mean_return'] - baseline_results['mean_return']) /
            (abs(baseline_results['mean_return']) + 1e-10) * 100
        ),
        'new_sharpe': new_results['sharpe_ratio'],
        'baseline_sharpe': baseline_results['sharpe_ratio'],
        'new_hit_rate': new_results['hit_rate'],
        'baseline_hit_rate': baseline_results['hit_rate']
    }

    return comparison
