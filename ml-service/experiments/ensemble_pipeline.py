"""
Ensemble Experiment Pipeline for ETF Stock Prediction Competition

Trains multiple models and combines their predictions using ensemble strategies.
Supports year-by-year execution for data download stability.

Usage:
    python -m src.ensemble_pipeline --name top3_rank_avg \
        --models xgboost catboost random_forest --strategy rank_avg \
        --year 2024 --features 100
"""
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from tqdm import tqdm

from .config import config, DATA_DIR, SUBMISSIONS_DIR
from .data.loader import DataLoader
from .data.preprocessor import Preprocessor, optimize_memory
from .features.technical import add_technical_features, TECHNICAL_FEATURES
from .features.momentum import add_momentum_features, MOMENTUM_FEATURES
from .features.volatility import add_volatility_features, VOLATILITY_FEATURES
from .features.volume import add_volume_features, VOLUME_FEATURES
from .features.returns import add_return_features, RETURN_FEATURES
from .features.cross_sectional import add_cross_sectional_features, CROSS_SECTIONAL_FEATURES
from .features.enhanced import (
    add_enhanced_features,
    add_enhanced_cross_sectional,
    ENHANCED_FEATURES,
    ENHANCED_CROSS_SECTIONAL_FEATURES
)
from .models.factory import create_model, get_available_models
from .utils.evaluation import validate_submission


class EnsembleExperimentPipeline:
    """
    Ensemble experiment pipeline for combining multiple models

    Key features:
    - Load data once per year, share across models
    - Support multiple ensemble strategies (rank_avg, weighted, etc.)
    - Year-by-year execution for stability
    """

    STRATEGIES = ['prediction_avg', 'rank_avg', 'weighted', 'borda']

    def __init__(
        self,
        ensemble_name: str,
        model_names: List[str],
        strategy: str = 'rank_avg',
        model_params: Optional[Dict[str, Dict]] = None,
        model_features: Optional[Dict[str, int]] = None,
        data_dir: Path = DATA_DIR,
        output_dir: Path = SUBMISSIONS_DIR,
        max_features: int = 100,
        max_train_samples: int = 50000,
        pred_chunk_size: int = 5000,
        timestamp: str = None
    ):
        """
        Initialize ensemble pipeline

        Args:
            ensemble_name: Name for the ensemble (e.g., 'top3_rank_avg')
            model_names: List of model names to ensemble
            strategy: Combination strategy ('prediction_avg', 'rank_avg', 'weighted', 'borda')
            model_params: Per-model hyperparameters
            model_features: Per-model feature counts (e.g., {'tabpfn': 100, 'random_forest': 150})
            data_dir: Data directory
            output_dir: Output directory for submissions
            max_features: Maximum number of features to select (default for models not in model_features)
            max_train_samples: Maximum training samples
            pred_chunk_size: Chunk size for batch prediction
            timestamp: Timestamp for submission filename
        """
        self.ensemble_name = ensemble_name
        self.model_names = model_names
        self.strategy = strategy
        self.model_params = model_params or {}
        self.model_features = model_features or {}
        self.data_dir = data_dir
        self.output_dir = output_dir / f"ensemble_{ensemble_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_features = max_features
        self.max_train_samples = max_train_samples
        self.pred_chunk_size = pred_chunk_size
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.loader = DataLoader(data_dir)
        self.preprocessor = Preprocessor(method='robust')
        self.feature_cols: List[str] = self._get_all_feature_cols()
        self.selected_features: List[str] = []

        self.results: Dict[int, pd.DataFrame] = {}
        self.scores: Dict[int, Dict] = {}

        # Validate inputs
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {self.STRATEGIES}")

        available = get_available_models()
        for name in model_names:
            if name not in available:
                raise ValueError(f"Unknown model: {name}. Available: {available}")

    def _get_all_feature_cols(self) -> List[str]:
        """Get list of all feature columns"""
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
        features = list(dict.fromkeys([f for f in all_features if f != 'target_3m']))
        return features

    def _select_features(
        self,
        panel: pd.DataFrame,
        max_features: int = 100
    ) -> List[str]:
        """Select top features by correlation with target"""
        all_features = [c for c in self.feature_cols if c in panel.columns]
        valid_panel = panel[panel['target_3m'].notna()].copy()

        if len(valid_panel) == 0:
            return all_features[:max_features]

        correlations = {}
        sample = valid_panel.sample(n=min(50000, len(valid_panel)), random_state=42)

        for col in all_features:
            if col in sample.columns:
                try:
                    corr = sample[col].corr(sample['target_3m'])
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
                except Exception:
                    continue

        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected = [f for f, _ in sorted_features[:max_features]]

        if len(selected) < 10:
            selected = all_features[:max_features]

        return selected

    def _create_features_for_ticker(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> Optional[pd.DataFrame]:
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
            tmp = add_enhanced_features(tmp)

            tmp['date'] = tmp.index
            tmp['ticker'] = ticker
            tmp = tmp.replace([np.inf, -np.inf], np.nan)

            return tmp

        except Exception:
            return None

    def _get_required_dates(self, pred_year: int) -> List[str]:
        """Get required prediction dates from sample submission"""
        sample_path = self.data_dir / f"{pred_year}_sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            dates = sample['date'].unique().tolist()
            return sorted(dates)
        return []

    def _prepare_training_data(
        self,
        panel: pd.DataFrame,
        pred_year: int,
        train_years: int = 5
    ):
        """Prepare training and validation data"""
        train_start = f"{pred_year - train_years}-01-01"

        pred_start_ts = pd.Timestamp(f"{pred_year}-01-01")
        cutoff_date = pred_start_ts - pd.Timedelta(days=95)
        train_end = cutoff_date.strftime('%Y-%m-%d')

        valid_start = (cutoff_date - pd.Timedelta(days=180)).strftime('%Y-%m-%d')

        # Training data
        train_mask = (
            (panel['date'] >= train_start) &
            (panel['date'] < valid_start) &
            panel['target_3m'].notna()
        )
        train_data = panel[train_mask].copy()

        # Validation data
        valid_mask = (
            (panel['date'] >= valid_start) &
            (panel['date'] <= train_end) &
            panel['target_3m'].notna()
        )
        valid_data = panel[valid_mask].copy()

        # Sampling
        if len(train_data) > self.max_train_samples:
            train_data = train_data.sort_values('date', ascending=False)
            recent_count = self.max_train_samples // 2
            recent_data = train_data.head(recent_count)
            older_data = train_data.iloc[recent_count:].sample(
                n=min(self.max_train_samples - recent_count, len(train_data) - recent_count),
                random_state=42
            )
            train_data = pd.concat([recent_data, older_data])

        X_train = train_data[self.selected_features].fillna(0)
        y_train = train_data['target_3m']

        X_valid = None
        y_valid = None
        if len(valid_data) > 0:
            if len(valid_data) > self.max_train_samples // 2:
                valid_data = valid_data.sample(n=self.max_train_samples // 2, random_state=42)
            X_valid = valid_data[self.selected_features].fillna(0)
            y_valid = valid_data['target_3m']

        return X_train, y_train, X_valid, y_valid

    def load_data_for_year(
        self,
        pred_year: int,
        train_years: int = 5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Load and prepare data panel for a specific year"""
        if verbose:
            print(f"Loading data for year {pred_year}...")

        data_start = f"{pred_year - train_years - 1}-01-01"
        pred_end = f"{pred_year}-12-31"

        universe = self.loader.load_universe(pred_year)
        if verbose:
            print(f"Universe: {len(universe)} tickers")

        if verbose:
            print(f"Loading data from {data_start} to {pred_end}...")

        frames = []
        failed = 0
        iterator = tqdm(universe, desc="Loading & features") if verbose else universe

        for ticker in iterator:
            df = self.loader.load_stock_data(ticker, data_start, pred_end)
            if df is None:
                failed += 1
                continue

            featured = self._create_features_for_ticker(df, ticker)
            if featured is not None:
                frames.append(featured)

            if len(frames) % 100 == 0:
                gc.collect()

        if verbose:
            print(f"Loaded {len(frames)}/{len(universe)} tickers (failed: {failed})")

        if not frames:
            raise ValueError(f"No valid data for {pred_year}")

        panel = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

        panel['date'] = pd.to_datetime(panel['date'])
        panel = optimize_memory(panel)

        if verbose:
            mem_mb = panel.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"Panel: {len(panel):,} rows, {mem_mb:.1f} MB")

        if verbose:
            print("Adding cross-sectional features...")
        panel = add_cross_sectional_features(panel)
        panel = add_enhanced_cross_sectional(panel)

        return panel

    def _combine_predictions(
        self,
        all_predictions: Dict[str, np.ndarray],
        validation_scores: Dict[str, float]
    ) -> np.ndarray:
        """Combine predictions using specified strategy"""

        preds_list = [all_predictions[name] for name in self.model_names]
        preds_matrix = np.column_stack(preds_list)

        if self.strategy == 'prediction_avg':
            return np.mean(preds_matrix, axis=1)

        elif self.strategy == 'rank_avg':
            ranks_list = []
            for i in range(preds_matrix.shape[1]):
                ranks = rankdata(-preds_matrix[:, i])  # 1 = highest
                ranks_list.append(ranks)
            ranks_matrix = np.column_stack(ranks_list)
            return -np.mean(ranks_matrix, axis=1)  # Negative so higher = better

        elif self.strategy == 'weighted':
            weights = np.array([validation_scores[n] for n in self.model_names])
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(len(self.model_names)) / len(self.model_names)
            return np.sum(preds_matrix * weights, axis=1)

        elif self.strategy == 'borda':
            n = preds_matrix.shape[0]
            scores = np.zeros(n)
            for i in range(preds_matrix.shape[1]):
                ranks = rankdata(-preds_matrix[:, i])
                scores += (n - ranks + 1)
            return scores

        else:
            return np.mean(preds_matrix, axis=1)

    def process_year(
        self,
        pred_year: int,
        train_years: int = 5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Process a single year with ensemble"""
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Ensemble Pipeline - {self.ensemble_name.upper()}")
            print(f"Processing {pred_year}")
            print(f"  Models: {self.model_names}")
            print(f"  Strategy: {self.strategy}")
            print(f"  Max features: {self.max_features}")
            print(f"{'='*60}")

        required_dates = self._get_required_dates(pred_year)
        if verbose and required_dates:
            print(f"Required dates: {len(required_dates)} days")

        # 1. Load data
        panel = self.load_data_for_year(pred_year, train_years, verbose)

        # 2. Feature selection
        pred_start_ts = pd.Timestamp(f"{pred_year}-01-01")
        feature_sel_cutoff = pred_start_ts - pd.Timedelta(days=95)

        # Select features for each model (different feature counts)
        max_feature_count = max(
            [self.model_features.get(m, self.max_features) for m in self.model_names]
        )
        if verbose:
            print(f"\nSelecting top {max_feature_count} features (max across models)...")
            print(f"  Using data prior to {feature_sel_cutoff.date()}")
            for m in self.model_names:
                feat_count = self.model_features.get(m, self.max_features)
                print(f"  {m}: {feat_count} features")

        history_panel = panel[panel['date'] <= feature_sel_cutoff]
        if len(history_panel) < 1000:
            history_panel = panel[panel['date'] < pred_start_ts]

        # Select max features needed (individual models will use subset)
        self.selected_features = self._select_features(history_panel, max_feature_count)
        if verbose:
            print(f"Selected {len(self.selected_features)} features")
            print(f"Top 5: {self.selected_features[:5]}")

        # 3. Prepare training data
        X_train, y_train, X_valid, y_valid = self._prepare_training_data(
            panel, pred_year, train_years
        )

        if verbose:
            print(f"\nTraining samples: {len(X_train):,}")
            if X_valid is not None:
                print(f"Validation samples: {len(X_valid):,}")

        # 4. Prepare prediction data
        if required_dates:
            target_dates = [pd.to_datetime(d) for d in required_dates]
        else:
            pred_end = f"{pred_year}-12-31"
            pred_panel_temp = panel[(panel['date'] >= pred_start_ts) & (panel['date'] <= pred_end)]
            target_dates = sorted(pred_panel_temp['date'].unique())

        pred_panel = panel[panel['date'].isin(target_dates)].copy()
        X_pred_all = pred_panel[self.selected_features].fillna(0)

        if verbose:
            print(f"Prediction samples: {len(X_pred_all):,}")

        # 5. Train models and collect predictions
        all_predictions = {}
        validation_scores = {}

        print(f"\n--- Training {len(self.model_names)} models ---")

        for model_name in self.model_names:
            # Get model-specific feature count
            model_feat_count = self.model_features.get(model_name, self.max_features)
            model_features = self.selected_features[:model_feat_count]
            
            print(f"\n[{model_name}] (features: {len(model_features)})")
            model_start = time.time()

            # Prepare model-specific training data
            X_train_model = X_train[model_features]
            X_valid_model = X_valid[model_features] if X_valid is not None else None

            params = self.model_params.get(model_name, {})
            model = create_model(model_name, params)
            model.fit(X_train_model, y_train, X_valid_model, y_valid, model_features)

            # Validation score
            if X_valid_model is not None:
                val_preds = model.predict(X_valid_model)
                corr, _ = spearmanr(val_preds, y_valid)
                corr = corr if not np.isnan(corr) else 0.0
                validation_scores[model_name] = max(0.0, corr)
                print(f"  Validation corr: {corr:.4f}")
            else:
                validation_scores[model_name] = 0.5

            # Chunk-based prediction
            chunk_size = self.pred_chunk_size
            predictions_list = []
            n_chunks = (len(X_pred_all) + chunk_size - 1) // chunk_size

            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(X_pred_all))
                X_chunk = X_pred_all.iloc[start_idx:end_idx][model_features]
                chunk_preds = model.predict(X_chunk)
                predictions_list.append(chunk_preds)

            all_predictions[model_name] = np.concatenate(predictions_list)

            model_time = time.time() - model_start
            print(f"  Time: {model_time:.1f}s")

            del model
            gc.collect()

        # 6. Combine predictions
        print(f"\n--- Combining with {self.strategy} strategy ---")
        combined = self._combine_predictions(all_predictions, validation_scores)
        pred_panel['prediction'] = combined

        # Store scores
        self.scores[pred_year] = validation_scores

        del X_train, y_train, X_valid, y_valid, X_pred_all
        gc.collect()

        # 7. Generate submission
        if verbose:
            print(f"\nSelecting Top-100 for {len(target_dates)} dates...")

        results = []
        available_dates = set(pred_panel['date'].unique())

        for date in tqdm(target_dates, desc="Ranking") if verbose else target_dates:
            date_ts = pd.Timestamp(date)
            date_str = date_ts.strftime('%Y-%m-%d')

            if date_ts not in available_dates:
                if results:
                    prev_preds = results[-100:]
                    for rank, pred in enumerate(prev_preds, 1):
                        results.append({
                            'date': date_str,
                            'rank': rank,
                            'ticker': pred['ticker']
                        })
                continue

            day_data = pred_panel[pred_panel['date'] == date_ts]
            top_100 = day_data.nlargest(100, 'prediction')

            for rank, (_, row) in enumerate(top_100.iterrows(), 1):
                results.append({
                    'date': date_str,
                    'rank': rank,
                    'ticker': row['ticker']
                })

        submission = pd.DataFrame(results)

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nGenerated {len(submission):,} predictions ({len(target_dates)} days)")
            print(f"Year {pred_year} completed in {elapsed/60:.1f} minutes")
            print(f"\nValidation scores: {validation_scores}")

        del panel, pred_panel
        gc.collect()

        return submission

    def run(
        self,
        pred_years: List[int] = None,
        train_years: int = 5,
        verbose: bool = True
    ) -> Dict[int, Path]:
        """Run ensemble pipeline for all years"""
        if pred_years is None:
            pred_years = config.data.pred_years

        start_time = time.time()

        print("\n" + "="*60)
        print(f"ENSEMBLE PIPELINE - {self.ensemble_name.upper()}")
        print("="*60)
        print(f"Models: {self.model_names}")
        print(f"Strategy: {self.strategy}")
        print(f"Max features: {self.max_features}")
        print(f"Years: {pred_years}")

        paths = {}

        for year in pred_years:
            try:
                submission = self.process_year(year, train_years, verbose)
                self.results[year] = submission

                filepath = self.output_dir / f"{year}.{self.ensemble_name}.{self.timestamp}.submission.csv"
                submission.to_csv(filepath, index=False)
                paths[year] = filepath

                if verbose:
                    print(f"Saved: {filepath}")

                gc.collect()

            except Exception as e:
                print(f"Error processing {year}: {e}")
                import traceback
                traceback.print_exc()
                continue

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Ensemble completed in {elapsed/60:.1f} minutes")
        print(f"{'='*60}")

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
    import argparse

    parser = argparse.ArgumentParser(description='Run Ensemble Experiment Pipeline')
    parser.add_argument('--name', type=str, required=True,
                        help='Ensemble name (e.g., top3_rank_avg)')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help=f'Model names: {get_available_models()}')
    parser.add_argument('--strategy', type=str, default='rank_avg',
                        choices=['prediction_avg', 'rank_avg', 'weighted', 'borda'],
                        help='Combination strategy')
    parser.add_argument('--year', type=int, nargs='+', default=None,
                        help='Prediction year(s)')
    parser.add_argument('--train-years', type=int, default=5)
    parser.add_argument('--features', type=int, default=100,
                        help='Default max features (used if model not in --model-features)')
    parser.add_argument('--model-features', type=str, nargs='+', default=None,
                        help='Per-model feature counts (format: model:count, e.g., tabpfn:100 random_forest:150)')
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    pred_years = args.year if args.year else [2020, 2021, 2022, 2023, 2024]

    # Parse model-features
    model_features = {}
    if args.model_features:
        for item in args.model_features:
            if ':' in item:
                model_name, feat_count = item.split(':')
                model_features[model_name] = int(feat_count)

    pipeline = EnsembleExperimentPipeline(
        ensemble_name=args.name,
        model_names=args.models,
        strategy=args.strategy,
        model_features=model_features,
        max_features=args.features,
        max_train_samples=args.samples,
        pred_chunk_size=args.chunk_size,
        timestamp=args.timestamp
    )

    paths = pipeline.run(
        pred_years=pred_years,
        train_years=args.train_years,
        verbose=not args.quiet
    )

    print("\nSubmission files:")
    for year, path in paths.items():
        print(f"  {year}: {path}")

    print("\nValidation scores:")
    for year, scores in pipeline.scores.items():
        print(f"  {year}: {scores}")


if __name__ == "__main__":
    main()
