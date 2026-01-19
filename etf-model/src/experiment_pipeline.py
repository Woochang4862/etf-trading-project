"""
Unified Experiment Pipeline for ETF Stock Prediction Competition

Supports multiple ML models via factory pattern.
Competition rule compliant: single model per year.

Usage:
    python -m src.experiment_pipeline --model xgboost --features 150 --years 2020 2021 2022 2023 2024
"""

import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import config, DATA_DIR, SUBMISSIONS_DIR
from .data.loader import DataLoader
from .data.preprocessor import Preprocessor, optimize_memory
from .features.technical import add_technical_features, TECHNICAL_FEATURES
from .features.momentum import add_momentum_features, MOMENTUM_FEATURES
from .features.volatility import add_volatility_features, VOLATILITY_FEATURES
from .features.volume import add_volume_features, VOLUME_FEATURES
from .features.returns import add_return_features, RETURN_FEATURES
from .features.cross_sectional import (
    add_cross_sectional_features,
    CROSS_SECTIONAL_FEATURES,
)
from .features.enhanced import (
    add_enhanced_features,
    add_enhanced_cross_sectional,
    ENHANCED_FEATURES,
    ENHANCED_CROSS_SECTIONAL_FEATURES,
)
from .features.patterns import add_pattern_features, PATTERN_FEATURES
from .features.regime import add_regime_features, REGIME_FEATURES
from .features.decomposition import add_decomposition_features, DECOMPOSITION_FEATURES
from .features.interaction import add_interaction_features, INTERACTION_FEATURES
from .features.autocorr import add_autocorr_features, AUTOCORR_FEATURES
from .features.portfolio import add_portfolio_features, PORTFOLIO_FEATURES
from .models.factory import create_model, get_available_models
from .models.factory import create_model, get_available_models
from .utils.evaluation import validate_submission, backtest_strategy


class ExperimentPipeline:
    """
    Unified experiment pipeline supporting multiple ML models

    Competition rules:
    - Single model per year
    - Same preprocessing for all years
    - Same model architecture for all years
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[Dict] = None,
        data_dir: Path = DATA_DIR,
        output_dir: Path = SUBMISSIONS_DIR,
        max_features: int = 100,
        max_train_samples: int = 50000,
        device: str = "auto",
        pred_chunk_size: int = 5000,
        timestamp: str = None,
    ):
        """
        Initialize pipeline

        Args:
            model_name: Name of the model (e.g., 'xgboost', 'ridge')
            model_params: Model hyperparameters (overrides defaults)
            data_dir: Data directory
            output_dir: Output directory for submissions
            max_features: Maximum number of features to select
            max_train_samples: Maximum training samples
            device: Compute device ('auto', 'cuda', 'cpu')
            pred_chunk_size: Chunk size for batch prediction
            timestamp: Timestamp for submission filename
        """
        self.model_name = model_name.lower()
        self.model_params = model_params or {}
        self.data_dir = data_dir
        self.output_dir = output_dir / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_features = max_features
        self.max_train_samples = max_train_samples
        self.device = device
        self.pred_chunk_size = pred_chunk_size
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.loader = DataLoader(data_dir)
        self.preprocessor = Preprocessor(method="robust")
        self.feature_cols: List[str] = self._get_all_feature_cols()
        self.selected_features: List[str] = []
        self.results: Dict[int, pd.DataFrame] = {}
        self.metrics: Dict[int, Dict] = {}
        self.scores: Dict[int, float] = {}

        # Validate model name
        available = get_available_models()
        if self.model_name not in available:
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    def _get_all_feature_cols(self) -> List[str]:
        """Get list of all feature columns"""
        all_features = (
            TECHNICAL_FEATURES
            + MOMENTUM_FEATURES
            + VOLATILITY_FEATURES
            + VOLUME_FEATURES
            + RETURN_FEATURES
            + CROSS_SECTIONAL_FEATURES
            + ENHANCED_FEATURES
            + ENHANCED_CROSS_SECTIONAL_FEATURES
            + PATTERN_FEATURES
            + REGIME_FEATURES
            + DECOMPOSITION_FEATURES
            + INTERACTION_FEATURES
            + AUTOCORR_FEATURES
            + PORTFOLIO_FEATURES
        )
        features = list(dict.fromkeys([f for f in all_features if f != "target_3m"]))
        return features

    def _select_features(
        self, panel: pd.DataFrame, max_features: int = 100
    ) -> List[str]:
        """Select top features by correlation with target"""
        all_features = [c for c in self.feature_cols if c in panel.columns]
        valid_panel = panel[panel["target_3m"].notna()].copy()

        if len(valid_panel) == 0:
            return all_features[:max_features]

        correlations = {}
        sample = valid_panel.sample(n=min(50000, len(valid_panel)), random_state=42)

        for col in all_features:
            if col in sample.columns:
                try:
                    corr = sample[col].corr(sample["target_3m"])
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
        self, df: pd.DataFrame, ticker: str
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
            tmp = add_pattern_features(tmp)
            tmp = add_regime_features(tmp)
            tmp = add_decomposition_features(tmp)
            tmp = add_interaction_features(tmp)
            tmp = add_autocorr_features(tmp)

            tmp["date"] = tmp.index
            tmp["ticker"] = ticker
            tmp = tmp.replace([np.inf, -np.inf], np.nan)

            return tmp

        except Exception:
            return None

            tmp = df.copy()
            tmp = add_technical_features(tmp)
            tmp = add_momentum_features(tmp)
            tmp = add_volatility_features(tmp)
            tmp = add_volume_features(tmp)
            tmp = add_return_features(tmp)
            tmp = add_enhanced_features(tmp)

            tmp["date"] = tmp.index
            tmp["ticker"] = ticker
            tmp = tmp.replace([np.inf, -np.inf], np.nan)

            return tmp

        except Exception:
            return None

    def _get_required_dates(self, pred_year: int) -> List[str]:
        """Get required prediction dates from sample submission"""
        sample_path = self.data_dir / f"{pred_year}_sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            dates = sample["date"].unique().tolist()
            return sorted(dates)
        return []

    def _prepare_training_data(
        self, panel: pd.DataFrame, pred_year: int, train_years: int = 5
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare training and validation data

        Returns:
            (X_train, y_train, X_valid, y_valid)
        """
        train_start = f"{pred_year - train_years}-01-01"

        # Leakage prevention: cutoff 95 days before prediction year
        pred_start_ts = pd.Timestamp(f"{pred_year}-01-01")
        cutoff_date = pred_start_ts - pd.Timedelta(days=95)
        train_end = cutoff_date.strftime("%Y-%m-%d")

        # Validation: last 6 months before cutoff
        valid_start = (cutoff_date - pd.Timedelta(days=180)).strftime("%Y-%m-%d")

        # Training data
        train_mask = (
            (panel["date"] >= train_start)
            & (panel["date"] < valid_start)
            & panel["target_3m"].notna()
        )
        train_data = panel[train_mask].copy()

        # Validation data
        valid_mask = (
            (panel["date"] >= valid_start)
            & (panel["date"] <= train_end)
            & panel["target_3m"].notna()
        )
        valid_data = panel[valid_mask].copy()

        # Sampling for training
        if len(train_data) > self.max_train_samples:
            train_data = train_data.sort_values("date", ascending=False)
            recent_count = self.max_train_samples // 2
            recent_data = train_data.head(recent_count)
            older_data = train_data.iloc[recent_count:].sample(
                n=min(
                    self.max_train_samples - recent_count,
                    len(train_data) - recent_count,
                ),
                random_state=42,
            )
            train_data = pd.concat([recent_data, older_data])

        X_train = train_data[self.selected_features].fillna(0)
        y_train = train_data["target_3m"]

        X_valid = None
        y_valid = None
        if len(valid_data) > 0:
            # Sample validation data too
            if len(valid_data) > self.max_train_samples // 2:
                valid_data = valid_data.sample(
                    n=self.max_train_samples // 2, random_state=42
                )
            X_valid = valid_data[self.selected_features].fillna(0)
            y_valid = valid_data["target_3m"]

        return X_train, y_train, X_valid, y_valid

    def load_data_for_year(
        self, pred_year: int, train_years: int = 5, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Load and prepare data panel for a specific year (model-agnostic)

        This includes:
        1. Loading universe
        2. Loading OHLCV data
        3. generating technical/macro features
        4. generating cross-sectional features
        """
        if verbose:
            print(f"Loading data for year {pred_year}...")

        # Data loading range
        data_start = f"{pred_year - train_years - 1}-01-01"
        pred_end = f"{pred_year}-12-31"

        # Load universe
        universe = self.loader.load_universe(pred_year)
        if verbose:
            print(f"Universe: {len(universe)} tickers")

        # Load data and create features
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

        # Create panel
        panel = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

        panel["date"] = pd.to_datetime(panel["date"])
        panel = optimize_memory(panel)

        if verbose:
            mem_mb = panel.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"Panel: {len(panel):,} rows, {mem_mb:.1f} MB")

        # Add cross-sectional features
        if verbose:
            print("Adding cross-sectional features...")
        panel = add_cross_sectional_features(panel)
        panel = add_enhanced_cross_sectional(panel)

        # Add portfolio-level features
        if verbose:
            print("Adding portfolio-level features...")
        panel = add_portfolio_features(panel)

        return panel

    def process_year(
        self,
        pred_year: int,
        train_years: int = 5,
        verbose: bool = True,
        panel: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Process a single year

        1. Load data and create features (if panel not provided)
        2. Select features
        3. Train single model (competition compliant)
        4. Predict for entire year

        Args:
            pred_year: Year to predict
            train_years: Number of training years
            verbose: Print progress
            panel: Pre-loaded dataframe (optional, for caching)
        """
        start_time = time.time()

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Experiment Pipeline - {self.model_name.upper()}")
            print(f"Processing {pred_year}")
            print(f"  Model: {self.model_name}")
            print(f"  Device: {self.device}")
            print(f"  Max features: {self.max_features}")
            print(f"  Max train samples: {self.max_train_samples}")
            print(f"{'=' * 60}")

        required_dates = self._get_required_dates(pred_year)
        if verbose and required_dates:
            print(f"Required dates: {len(required_dates)} days")

        # 1. Load data or use cached panel
        if panel is None:
            panel = self.load_data_for_year(pred_year, train_years, verbose)
        else:
            if verbose:
                print("Using pre-loaded data panel")

        # Feature selection (with leakage prevention)
        pred_start_ts = pd.Timestamp(f"{pred_year}-01-01")
        feature_sel_cutoff = pred_start_ts - pd.Timedelta(days=95)

        if verbose:
            print(f"Selecting top {self.max_features} features...")
            print(f"  Using data prior to {feature_sel_cutoff.date()}")

        history_panel = panel[panel["date"] <= feature_sel_cutoff]
        if len(history_panel) < 1000:
            history_panel = panel[panel["date"] < pred_start_ts]

        self.selected_features = self._select_features(history_panel, self.max_features)
        if verbose:
            print(f"Selected {len(self.selected_features)} features")
            print(f"Top 5: {self.selected_features[:5]}")

        # ============================================
        # Train single model (competition compliant)
        # ============================================
        if verbose:
            print(f"\nTraining {self.model_name} model for {pred_year}...")

        X_train, y_train, X_valid, y_valid = self._prepare_training_data(
            panel, pred_year, train_years
        )

        if verbose:
            print(f"Training samples: {len(X_train):,}")
            if X_valid is not None:
                print(f"Validation samples: {len(X_valid):,}")

        # Create model via factory
        model_kwargs = {}
        if self.device != "auto" and self.model_name in ["xgboost", "catboost"]:
            model_kwargs["device"] = self.device

        model = create_model(self.model_name, self.model_params, **model_kwargs)

        # Train
        train_start_time = time.time()
        model.fit(
            X_train,
            y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            feature_names=self.selected_features,
        )
        train_time = time.time() - train_start_time

        if verbose:
            print(f"Training completed in {train_time:.1f}s")

        # Feature importance
        if verbose and model.get_feature_importance() is not None:
            top_imp = model.get_feature_importance(top_n=5)
            print("Top 5 features by importance:")
            for _, row in top_imp.iterrows():
                print(f"  {row['feature']}: {row['importance_pct']:.2f}%")

        del X_train, y_train, X_valid, y_valid
        gc.collect()

        # ============================================
        # Batch prediction for entire year
        # ============================================
        pred_start = f"{pred_year}-01-01"
        if required_dates:
            target_dates = [pd.to_datetime(d) for d in required_dates]
        else:
            pred_panel = panel[
                (panel["date"] >= pred_start) & (panel["date"] <= pred_end)
            ]
            target_dates = sorted(pred_panel["date"].unique())

        pred_panel = panel[panel["date"].isin(target_dates)].copy()

        if verbose:
            print(f"\nBatch predicting {len(pred_panel):,} rows...")

        # Chunk-based prediction
        chunk_size = self.pred_chunk_size
        X_pred_all = pred_panel[self.selected_features].fillna(0)

        predictions_list = []
        n_chunks = (len(X_pred_all) + chunk_size - 1) // chunk_size

        if verbose:
            print(f"  Processing in {n_chunks} chunks (chunk_size={chunk_size})")

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(X_pred_all))
            X_chunk = X_pred_all.iloc[start_idx:end_idx]

            chunk_preds = model.predict(X_chunk)
            predictions_list.append(chunk_preds)

            del X_chunk
            gc.collect()

            if verbose and (i + 1) % 20 == 0:
                print(f"  Chunk {i + 1}/{n_chunks} done")

        predictions_all = np.concatenate(predictions_list)
        pred_panel["prediction"] = predictions_all

        del X_pred_all, predictions_list
        gc.collect()

        if verbose:
            print(f"Selecting Top-100 for {len(target_dates)} dates...")

        # Select Top-100 for each date
        results = []
        available_dates = set(pred_panel["date"].unique())

        for date in tqdm(target_dates, desc="Ranking") if verbose else target_dates:
            date_ts = pd.Timestamp(date)
            date_str = date_ts.strftime("%Y-%m-%d")

            if date_ts not in available_dates:
                if results:
                    prev_preds = results[-100:]
                    for rank, pred in enumerate(prev_preds, 1):
                        results.append(
                            {"date": date_str, "rank": rank, "ticker": pred["ticker"]}
                        )
                continue

            day_data = pred_panel[pred_panel["date"] == date_ts]
            top_100 = day_data.nlargest(100, "prediction")

            for rank, (_, row) in enumerate(top_100.iterrows(), 1):
                results.append(
                    {"date": date_str, "rank": rank, "ticker": row["ticker"]}
                )

        del panel, model
        gc.collect()

        submission = pd.DataFrame(results)
        elapsed = time.time() - start_time

        if verbose:
            print(
                f"Generated {len(submission):,} predictions ({len(target_dates)} days)"
            )
            print(f"Year {pred_year} completed in {elapsed / 60:.1f} minutes")

        # Backtest
        try:
            metrics = backtest_strategy(submission, panel, target_col="target_3m")
            self.metrics[pred_year] = metrics
            if verbose:
                print(f"Backtest Score ({pred_year}):")
                print(f"  Mean Return:   {metrics['mean_return']:.6f}")
                print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.4f}")
                print(
                    f"  Positive Days: {metrics['positive_days']}/{metrics['total_days']}"
                )
        except Exception as e:
            print(f"Backtest failed for {pred_year}: {e}")

        return submission

    def run(
        self,
        pred_years: List[int] = None,
        train_years: int = 5,
        verbose: bool = True,
        panels: Dict[int, pd.DataFrame] = None,
    ) -> Dict[int, Path]:
        """Run pipeline for all years"""
        if pred_years is None:
            pred_years = config.data.pred_years

        start_time = time.time()

        print("\n" + "=" * 60)
        print(f"EXPERIMENT PIPELINE - {self.model_name.upper()}")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Model params: {self.model_params}")
        print(f"Device: {self.device}")
        print(f"Max features: {self.max_features}")
        print(f"Max train samples: {self.max_train_samples}")
        print(f"Years: {pred_years}")

        paths = {}

        for year in pred_years:
            try:
                panel = panels.get(year) if panels else None
                submission = self.process_year(year, train_years, verbose, panel=panel)
                self.results[year] = submission

                filepath = (
                    self.output_dir
                    / f"{year}.{self.model_name}.{self.timestamp}.submission.csv"
                )
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
        print(f"\n{'=' * 60}")
        print(f"Pipeline completed in {elapsed / 60:.1f} minutes")
        print(f"{'=' * 60}")

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

    parser = argparse.ArgumentParser(description="Run ML Experiment Pipeline")
    parser.add_argument(
        "--model", type=str, required=True, help=f"Model name: {get_available_models()}"
    )
    parser.add_argument("--year", type=int, nargs="+", default=None)
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    # Model-specific params (as JSON string)
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Model params as JSON string, e.g., '{\"alpha\": 0.5}'",
    )

    args = parser.parse_args()

    # Parse model params
    model_params = None
    if args.params:
        import json

        model_params = json.loads(args.params)

    pred_years = args.year if args.year else [2020, 2021, 2022, 2023, 2024]

    pipeline = ExperimentPipeline(
        model_name=args.model,
        model_params=model_params,
        max_features=args.features,
        max_train_samples=args.samples,
        device=args.device,
        pred_chunk_size=args.chunk_size,
        timestamp=args.timestamp,
    )

    paths = pipeline.run(
        pred_years=pred_years, train_years=args.train_years, verbose=not args.quiet
    )

    print("\nSubmission files:")
    for year, path in paths.items():
        print(f"  {year}: {path}")

    print(f"\nAvailable models: {get_available_models()}")


if __name__ == "__main__":
    main()
