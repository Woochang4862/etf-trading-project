#!/usr/bin/env python3
"""
Feature Engineering Experiment Script

Tests different feature sets to find the best combination:
1. Baseline (existing features)
2. + Advanced Technical Indicators
3. + New Interaction Features
4. + Advanced Decomposition Features
5. All Combined

Usage:
    python run_feature_engineering_experiment.py --year 2020
    python run_feature_engineering_experiment.py --year 2020 --model xgboost
    python run_feature_engineering_experiment.py --year 2020 --dry-run
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config, DATA_DIR, SUBMISSIONS_DIR
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor, optimize_memory
from src.features.technical import add_technical_features, TECHNICAL_FEATURES
from src.features.momentum import add_momentum_features, MOMENTUM_FEATURES
from src.features.volatility import add_volatility_features, VOLATILITY_FEATURES
from src.features.volume import add_volume_features, VOLUME_FEATURES
from src.features.returns import add_return_features, RETURN_FEATURES
from src.features.cross_sectional import (
    add_cross_sectional_features,
    CROSS_SECTIONAL_FEATURES,
)
from src.features.enhanced import (
    add_enhanced_features,
    add_enhanced_cross_sectional,
    ENHANCED_FEATURES,
    ENHANCED_CROSS_SECTIONAL_FEATURES,
)
from src.features.patterns import add_pattern_features, PATTERN_FEATURES
from src.features.regime import add_regime_features, REGIME_FEATURES
from src.features.decomposition import (
    add_decomposition_features,
    DECOMPOSITION_FEATURES,
)
from src.features.interaction import add_interaction_features, INTERACTION_FEATURES
from src.features.autocorr import add_autocorr_features, AUTOCORR_FEATURES
from src.features.portfolio import add_portfolio_features, PORTFOLIO_FEATURES
from src.features.advanced_technical import (
    add_advanced_technical_features,
    ADVANCED_TECHNICAL_FEATURES,
)
from src.features.new_interactions import (
    add_new_interaction_features,
    NEW_INTERACTION_FEATURES,
)
from src.features.advanced_decomposition import (
    add_advanced_decomposition_features,
    ADVANCED_DECOMPOSITION_FEATURES,
)
from src.models.factory import create_model, get_available_models
from src.utils.evaluation import backtest_strategy

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


BASELINE_FEATURES = (
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


class FeatureEngineeringExperiment:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        output_dir: Path = SUBMISSIONS_DIR,
        max_features: int = 150,
        max_train_samples: int = 50000,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_features = max_features
        self.max_train_samples = max_train_samples
        self.loader = DataLoader(data_dir)
        self.preprocessor = Preprocessor(method="robust")
        self.results: Dict[str, Any] = {}

    def _create_baseline_features(
        self, df: pd.DataFrame, ticker: str
    ) -> Optional[pd.DataFrame]:
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

    def _create_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_advanced_technical_features(df)

    def _create_new_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_new_interaction_features(df)

    def _create_advanced_decomposition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_advanced_decomposition_features(df)

    def load_data_with_feature_set(
        self,
        pred_year: int,
        feature_set: str,
        train_years: int = 5,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        if verbose:
            print(f"\nLoading data for {pred_year} with feature set: {feature_set}")

        data_start = f"{pred_year - train_years - 1}-01-01"
        pred_end = f"{pred_year}-12-31"

        universe = self.loader.load_universe(pred_year)
        if verbose:
            print(f"Universe: {len(universe)} tickers")

        frames = []
        failed = 0
        iterator = tqdm(universe, desc="Loading data") if verbose else universe

        for ticker in iterator:
            df = self.loader.load_stock_data(ticker, data_start, pred_end)
            if df is None:
                failed += 1
                continue

            featured = self._create_baseline_features(df, ticker)
            if featured is None:
                failed += 1
                continue

            if feature_set in ["advanced_technical", "all"]:
                featured = self._create_advanced_technical_features(featured)

            if feature_set in ["new_interactions", "all"]:
                featured = self._create_new_interaction_features(featured)

            if feature_set in ["advanced_decomposition", "all"]:
                featured = self._create_advanced_decomposition_features(featured)

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

        panel["date"] = pd.to_datetime(panel["date"])
        panel = optimize_memory(panel)

        if verbose:
            print("Adding cross-sectional features...")
        panel = add_cross_sectional_features(panel)
        panel = add_enhanced_cross_sectional(panel)
        panel = add_portfolio_features(panel)

        feature_cols = self._get_feature_cols(feature_set)
        available_features = [c for c in feature_cols if c in panel.columns]

        if verbose:
            print(f"Panel: {len(panel):,} rows, {len(available_features)} features")

        return panel, available_features

    def _get_feature_cols(self, feature_set: str) -> List[str]:
        base_features = list(BASELINE_FEATURES)

        if feature_set == "baseline":
            return base_features
        elif feature_set == "advanced_technical":
            return base_features + list(ADVANCED_TECHNICAL_FEATURES)
        elif feature_set == "new_interactions":
            return base_features + list(NEW_INTERACTION_FEATURES)
        elif feature_set == "advanced_decomposition":
            return base_features + list(ADVANCED_DECOMPOSITION_FEATURES)
        elif feature_set == "all":
            return (
                base_features
                + list(ADVANCED_TECHNICAL_FEATURES)
                + list(NEW_INTERACTION_FEATURES)
                + list(ADVANCED_DECOMPOSITION_FEATURES)
            )
        else:
            return base_features

    def _select_features(
        self, panel: pd.DataFrame, feature_cols: List[str], max_features: int = 150
    ) -> List[str]:
        all_features = [c for c in feature_cols if c in panel.columns]
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

    def _prepare_training_data(
        self,
        panel: pd.DataFrame,
        selected_features: List[str],
        pred_year: int,
        train_years: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        train_start = f"{pred_year - train_years}-01-01"
        pred_start_ts = pd.Timestamp(f"{pred_year}-01-01")
        cutoff_date = pred_start_ts - pd.Timedelta(days=95)
        train_end = cutoff_date.strftime("%Y-%m-%d")
        valid_start = (cutoff_date - pd.Timedelta(days=180)).strftime("%Y-%m-%d")

        train_mask = (
            (panel["date"] >= train_start)
            & (panel["date"] < valid_start)
            & panel["target_3m"].notna()
        )
        train_data = panel[train_mask].copy()

        valid_mask = (
            (panel["date"] >= valid_start)
            & (panel["date"] <= train_end)
            & panel["target_3m"].notna()
        )
        valid_data = panel[valid_mask].copy()

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

        X_train = train_data[selected_features].fillna(0)
        y_train = train_data["target_3m"]

        X_valid = None
        y_valid = None
        if len(valid_data) > 0:
            if len(valid_data) > self.max_train_samples // 2:
                valid_data = valid_data.sample(
                    n=self.max_train_samples // 2, random_state=42
                )
            X_valid = valid_data[selected_features].fillna(0)
            y_valid = valid_data["target_3m"]

        return X_train, y_train, X_valid, y_valid

    def run_single_experiment(
        self,
        model_name: str,
        feature_set: str,
        pred_year: int,
        panel: pd.DataFrame,
        feature_cols: List[str],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        experiment_name = f"{model_name}_{feature_set}"
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Experiment: {experiment_name}")
            print(f"{'=' * 60}")

        start_time = time.time()
        result = {
            "model": model_name,
            "feature_set": feature_set,
            "year": pred_year,
            "status": "pending",
        }

        try:
            pred_start_ts = pd.Timestamp(f"{pred_year}-01-01")
            feature_sel_cutoff = pred_start_ts - pd.Timedelta(days=95)
            history_panel = panel[panel["date"] <= feature_sel_cutoff]

            if len(history_panel) < 1000:
                history_panel = panel[panel["date"] < pred_start_ts]

            selected_features = self._select_features(
                history_panel, feature_cols, self.max_features
            )
            result["num_features_selected"] = len(selected_features)
            result["top_features"] = selected_features[:10]

            if verbose:
                print(f"Selected {len(selected_features)} features")
                print(f"Top 5: {selected_features[:5]}")

            X_train, y_train, X_valid, y_valid = self._prepare_training_data(
                panel, selected_features, pred_year
            )

            if verbose:
                print(f"Training samples: {len(X_train):,}")
                if X_valid is not None:
                    print(f"Validation samples: {len(X_valid):,}")

            model = create_model(model_name)
            train_start = time.time()
            model.fit(
                X_train,
                y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                feature_names=selected_features,
            )
            result["training_time_sec"] = round(time.time() - train_start, 2)

            del X_train, y_train, X_valid, y_valid
            gc.collect()

            sample_path = self.data_dir / f"{pred_year}_sample_submission.csv"
            if sample_path.exists():
                sample = pd.read_csv(sample_path)
                required_dates = [pd.to_datetime(d) for d in sample["date"].unique()]
            else:
                required_dates = sorted(
                    panel[panel["date"] >= f"{pred_year}-01-01"]["date"].unique()
                )

            pred_panel = panel[panel["date"].isin(required_dates)].copy()
            X_pred = pred_panel[selected_features].fillna(0)

            predictions = model.predict(X_pred)
            pred_panel["prediction"] = predictions

            results_list = []
            for date in required_dates:
                date_ts = pd.Timestamp(date)
                date_str = date_ts.strftime("%Y-%m-%d")
                day_data = pred_panel[pred_panel["date"] == date_ts]

                if len(day_data) == 0:
                    continue

                top_100 = day_data.nlargest(100, "prediction")
                for rank, (_, row) in enumerate(top_100.iterrows(), 1):
                    results_list.append(
                        {"date": date_str, "rank": rank, "ticker": row["ticker"]}
                    )

            submission = pd.DataFrame(results_list)

            metrics = backtest_strategy(submission, panel, target_col="target_3m")
            result["backtest_score"] = metrics.get("mean_return", 0)
            result["sharpe_ratio"] = metrics.get("sharpe_ratio", 0)
            result["positive_days_ratio"] = metrics.get("positive_days", 0) / max(
                metrics.get("total_days", 1), 1
            )

            output_dir = self.output_dir / "feature_experiments"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"{pred_year}.{experiment_name}.{timestamp}.csv"
            submission.to_csv(filepath, index=False)
            result["submission_file"] = str(filepath)

            result["status"] = "success"
            result["total_time_sec"] = round(time.time() - start_time, 2)

            if verbose:
                print(f"Backtest Score: {result['backtest_score']:.6f}")
                print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
                print(f"Total time: {result['total_time_sec']:.1f}s")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["total_time_sec"] = round(time.time() - start_time, 2)
            if verbose:
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()

        gc.collect()
        return result

    def run_all_experiments(
        self,
        model_name: str,
        pred_year: int,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        feature_sets = [
            "baseline",
            "advanced_technical",
            "new_interactions",
            "advanced_decomposition",
            "all",
        ]

        experiment_results = {
            "model": model_name,
            "year": pred_year,
            "timestamp": datetime.now().isoformat(),
            "max_features": self.max_features,
            "experiments": [],
        }

        print(f"\n{'#' * 70}")
        print(f"# FEATURE ENGINEERING EXPERIMENT - Year {pred_year}")
        print(f"# Model: {model_name}")
        print(f"# Feature sets to test: {len(feature_sets)}")
        print(f"{'#' * 70}")

        for feature_set in feature_sets:
            print(f"\n{'=' * 70}")
            print(f"Loading data with feature set: {feature_set.upper()}")
            print(f"{'=' * 70}")

            panel, feature_cols = self.load_data_with_feature_set(
                pred_year, feature_set, train_years=5, verbose=verbose
            )

            result = self.run_single_experiment(
                model_name, feature_set, pred_year, panel, feature_cols, verbose
            )
            experiment_results["experiments"].append(result)

            del panel
            gc.collect()

        experiment_results["experiments"].sort(
            key=lambda x: x.get("backtest_score", -999), reverse=True
        )

        print(f"\n{'=' * 70}")
        print("FEATURE ENGINEERING RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(
            f"{'Feature Set':<25} {'Score':<12} {'Sharpe':<10} {'Time(s)':<10} {'Status'}"
        )
        print("-" * 70)

        for exp in experiment_results["experiments"]:
            score = exp.get("backtest_score", np.nan)
            sharpe = exp.get("sharpe_ratio", np.nan)
            time_sec = exp.get("total_time_sec", np.nan)
            status = exp.get("status", "unknown")

            score_str = f"{score:.6f}" if not np.isnan(score) else "N/A"
            sharpe_str = f"{sharpe:.4f}" if not np.isnan(sharpe) else "N/A"
            time_str = f"{time_sec:.1f}" if not np.isnan(time_sec) else "N/A"

            print(
                f"{exp['feature_set']:<25} {score_str:<12} {sharpe_str:<10} {time_str:<10} {status}"
            )

        best_exp = experiment_results["experiments"][0]
        print(f"\nBEST FEATURE SET: {best_exp['feature_set']}")
        print(f"  Score: {best_exp.get('backtest_score', 'N/A')}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = (
            RESULTS_DIR
            / f"feature_experiment_{model_name}_{pred_year}_{timestamp}.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {results_file}")

        return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Experiment")
    parser.add_argument("--year", type=int, default=2020, help="Year to test")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=get_available_models(),
        help="Model to use",
    )
    parser.add_argument(
        "--features", type=int, default=150, help="Max features to select"
    )
    parser.add_argument(
        "--samples", type=int, default=50000, help="Max training samples"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show configuration only"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output")

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("FEATURE ENGINEERING EXPERIMENT CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Year: {args.year}")
    print(f"Model: {args.model}")
    print(f"Max features: {args.features}")
    print(f"Max samples: {args.samples}")
    print(f"\nFeature sets to test:")
    print("  1. baseline - Existing features (~200)")
    print("  2. advanced_technical - + Ichimoku, Keltner, Donchian, etc. (~50)")
    print("  3. new_interactions - + Risk-adjusted, volume-price, composite (~35)")
    print("  4. advanced_decomposition - + FFT, Wavelet, Hilbert, SSA (~23)")
    print("  5. all - All features combined")

    if args.dry_run:
        print(f"\n{'=' * 70}")
        print("DRY RUN - No experiments will be executed")
        print(f"{'=' * 70}")
        return

    experiment = FeatureEngineeringExperiment(
        max_features=args.features,
        max_train_samples=args.samples,
    )

    results = experiment.run_all_experiments(
        model_name=args.model,
        pred_year=args.year,
        verbose=not args.quiet,
    )

    print(f"\nExperiment completed!")


if __name__ == "__main__":
    main()
