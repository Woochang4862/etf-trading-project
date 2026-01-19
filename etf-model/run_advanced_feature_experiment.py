#!/usr/bin/env python3
"""
Advanced Feature Engineering Experiment Script

Tests baseline vs enhanced features on best performing models:
- CatBoost (score: 0.10486, depth=6, lr=0.005, n=500)
- XGBoost (score: 0.15925, depth=4, lr=0.005, n=500)
- Random Forest (score: 0.19395, depth=20, mf=0.05, n=500)
- LightGBM (default params)

Usage:
    # Test all models on 2024 with enhanced features
    python run_advanced_feature_experiment.py --years 2024

    # Compare baseline vs enhanced on single model
    python run_advanced_feature_experiment.py --model xgboost --years 2024
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.experiment_pipeline import ExperimentPipeline
from src.models.factory import get_available_models
from src.utils.evaluation import backtest_strategy


def run_feature_comparison(
    model_name: str, year: int = 2024, max_features: int = 100
) -> Dict[str, float]:
    """
    Run feature comparison: baseline vs enhanced features

    Returns:
        Dictionary mapping feature group to score
    """
    print(f"\n{'=' * 60}")
    print(f"Testing {model_name.upper()} on year {year}")
    print(f"{'=' * 60}")

    results = {}

    # Test with all features (enhanced)
    print(f"Testing ENHANCED features ({max_features} max)...")

    try:
        model_pipeline = ExperimentPipeline(
            model_name=model_name,
            max_features=max_features,
            max_train_samples=50000,
            device="cpu",
            pred_chunk_size=5000,
        )

        panel_full = model_pipeline.load_data_for_year(year, train_years=5)
        submission_full = model_pipeline.process_year(
            year, train_years=5, verbose=False, panel=panel_full
        )
        metrics_full = backtest_strategy(
            submission_full, panel_full, target_col="target_3m"
        )
        results["enhanced"] = metrics_full.get("mean_return", 0)

        print(f"  Enhanced Score: {results['enhanced']:.6f}")
    except Exception as e:
        print(f"  Enhanced FAILED: {e}")
        results["enhanced"] = np.nan

    gc.collect()

    return results


def run_comparison(
    models: List[str], years: List[int] = [2024], max_features: int = 100
) -> None:
    """
    Run comparison for all models
    """
    print("\n" + "=" * 70)
    print("ADVANCED FEATURE ENGINEERING EXPERIMENT")
    print("=" * 70)
    print(f"Models: {', '.join(models)}")
    print(f"Years: {years}")
    print(f"Max features: {max_features}")
    print("=" * 70)

    all_results = {}

    for model_name in models:
        model_results = {}

        for year in years:
            print(f"\n{'=' * 70}")
            print(f"Model: {model_name.upper()} - Year: {year}")
            print(f"{'=' * 70}")

            year_results = run_feature_comparison(model_name, year, max_features)
            model_results[year] = year_results

            gc.collect()

        all_results[model_name] = model_results

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for model_name, model_results in all_results.items():
        print(f"\n{model_name.upper()}:")

        for year, year_results in model_results.items():
            enhanced_score = year_results.get("enhanced", np.nan)

            if not np.isnan(enhanced_score):
                print(f"  Year {year}: {enhanced_score:.6f}")
            else:
                print(f"  Year {year}: FAILED")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    for model_name, model_results in all_results.items():
        enhanced_scores = [r.get("enhanced", np.nan) for r in model_results.values()]
        valid_enhanced = [s for s in enhanced_scores if not np.isnan(s)]

        if valid_enhanced:
            avg_enhanced = np.mean(valid_enhanced)

            print(f"{model_name.upper()}:")
            print(f"  Avg Score: {avg_enhanced:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Feature Engineering Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["xgboost", "catboost", "random_forest", "lightgbm"],
        choices=get_available_models(),
        help="Models to test",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2024], help="Years to predict"
    )
    parser.add_argument(
        "--features", type=int, default=100, help="Maximum number of features"
    )
    parser.add_argument(
        "--samples", type=int, default=50000, help="Max training samples"
    )
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for m in get_available_models():
            print(f"  - {m}")
        print("\nBest performing models (2020 grid search):")
        print("  - CatBoost:    score 0.10486 (depth=6, lr=0.005, n=500)")
        print("  - XGBoost:     score 0.15925 (depth=4, lr=0.005, n=500)")
        print("  - Random Forest: score 0.19395 (depth=20, mf=0.05, n=500)")
        print("\nNote: Using enhanced features (~215 new features)")
        return

    # Run comparison
    run_comparison(models=args.models, years=args.years, max_features=args.features)


if __name__ == "__main__":
    main()
