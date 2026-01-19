"""
Hyperparameter Tuning Experiment Runner with Grid Search

Performs grid search across 3 hyperparameters with 5 values each,
for 3 models (XGBoost, CatBoost, Random Forest).
Total experiments: 5^3 * 3 = 375

Usage:
    python run_tuning_experiment.py --year 2020
    python run_tuning_experiment.py --year 2020 2021 2022 2023 2024
    python run_tuning_experiment.py --year 2020 --dry-run  # Show experiment count without running
"""

import argparse
import gc
import itertools
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import requests
from bs4 import BeautifulSoup

# Project paths
PROJECT_ROOT = Path(__file__).parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Server configuration
LOGIN_URL = "http://ahnbi1.suwon.ac.kr:5151/login"
SUBMIT_URL = "http://ahnbi1.suwon.ac.kr:5151/submit_test"
USERNAME = "21016082"
PASSWORD = "asdf0706@@"

# =============================================================================
# GRID SEARCH CONFIGURATION
# =============================================================================
# Each model has 3 key hyperparameters with 5 values each
# Total combinations per model: 5^3 = 125
# Total experiments: 125 * 4 models = 500

GRID_SEARCH_CONFIG = {
    "xgboost": {
        "param_grid": {
            "max_depth": [4, 6, 8, 10, 12],
            "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1],
            "n_estimators": [500, 750, 1000, 1250, 1500],
        },
        "fixed_params": {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 0.2,
            "min_child_weight": 3,
        },
    },
    "catboost": {
        "param_grid": {
            "depth": [4, 6, 8, 10, 12],
            "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1],
            "iterations": [500, 750, 1000, 1250, 1500],
        },
        "fixed_params": {
            "l2_leaf_reg": 5.0,
            "subsample": 0.8,
            "border_count": 254,
            "random_strength": 0.5,
        },
    },
    "random_forest": {
        "param_grid": {
            "max_depth": [10, 15, 20, 25, 30],
            "max_features": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_estimators": [500, 750, 1000, 1250, 1500],
        },
        "fixed_params": {
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
    },
    "lightgbm": {
        "param_grid": {
            "num_leaves": [31, 63, 127, 255, 511],
            "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1],
            "n_estimators": [500, 750, 1000, 1250, 1500],
        },
        "fixed_params": {
            "objective": "regression",
            "boosting_type": "gbdt",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 0.2,
            "min_child_samples": 20,
            "use_lambdarank": False,
        },
    },
}


def generate_param_combinations(model_name: str) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for a model"""
    config = GRID_SEARCH_CONFIG[model_name]
    param_grid = config["param_grid"]
    fixed_params = config["fixed_params"]

    # Get all keys and values
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # Generate all combinations
    combinations = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params.update(fixed_params)
        combinations.append(params)

    return combinations


def generate_short_name(model_name: str, params: Dict[str, Any]) -> str:
    """Generate a short descriptive name for the parameter combination"""
    if model_name == "xgboost":
        return f"xgb_d{params['max_depth']}_lr{str(params['learning_rate']).replace('.', '')}_n{params['n_estimators']}"
    elif model_name == "catboost":
        return f"cb_d{params['depth']}_lr{str(params['learning_rate']).replace('.', '')}_n{params['iterations']}"
    elif model_name == "random_forest":
        mf = str(params["max_features"]).replace(".", "")
        return f"rf_d{params['max_depth']}_mf{mf}_n{params['n_estimators']}"
    elif model_name == "lightgbm":
        return f"lgb_l{params['num_leaves']}_lr{str(params['learning_rate']).replace('.', '')}_n{params['n_estimators']}"
    return f"{model_name}_{hash(str(params)) % 10000}"


def get_total_experiments() -> Tuple[int, Dict[str, int]]:
    """Calculate total number of experiments"""
    model_counts = {}
    total = 0

    for model_name in GRID_SEARCH_CONFIG:
        count = len(generate_param_combinations(model_name))
        model_counts[model_name] = count
        total += count

    return total, model_counts


def get_score_from_server(session, file_path: Path, year: int) -> float:
    """Submit file to server and get score"""
    try:
        with open(file_path, "rb") as f:
            files_data = {"file": f}
            data = {"year": str(year)}

            r = session.post(SUBMIT_URL, data=data, files=files_data)

            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.get_text(strip=True)
                score_match = re.search(r"점수\s*=\s*([\d\.]+)", text)
                if score_match:
                    return float(score_match.group(1))
        return None
    except Exception as e:
        print(f"Error getting score: {e}")
        return None


def run_single_experiment(
    model_name: str,
    params: Dict[str, Any],
    year: int,
    panel: "pd.DataFrame",
    features: int,
    experiment_num: int,
    total_experiments: int,
    timestamp: str,
) -> Tuple[Path, Dict[str, Any]]:
    """Run a single experiment with specific parameters"""
    from src.experiment_pipeline import ExperimentPipeline

    short_name = generate_short_name(model_name, params)

    # Create output directory for this experiment
    output_dir = SUBMISSIONS_DIR / f"grid_search_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[{experiment_num}/{total_experiments}] Running {model_name.upper()} - {short_name}"
    )
    print(f"  Params: {params}")

    start_time = time.time()
    result = {
        "model": model_name,
        "params": params,
        "short_name": short_name,
        "experiment_num": experiment_num,
        "year": year,
        "status": "pending",
    }

    try:
        # Run pipeline with pre-loaded panel
        pipeline = ExperimentPipeline(
            model_name=model_name,
            model_params=params,
            max_features=features,
            max_train_samples=50000,
            device="auto",
            pred_chunk_size=5000,
            timestamp=timestamp,
        )

        # Override output directory
        pipeline.output_dir = output_dir

        # Use pre-loaded panel data
        paths = pipeline.run(
            pred_years=[year],
            train_years=5,
            verbose=False,  # Reduce output for grid search
            panels={year: panel},
        )

        elapsed = time.time() - start_time
        result["training_time_sec"] = round(elapsed, 2)

        if year in paths:
            old_path = paths[year]
            new_filename = f"{year}.{short_name}.{timestamp}.submission.csv"
            new_path = output_dir / new_filename

            if old_path.exists():
                old_path.rename(new_path)
                result["status"] = "success"
                result["submission_file"] = str(new_path)
                print(f"  ✓ Completed in {elapsed:.1f}s - {new_filename}")
                return new_path, result

        result["status"] = "failed"
        result["error"] = "No submission file generated"
        print(f"  ✗ Failed - No submission file")

    except Exception as e:
        elapsed = time.time() - start_time
        result["status"] = "error"
        result["error"] = str(e)
        result["training_time_sec"] = round(elapsed, 2)
        print(f"  ✗ Error in {elapsed:.1f}s: {e}")

    # Cleanup
    gc.collect()

    return None, result


def run_grid_search_for_year(
    year: int, features: int = 150, skip_scoring: bool = False
):
    """Run grid search for all models for a specific year"""
    from src.experiment_pipeline import ExperimentPipeline

    total_experiments, model_counts = get_total_experiments()

    print(f"\n{'#' * 70}")
    print(f"# GRID SEARCH EXPERIMENT - Year {year}")
    print(f"# Total experiments: {total_experiments}")
    for model, count in model_counts.items():
        print(f"#   {model}: {count} combinations")
    print(f"{'#' * 70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_results = {
        "year": year,
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "total_experiments": total_experiments,
        "experiments": [],
    }

    # Load data ONCE and reuse for ALL experiments
    print(f"\n{'=' * 60}")
    print(
        f"Loading data for year {year} (will be reused by all {total_experiments} experiments)"
    )
    print(f"{'=' * 60}")

    data_load_start = time.time()

    temp_pipeline = ExperimentPipeline(
        model_name="xgboost",
        max_features=features,
        max_train_samples=50000,
        device="auto",
    )
    panel = temp_pipeline.load_data_for_year(year, train_years=5, verbose=True)

    data_load_time = time.time() - data_load_start
    print(f"Data loaded in {data_load_time:.1f}s")
    print(f"Panel shape: {panel.shape}")

    experiment_results["data_load_time_sec"] = round(data_load_time, 2)

    # Track submission files for scoring
    submission_files = []

    # Run all experiments
    experiment_num = 0
    total_start_time = time.time()

    for model_name in GRID_SEARCH_CONFIG:
        print(f"\n{'=' * 60}")
        print(
            f"Starting {model_name.upper()} experiments ({model_counts[model_name]} combinations)"
        )
        print(f"{'=' * 60}")

        param_combinations = generate_param_combinations(model_name)

        for params in param_combinations:
            experiment_num += 1

            submission_path, result = run_single_experiment(
                model_name=model_name,
                params=params,
                year=year,
                panel=panel,
                features=features,
                experiment_num=experiment_num,
                total_experiments=total_experiments,
                timestamp=timestamp,
            )

            experiment_results["experiments"].append(result)

            if submission_path:
                submission_files.append({"path": submission_path, "result": result})

            # Periodic save (every 25 experiments)
            if experiment_num % 25 == 0:
                interim_file = (
                    RESULTS_DIR / f"grid_search_{year}_interim_{timestamp}.json"
                )
                with open(interim_file, "w", encoding="utf-8") as f:
                    json.dump(experiment_results, f, indent=2, ensure_ascii=False)
                print(
                    f"\n  [Interim save: {experiment_num}/{total_experiments} experiments]"
                )

    total_elapsed = time.time() - total_start_time
    experiment_results["total_training_time_sec"] = round(total_elapsed, 2)

    print(f"\n{'=' * 60}")
    print(
        f"All {total_experiments} experiments completed in {total_elapsed / 60:.1f} minutes"
    )
    print(f"{'=' * 60}")

    # Get scores from server
    if not skip_scoring and submission_files:
        print(f"\n{'=' * 60}")
        print(f"Submitting {len(submission_files)} successful experiments to server...")
        print(f"{'=' * 60}")

        with requests.Session() as session:
            print(f"Logging in as {USERNAME}...")
            payload = {"username": USERNAME, "password": PASSWORD}
            r = session.post(LOGIN_URL, data=payload)

            if r.status_code != 200:
                print(f"Login failed (Status {r.status_code})")
                experiment_results["server_status"] = "login_failed"
            else:
                print("Login successful.")
                experiment_results["server_status"] = "ok"

                for i, item in enumerate(submission_files):
                    file_path = item["path"]
                    result = item["result"]

                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i + 1}/{len(submission_files)} scored...")

                    score = get_score_from_server(session, file_path, year)

                    # Update result in experiments list
                    for exp in experiment_results["experiments"]:
                        if exp.get("short_name") == result["short_name"]:
                            exp["score"] = score
                            break

                    # Brief pause to avoid overwhelming server
                    time.sleep(0.5)

    # Calculate statistics
    scores = [
        exp.get("score")
        for exp in experiment_results["experiments"]
        if exp.get("score")
    ]
    if scores:
        experiment_results["score_statistics"] = {
            "count": len(scores),
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "best_experiment": max(
                [exp for exp in experiment_results["experiments"] if exp.get("score")],
                key=lambda x: x["score"],
            ),
        }

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"GRID SEARCH SUMMARY - Year {year}")
    print(f"{'=' * 70}")

    # Summary by model
    for model_name in GRID_SEARCH_CONFIG:
        model_exps = [
            exp
            for exp in experiment_results["experiments"]
            if exp["model"] == model_name
        ]
        model_scores = [exp.get("score") for exp in model_exps if exp.get("score")]

        success = sum(1 for exp in model_exps if exp["status"] == "success")
        failed = sum(1 for exp in model_exps if exp["status"] in ["failed", "error"])

        print(f"\n{model_name.upper()}:")
        print(f"  Completed: {success}/{len(model_exps)}", end="")
        if failed > 0:
            print(f" (Failed: {failed})", end="")
        print()

        if model_scores:
            best_score = max(model_scores)
            best_exp = max(
                [exp for exp in model_exps if exp.get("score")],
                key=lambda x: x["score"],
            )
            avg_score = sum(model_scores) / len(model_scores)
            print(f"  Score Range: {min(model_scores):.5f} - {max(model_scores):.5f}")
            print(f"  Average Score: {avg_score:.5f}")
            print(f"  Best Config: {best_exp['short_name']} ({best_score:.5f})")

    # Overall best
    if scores:
        print(f"\n{'=' * 70}")
        best = experiment_results["score_statistics"]["best_experiment"]
        print(f"OVERALL BEST: {best['model']} - {best['short_name']}")
        print(f"  Score: {best['score']:.5f}")
        print(f"  Params: {best['params']}")

    # Save final results
    results_file = RESULTS_DIR / f"grid_search_{year}_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")

    # Generate CSV summary
    csv_file = RESULTS_DIR / f"grid_search_{year}_{timestamp}.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        # Header
        f.write(
            "experiment_num,model,short_name,score,training_time_sec,status,params\n"
        )
        for exp in sorted(
            experiment_results["experiments"],
            key=lambda x: x.get("score") or 0,
            reverse=True,
        ):
            score = exp.get("score", "")
            f.write(
                f'{exp["experiment_num"]},{exp["model"]},{exp["short_name"]},{score},{exp.get("training_time_sec", "")},{exp["status"]},"{exp["params"]}"\n'
            )

    print(f"CSV summary saved to: {csv_file}")

    return experiment_results


def main():
    parser = argparse.ArgumentParser(
        description="Run grid search hyperparameter tuning experiment"
    )
    parser.add_argument(
        "--year",
        type=int,
        nargs="+",
        required=True,
        help="Year(s) to run experiment for (e.g., 2020 or 2020 2021 2022)",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=150,
        help="Number of features to use (default: 150)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show experiment count and configuration without running",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip server scoring (only train models)",
    )

    args = parser.parse_args()

    total_experiments, model_counts = get_total_experiments()

    # Show configuration
    print(f"\n{'=' * 70}")
    print("GRID SEARCH CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Total experiments per year: {total_experiments}")
    print(f"Years to process: {args.year}")
    print(f"Grand total experiments: {total_experiments * len(args.year)}")
    print()

    for model_name, config in GRID_SEARCH_CONFIG.items():
        print(f"\n{model_name.upper()} ({model_counts[model_name]} combinations):")
        for param, values in config["param_grid"].items():
            print(f"  {param}: {values}")
        print(f"  Fixed: {config['fixed_params']}")

    if args.dry_run:
        print(f"\n{'=' * 70}")
        print("DRY RUN - No experiments will be executed")
        print(f"{'=' * 70}")
        return

    # Confirm before running
    print(f"\n{'=' * 70}")
    print(f"Ready to run {total_experiments * len(args.year)} experiments")
    print(
        f"Estimated time: {total_experiments * len(args.year) * 2 / 60:.0f} - {total_experiments * len(args.year) * 5 / 60:.0f} minutes"
    )
    print(f"{'=' * 70}")

    all_results = {}

    for year in args.year:
        results = run_grid_search_for_year(year, args.features, args.skip_scoring)
        all_results[year] = results

    # If multiple years, save combined results
    if len(args.year) > 1:
        combined_file = (
            RESULTS_DIR
            / f"grid_search_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Find best configurations across all years
        all_experiments = []
        for year, results in all_results.items():
            for exp in results.get("experiments", []):
                exp["year"] = year
                all_experiments.append(exp)

        # Group by configuration and calculate average scores
        config_scores = {}
        for exp in all_experiments:
            key = (exp["model"], exp["short_name"])
            if key not in config_scores:
                config_scores[key] = {
                    "scores": [],
                    "params": exp["params"],
                    "model": exp["model"],
                }
            if exp.get("score"):
                config_scores[key]["scores"].append(exp["score"])

        # Calculate averages
        config_averages = []
        for (model, short_name), data in config_scores.items():
            if data["scores"]:
                avg = sum(data["scores"]) / len(data["scores"])
                config_averages.append(
                    {
                        "model": model,
                        "short_name": short_name,
                        "params": data["params"],
                        "average_score": avg,
                        "years_count": len(data["scores"]),
                        "scores": data["scores"],
                    }
                )

        config_averages.sort(key=lambda x: x["average_score"], reverse=True)

        combined = {
            "years": args.year,
            "timestamp": datetime.now().isoformat(),
            "features": args.features,
            "total_experiments": total_experiments * len(args.year),
            "results_by_year": {str(k): v for k, v in all_results.items()},
            "best_configurations": config_averages[:20],  # Top 20 configurations
        }

        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n{'#' * 70}")
        print("COMBINED RESULTS - TOP 10 CONFIGURATIONS")
        print(f"{'#' * 70}")
        print(f"{'Rank':<5} {'Model':<15} {'Config':<30} {'Avg Score':<12} {'Years'}")
        print("-" * 70)

        for i, cfg in enumerate(config_averages[:10], 1):
            print(
                f"{i:<5} {cfg['model']:<15} {cfg['short_name']:<30} {cfg['average_score']:.5f}      {cfg['years_count']}"
            )

        print(f"\nCombined results saved to: {combined_file}")


if __name__ == "__main__":
    main()
