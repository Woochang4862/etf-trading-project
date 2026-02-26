#!/usr/bin/env python3
"""
Quick experiment runner for Mac

Usage:
    # 단일 모델, 단일 연도 테스트 (빠름)
    python run_experiment.py --model ridge --year 2024 --features 50

    # 여러 모델 비교
    python run_experiment.py --model ridge lasso random_forest --year 2024

    # 전체 연도 실험
    python run_experiment.py --model ridge --year 2020 2021 2022 2023 2024

    # AhnLab LGBM 모델 (lambdarank 기반, 특수 학습 파이프라인 사용)
    python run_experiment.py --model ahnlab_lgbm --year 2024

    # AhnLab LGBM 모델 with FeaturePipeline (live data generation)
    python run_experiment.py --model ahnlab_lgbm --year 2024 --use-pipeline

    # 사용 가능한 모델 목록 확인
    python run_experiment.py --list
"""
import argparse
import sys
import time
import gc
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.factory import get_available_models


def load_ahnlab_panel(
    data_dir: Path = Path("data"),
    use_pipeline: bool = False,
    tickers: list = None,
    start_date: str = "2010-01-01",
    end_date: str = None,
    data_provider: str = "yfinance",
) -> pd.DataFrame:
    """Load panel data from parquet or generate via FeaturePipeline

    Args:
        data_dir: Directory for parquet files
        use_pipeline: If True, use FeaturePipeline to generate data
        tickers: List of tickers (required if use_pipeline=True)
        start_date: Start date for pipeline
        end_date: End date for pipeline (defaults to today)
        data_provider: Data source ("yfinance" or "mysql")

    Returns:
        Panel DataFrame with features

    Raises:
        FileNotFoundError: If panel data not found and use_pipeline=False
        ValueError: If use_pipeline=True but tickers not provided
    """
    if use_pipeline:
        from src.features.pipeline import FeaturePipeline

        if not tickers:
            raise ValueError("tickers required when use_pipeline=True")

        if not end_date:
            end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

        print(f"Generating features via FeaturePipeline...")
        print(f"  Data provider: {data_provider}")
        print(f"  Tickers: {len(tickers)}")
        print(f"  Date range: {start_date} to {end_date}")

        pipeline = FeaturePipeline(
            data_provider=data_provider,
            include_macro=True,
        )
        panel = pipeline.create_panel(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            shift_features=False,  # Model handles shifting
        )
        return panel

    # Original parquet loading logic
    parquet_path = data_dir / "stock_panel_data.parquet"
    csv_path = data_dir / "stock_panel_data.csv"

    if parquet_path.exists():
        print(f"Loading pre-downloaded panel data from {parquet_path}")
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"Loading pre-downloaded panel data from {csv_path}")
        return pd.read_csv(csv_path, parse_dates=["date"])
    else:
        raise FileNotFoundError(
            f"Pre-downloaded panel data not found in {data_dir}/\n"
            f"Run the following command to download data:\n"
            f"  python scripts/download_ahnlab_data.py --years 2020 2021 2022 2023 2024"
        )


def run_quick_test():
    """빠른 모델 테스트 (2024년만, 적은 피처)"""
    from src.experiment_pipeline import ExperimentPipeline

    # 테스트할 모델들 (sklearn 기반, 설치 필요 없음)
    # Note: ahnlab_lgbm uses different training pipeline (fit_with_panel)
    models_to_test = ['ridge', 'random_forest']

    print("\n" + "="*60)
    print("QUICK MODEL TEST (Mac)")
    print("="*60)
    print(f"Testing models: {models_to_test}")
    print(f"Year: 2024 only")
    print(f"Features: 50 (reduced for speed)")
    print("="*60)

    # Pre-load data once
    print("\nPre-loading 2024 data for shared use...")
    loader = ExperimentPipeline(model_name=models_to_test[0], max_features=50)
    panel_2024 = loader.load_data_for_year(2024, train_years=3)
    print("Data loaded.\n")

    results = {}

    for model_name in models_to_test:
        print(f"\n>>> Testing {model_name}...")
        start = time.time()

        try:
            if model_name == 'ahnlab_lgbm':
                # Special handling for ahnlab_lgbm in quick test
                from src.models.ahnlab_lgbm import AhnLabLGBMRankingModel

                # Try to use pre-downloaded data, fallback to loaded panel
                try:
                    ahnlab_panel = load_ahnlab_panel(use_pipeline=False)
                except FileNotFoundError:
                    print("  Warning: Using standard panel data instead of pre-downloaded data")
                    ahnlab_panel = panel_2024

                model = AhnLabLGBMRankingModel()
                model.fit_with_panel(ahnlab_panel, pred_year=2024)
                submission = model.predict_top_k_for_year(ahnlab_panel, 2024, k=100)
            else:
                # Standard pipeline
                pipeline = ExperimentPipeline(
                    model_name=model_name,
                    max_features=50,
                    max_train_samples=10000,
                    device='cpu',
                    pred_chunk_size=5000
                )
                submission = pipeline.process_year(2024, train_years=3, verbose=True, panel=panel_2024)

            elapsed = time.time() - start

            results[model_name] = {
                'status': 'OK',
                'rows': len(submission),
                'time': elapsed
            }
            print(f"  {model_name}: OK ({len(submission)} rows, {elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start
            results[model_name] = {
                'status': 'FAILED',
                'error': str(e),
                'time': elapsed
            }
            print(f"  {model_name}: FAILED - {e}")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for model, info in results.items():
        status = info['status']
        time_s = info['time']
        if status == 'OK':
            print(f"  {model:<20} {status:<8} {info['rows']:>6} rows  {time_s:>6.1f}s")
        else:
            print(f"  {model:<20} {status:<8} {info.get('error', '')[:40]}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run ML experiments on Mac')
    parser.add_argument('--model', type=str, nargs='+',
                        default=['ridge'],
                        help='Model(s) to test')
    parser.add_argument('--year', type=int, nargs='+',
                        default=[2024],
                        help='Year(s) to predict')
    parser.add_argument('--features', type=int, default=50,
                        help='Number of features (default: 50 for quick test)')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Max training samples (default: 10000)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with multiple models')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--use-pipeline', action='store_true',
                        help='Use FeaturePipeline instead of pre-downloaded parquet')
    parser.add_argument('--data-provider', type=str, default='yfinance',
                        choices=['yfinance', 'mysql'],
                        help='Data provider for --use-pipeline (default: yfinance)')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for m in get_available_models():
            print(f"  - {m}")
        print("\nNote: xgboost, catboost require installation:")
        print("  pip install xgboost catboost")
        return

    if args.quick:
        run_quick_test()
        return

    # Run specified experiments
    from src.experiment_pipeline import ExperimentPipeline

    all_paths = {}

    # Check if ahnlab_lgbm is in models to test
    has_ahnlab = 'ahnlab_lgbm' in args.model

    # For ahnlab_lgbm, load panel data
    if has_ahnlab:
        try:
            if args.use_pipeline:
                print("\n>>> Generating panel data via FeaturePipeline for ahnlab_lgbm...")

                # Collect all tickers from universe files
                all_tickers = set()
                for year in args.year:
                    universe_file = Path("data") / f"{year}_final_universe.csv"
                    if universe_file.exists():
                        df = pd.read_csv(universe_file)
                        all_tickers.update(df["ticker"].str.strip().tolist())

                if not all_tickers:
                    raise FileNotFoundError(
                        f"No universe files found for years {args.year}\n"
                        f"Expected files: data/{{year}}_final_universe.csv"
                    )

                # Set end_date to the last prediction year's end
                max_year = max(args.year)
                ahnlab_panel = load_ahnlab_panel(
                    use_pipeline=True,
                    tickers=list(all_tickers),
                    start_date="2010-01-01",
                    end_date=f"{max_year}-12-31",
                    data_provider=args.data_provider,
                )
            else:
                print("\n>>> Loading pre-downloaded panel data for ahnlab_lgbm...")
                ahnlab_panel = load_ahnlab_panel()

            print(f"Panel loaded: {len(ahnlab_panel)} rows, columns: {list(ahnlab_panel.columns[:10])}...")
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            # Remove ahnlab_lgbm from models to test
            args.model = [m for m in args.model if m != 'ahnlab_lgbm']
            has_ahnlab = False

    # Use first non-ahnlab model as loader for standard models
    standard_models = [m for m in args.model if m != 'ahnlab_lgbm']
    if standard_models:
        loader = ExperimentPipeline(
            model_name=standard_models[0],
            max_features=args.features,
            max_train_samples=args.samples,
            device='cpu'
        )

    for year in args.year:
        print(f"\n{'='*60}")
        print(f"Processing Year: {year}")
        print(f"{'='*60}")

        # 1. Load data for standard models (non-ahnlab)
        panel = None
        if standard_models:
            try:
                panel = loader.load_data_for_year(year, train_years=5)
            except Exception as e:
                print(f"Error loading data for {year}: {e}")
                # Skip standard models but continue with ahnlab if available
                if not has_ahnlab:
                    continue

        # 2. Run all models on this data
        for model_name in args.model:
            print(f"\n>>> Running {model_name} for {year}")

            try:
                if model_name == 'ahnlab_lgbm':
                    # Special handling for ahnlab_lgbm
                    from src.models.ahnlab_lgbm import AhnLabLGBMRankingModel

                    model = AhnLabLGBMRankingModel()
                    print(f"Training ahnlab_lgbm with panel data for year {year}...")
                    model.fit_with_panel(ahnlab_panel, pred_year=year)

                    # Generate predictions and save submission
                    print(f"Generating predictions for {year}...")
                    submission = model.predict_top_k_for_year(ahnlab_panel, year, k=100)

                    # Save submission file
                    output_dir = Path("submissions") / "ahnlab_lgbm"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{year}.ahnlab_lgbm.submission.csv"
                    submission.to_csv(output_path, index=False)
                    print(f"Saved submission to {output_path}")

                    # Collect results
                    if model_name not in all_paths:
                        all_paths[model_name] = {}
                    all_paths[model_name][year] = str(output_path)

                else:
                    # Standard model pipeline
                    pipeline = ExperimentPipeline(
                        model_name=model_name,
                        max_features=args.features,
                        max_train_samples=args.samples,
                        device='cpu'
                    )

                    paths = pipeline.run(
                        pred_years=[year],
                        train_years=5,
                        verbose=True,
                        panels={year: panel}
                    )

                    # Collect results
                    if model_name not in all_paths:
                        all_paths[model_name] = {}
                    all_paths[model_name].update(paths)

            except Exception as e:
                print(f"Error with {model_name} on {year}: {e}")
                import traceback
                traceback.print_exc()

        # 3. Clean up
        if panel is not None:
            del panel
        gc.collect()

    print(f"\n{'='*60}")
    print("All Experiments Completed")
    print(f"{'='*60}")
    
    for model_name, paths in all_paths.items():
        print(f"\nSubmission files for {model_name}:")
        for year, path in sorted(paths.items()):
            print(f"  {year}: {path}")


if __name__ == "__main__":
    main()
