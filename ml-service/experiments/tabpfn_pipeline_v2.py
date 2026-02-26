"""
TabPFN Pipeline V2 for ETF Stock Prediction Competition

규칙 준수 버전: 연도별 단일 모델 학습
- 각 연도 시작 전 1개 모델만 학습
- 해당 연도 전체 예측에 동일 모델 사용
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
from .models.tabpfn_model import TabPFNRankingModel
from .utils.evaluation import validate_submission


class TabPFNPipelineV2:
    """
    규칙 준수 TabPFN 파이프라인

    - 연도별 단일 모델 학습 (규칙 준수)
    - 동일한 전처리, 입력 구조, 모델 구조
    - 연도별 재학습만 수행
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        output_dir: Path = SUBMISSIONS_DIR,
        max_features: int = None,
        max_train_samples: int = None,
        device: str = None,
        n_estimators: int = None,
        pred_chunk_size: int = None,
        timestamp: str = None
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # 하이퍼파라미터 (모든 연도 동일)
        self.max_features = max_features or 100
        self.max_train_samples = max_train_samples or 50000
        self.device = device or "cpu"
        self.n_estimators = n_estimators or 8
        self.pred_chunk_size = pred_chunk_size or 5000

        # 타임스탬프 (제출 파일명에 사용)
        from datetime import datetime
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.loader = DataLoader(data_dir)
        self.preprocessor = Preprocessor(method='robust')
        self.feature_cols: List[str] = []
        self.selected_features: List[str] = []
        self.results: Dict[int, pd.DataFrame] = {}

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
        """Create features for a single ticker (동일한 전처리)"""
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

    def _prepare_training_data(
        self,
        panel: pd.DataFrame,
        pred_year: int,
        train_years: int = 5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        연도별 학습 데이터 준비

        pred_year 이전 train_years 기간의 데이터 사용
        """
        train_start = f"{pred_year - train_years}-01-01"
        
        # LEAKAGE PREVENTION FIX
        # Target is 3-month return, so target for date T is known at T + ~90 days
        # We must filter for dates where target is revealed before pred_year-01-01
        # otherwise we are using future information (prices from the prediction year)
        
        pred_start_ts = pd.Timestamp(f"{pred_year}-01-01")
        cutoff_date = pred_start_ts - pd.Timedelta(days=95)
        train_end = cutoff_date.strftime('%Y-%m-%d')
        
        print(f"  Training data cutoff: {train_end} (to prevent leakage)")

        # 학습 데이터 필터링
        mask = (
            (panel['date'] >= train_start) &
            (panel['date'] <= train_end) &
            panel['target_3m'].notna()
        )
        train_data = panel[mask].copy()

        # 샘플링 (TabPFN 제한 고려)
        if len(train_data) > self.max_train_samples:
            # 최근 데이터 우선 + 균등 샘플링
            train_data = train_data.sort_values('date', ascending=False)

            # 최근 50% + 나머지 랜덤 샘플링
            recent_count = self.max_train_samples // 2
            recent_data = train_data.head(recent_count)
            older_data = train_data.iloc[recent_count:].sample(
                n=min(self.max_train_samples - recent_count, len(train_data) - recent_count),
                random_state=42
            )
            train_data = pd.concat([recent_data, older_data])

        X = train_data[self.selected_features].fillna(0)
        y = train_data['target_3m']

        return X, y

    def process_year(
        self,
        pred_year: int,
        train_years: int = 5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        단일 연도 처리 (규칙 준수)

        1. 연도 시작 전 1개 모델만 학습
        2. 해당 연도 전체 예측에 동일 모델 사용
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TabPFN V2 Processing {pred_year}")
            print(f"  Device: {self.device}")
            print(f"  Max features: {self.max_features}")
            print(f"  Max train samples: {self.max_train_samples}")
            print(f"  N estimators: {self.n_estimators}")
            print(f"{'='*60}")

        required_dates = self._get_required_dates(pred_year)
        if verbose and required_dates:
            print(f"Required dates: {len(required_dates)} days")

        # 데이터 로딩 범위
        data_start = f"{pred_year - train_years - 1}-01-01"
        pred_end = f"{pred_year}-12-31"

        # Universe 로드
        universe = self.loader.load_universe(pred_year)
        if verbose:
            print(f"Universe: {len(universe)} tickers")

        # 데이터 로딩 및 피처 생성
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

        # 패널 생성
        panel = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

        panel['date'] = pd.to_datetime(panel['date'])
        panel = optimize_memory(panel)

        if verbose:
            mem_mb = panel.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"Panel: {len(panel):,} rows, {mem_mb:.1f} MB")

        # 횡단면 피처 추가
        if verbose:
            print("Adding cross-sectional features...")
        panel = add_cross_sectional_features(panel)
        panel = add_enhanced_cross_sectional(panel)

        # 피처 선택 (누수 방지 적용)
        # STRICT LEAKAGE PREVENTION: Only use data available before the start of the prediction year
        pred_start_ts = pd.Timestamp(f"{pred_year}-01-01")
        feature_sel_cutoff = pred_start_ts - pd.Timedelta(days=95)

        if verbose:
            print(f"Selecting top {self.max_features} features...")
            print(f"  Using data prior to {feature_sel_cutoff.date()} for feature selection")

        # Create safe history panel for feature selection
        history_panel = panel[panel['date'] <= feature_sel_cutoff]
        
        if len(history_panel) < 1000:
             # Fallback if insufficient history (mostly for very first year if history is short)
            if verbose:
                print("  Warning: Insufficient historical data for strict selection. Using data up to pred_start.")
            history_panel = panel[panel['date'] < pred_start_ts]

        self.selected_features = self._select_features(history_panel, self.max_features)
        if verbose:
            print(f"Selected {len(self.selected_features)} features")
            print(f"Top 5: {self.selected_features[:5]}")

        # ============================================
        # 핵심: 연도별 단일 모델 학습 (규칙 준수)
        # ============================================
        if verbose:
            print(f"\nTraining single model for {pred_year}...")

        X_train, y_train = self._prepare_training_data(panel, pred_year, train_years)

        if verbose:
            print(f"Training samples: {len(X_train):,}")

        # 단일 모델 생성 및 학습
        model = TabPFNRankingModel(
            n_estimators=self.n_estimators,
            device=self.device,
            ignore_pretraining_limits=True,
            memory_saving_mode=True,
            random_state=config.seed
        )
        model.fit(X_train, y_train, feature_names=self.selected_features)

        del X_train, y_train
        gc.collect()

        # ============================================
        # 동일 모델로 연도 전체 예측 (배치 처리)
        # ============================================
        pred_start = f"{pred_year}-01-01"
        if required_dates:
            target_dates = [pd.to_datetime(d) for d in required_dates]
        else:
            pred_panel = panel[(panel['date'] >= pred_start) & (panel['date'] <= pred_end)]
            target_dates = sorted(pred_panel['date'].unique())

        # 예측 기간 데이터 필터링
        pred_panel = panel[panel['date'].isin(target_dates)].copy()

        if verbose:
            print(f"\nBatch predicting {len(pred_panel):,} rows...")

        # GPU 메모리 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        # 청크 단위로 예측 (메모리 절약)
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

            # 매 청크마다 메모리 정리
            del X_chunk
            gc.collect()

            if verbose and (i + 1) % 10 == 0:
                print(f"  Chunk {i+1}/{n_chunks} done")

        predictions_all = np.concatenate(predictions_list)
        pred_panel['prediction'] = predictions_all

        del X_pred_all, predictions_list
        gc.collect()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        if verbose:
            print(f"Selecting Top-100 for {len(target_dates)} dates...")

        # 날짜별 Top-100 선택
        results = []
        available_dates = set(pred_panel['date'].unique())

        for date in tqdm(target_dates, desc="Ranking") if verbose else target_dates:
            date_ts = pd.Timestamp(date)
            date_str = date_ts.strftime('%Y-%m-%d')

            if date_ts not in available_dates:
                # 이전 예측 사용
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

        del panel, model
        gc.collect()

        submission = pd.DataFrame(results)
        if verbose:
            print(f"Generated {len(submission):,} predictions ({len(target_dates)} days)")

        return submission

    def run(
        self,
        pred_years: List[int] = None,
        train_years: int = 5,
        verbose: bool = True
    ) -> Dict[int, Path]:
        """Run pipeline for all years"""
        if pred_years is None:
            pred_years = config.data.pred_years

        start_time = time.time()

        print("\n" + "="*60)
        print("TABPFN V2 - 규칙 준수 버전")
        print("연도별 단일 모델 학습")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Max features: {self.max_features}")
        print(f"Max train samples: {self.max_train_samples}")
        print(f"N estimators: {self.n_estimators}")

        paths = {}

        for year in pred_years:
            try:
                submission = self.process_year(year, train_years, verbose)
                self.results[year] = submission

                filepath = self.output_dir / f"{year}.tabpfn_v2.{self.timestamp}.submission.csv"
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
        print(f"Pipeline completed in {elapsed/60:.1f} minutes")
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


def run_single_year(args_tuple):
    """단일 연도 처리 (멀티프로세싱용)"""
    year, gpu_id, max_features, max_train_samples, n_estimators, train_years, log_dir = args_tuple

    import os
    import sys
    from datetime import datetime

    # CUDA_VISIBLE_DEVICES는 torch import 전에 설정해야 함
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # torch를 여기서 import해서 CUDA_VISIBLE_DEVICES 적용
    import torch
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES로 인해 실제로는 gpu_id가 됨

    # 연도별 로그 파일 생성
    log_file = Path(log_dir) / f"tabpfn_v2_{year}_gpu{gpu_id}.log"
    log_file.parent.mkdir(exist_ok=True)

    # stdout/stderr를 로그 파일로 리다이렉트
    with open(log_file, 'w') as f:
        sys.stdout = f
        sys.stderr = f

        print(f"[{datetime.now()}] Starting {year} on GPU {gpu_id}")
        print(f"Log file: {log_file}")

        pipeline = TabPFNPipelineV2(
            max_features=max_features,
            max_train_samples=max_train_samples,
            device='cuda',
            n_estimators=n_estimators
        )

        try:
            submission = pipeline.process_year(year, train_years, verbose=True)
            filepath = pipeline.output_dir / f"{year}.tabpfn_v2.{pipeline.timestamp}.submission.csv"
            submission.to_csv(filepath, index=False)
            print(f"\n[{datetime.now()}] Saved: {filepath}")

            # 원래 stdout으로 복구하고 완료 메시지
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"[GPU {gpu_id}] {year} completed -> {log_file}")

            return (year, str(filepath), str(log_file))
        except Exception as e:
            import traceback
            print(f"\n[{datetime.now()}] Error: {e}")
            traceback.print_exc()

            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"[GPU {gpu_id}] {year} FAILED -> {log_file}")

            return (year, None, str(log_file))


def main():
    """Main entry point"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Run TabPFN V2 (규칙 준수)')
    parser.add_argument('--year', type=int, nargs='+', default=None)
    parser.add_argument('--train-years', type=int, default=5)
    parser.add_argument('--features', type=int, default=100)
    parser.add_argument('--samples', type=int, default=10000,
                       help='Max training samples (default: 10000, reduce if OOM)')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Prediction chunk size (default: 5000, reduce if OOM)')
    parser.add_argument('--estimators', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multiple GPUs (one year per GPU)')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0, 1],
                       help='GPU IDs to use (default: 0 1)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Timestamp for submission filename (auto-generated if not provided)')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    # 타임스탬프 생성 (전체 실행에서 동일하게 사용)
    from datetime import datetime
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    pred_years = args.year if args.year else [2020, 2021, 2022, 2023, 2024]

    # ============================================
    # 멀티 GPU 모드 (subprocess로 완전 분리)
    # ============================================
    if args.multi_gpu:
        import subprocess

        num_gpus = len(args.gpu_ids)
        log_dir = Path(f"logs/multi_gpu_{timestamp}")
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"MULTI-GPU MODE: {num_gpus} GPUs (subprocess)")
        print(f"GPU IDs: {args.gpu_ids}")
        print(f"Log directory: {log_dir}/")
        print(f"{'='*60}")

        # 연도를 GPU에 분배
        tasks = []
        for i, year in enumerate(pred_years):
            gpu_id = args.gpu_ids[i % num_gpus]
            tasks.append((year, gpu_id))

        print(f"\nTask distribution:")
        for year, gpu_id in tasks:
            print(f"  {year} -> GPU {gpu_id}")

        print(f"\nStarting parallel execution...")

        # subprocess로 병렬 실행
        processes = []
        for year, gpu_id in tasks:
            log_file = log_dir / f"tabpfn_v2_{year}_gpu{gpu_id}.log"
            cmd = [
                "python", "-m", "src.tabpfn_pipeline_v2",
                "--year", str(year),
                "--device", "cuda",
                "--features", str(args.features),
                "--samples", str(args.samples),
                "--chunk-size", str(args.chunk_size),
                "--estimators", str(args.estimators),
                "--train-years", str(args.train_years),
                "--timestamp", timestamp,
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

            with open(log_file, 'w') as f:
                proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=f)
                processes.append((year, gpu_id, proc, log_file))
                print(f"  Started {year} on GPU {gpu_id} (PID: {proc.pid})")

        # 완료 대기
        print(f"\nWaiting for completion...")
        print(f"Monitor: tail -f {log_dir}/*.log")

        results = []
        for year, gpu_id, proc, log_file in processes:
            proc.wait()
            status = "OK" if proc.returncode == 0 else "FAILED"
            results.append((year, gpu_id, status, log_file))
            print(f"  [GPU {gpu_id}] {year}: {status}")

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        for year, gpu_id, status, log_file in results:
            submission_file = f"submissions/{year}.tabpfn_v2.{timestamp}.submission.csv"
            print(f"  {year}: [{status}] {submission_file if status == 'OK' else 'N/A'}")

        print(f"\nLog files: {log_dir}/")
        return

    # ============================================
    # 단일 GPU/CPU 모드
    # ============================================
    pipeline = TabPFNPipelineV2(
        max_features=args.features,
        max_train_samples=args.samples,
        device=args.device,
        n_estimators=args.estimators,
        pred_chunk_size=args.chunk_size,
        timestamp=timestamp
    )

    paths = pipeline.run(
        pred_years=pred_years,
        train_years=args.train_years,
        verbose=not args.quiet
    )

    print("\nSubmission files:")
    for year, path in paths.items():
        print(f"  {year}: {path}")


if __name__ == "__main__":
    main()
