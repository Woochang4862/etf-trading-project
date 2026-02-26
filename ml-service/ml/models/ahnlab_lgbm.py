"""
AhnLab LightGBM LambdaRank Model for ETF Stock Ranking

Reproduces the training pipeline from AhnLab_LGBM_rank_0.19231/train.py
with rolling 2-fold cross-validation and ensemble averaging.

Supports GPU acceleration via device='gpu' parameter.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")


def _detect_gpu_available() -> bool:
    """Check if LightGBM GPU is available."""
    try:
        # Try to create a small GPU dataset to verify GPU support
        import lightgbm as lgb
        # Check if lightgbm was built with GPU support
        # LightGBM doesn't have a direct way to check, so we check CUDA
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # No torch, try subprocess to check nvidia-smi
            import subprocess
            result = subprocess.run(
                ['nvidia-smi'], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
    except Exception:
        return False

from .base_model import BaseRankingModel
from ..features.ahnlab.constants import (
    LGB_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    TOP_K,
    MIN_HISTORY_DAYS,
    RELEVANCE_BINS,
    SEED,
    ALL_FEATURE_COLS,
)


def get_rolling_folds(pred_year: int) -> List[Dict[str, pd.Timestamp]]:
    """
    Get rolling 2-fold cross-validation configuration for a prediction year.

    Fold 1: validation on pred_year - 3
    Fold 2: validation on pred_year - 1

    Args:
        pred_year: Target prediction year

    Returns:
        List of fold configurations with train_end, valid_start, valid_end
    """
    valid_year_1 = pred_year - 3
    valid_year_2 = pred_year - 1
    return [
        {
            "train_end": pd.Timestamp(f"{valid_year_1-1}-12-31"),
            "valid_start": pd.Timestamp(f"{valid_year_1}-01-01"),
            "valid_end": pd.Timestamp(f"{valid_year_1}-12-31"),
        },
        {
            "train_end": pd.Timestamp(f"{valid_year_2-1}-12-31"),
            "valid_start": pd.Timestamp(f"{valid_year_2}-01-01"),
            "valid_end": pd.Timestamp(f"{valid_year_2}-12-31"),
        },
    ]


def shift_features_for_prediction(panel: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Shift all features by 1 day to avoid future data leakage.

    When predicting day t, we should only use features from day t-1.
    This ensures compliance with the competition rule:
    '예측 누수 금지: 2020-03-05 예측 시 2020-03-04까지의 데이터만 사용'

    Args:
        panel: Panel DataFrame with ticker and date columns
        feature_cols: List of feature column names to shift

    Returns:
        Panel with shifted features
    """
    panel = panel.copy()
    grouped = panel.groupby("ticker")

    for col in feature_cols:
        if col in panel.columns:
            panel[col] = grouped[col].shift(1)

    return panel


def add_relevance_labels(df: pd.DataFrame, relevance_bins: int = RELEVANCE_BINS) -> pd.DataFrame:
    """
    Add relevance labels for LambdaRank training.

    Converts continuous target (future returns) to discrete relevance scores
    using quantile binning within each date group.

    Args:
        df: DataFrame with target_3m and date columns
        relevance_bins: Number of relevance bins (default 50)

    Returns:
        DataFrame with relevance column added
    """
    df = df.copy()

    def _label(series: pd.Series) -> pd.Series:
        q = min(relevance_bins, series.shape[0])
        if q <= 1:
            return pd.Series(0, index=series.index, dtype=int)
        ranks = series.rank(method="first")
        labels = pd.qcut(
            ranks,
            q=q,
            labels=False,
            duplicates="drop",
        )
        return labels.fillna(0).astype(int)

    df["relevance"] = df.groupby("date")["target_3m"].transform(_label)
    df["relevance"] = df["relevance"].fillna(0).astype(int)
    return df


class AhnLabLGBMRankingModel(BaseRankingModel):
    """
    LightGBM LambdaRank model reproducing AhnLab pipeline.

    Key features:
    - LambdaRank objective with custom label_gain
    - Rolling 2-fold cross-validation for ensemble
    - Feature shifting to prevent data leakage
    - Relevance label binning for ranking

    Usage:
        # Simple training (single model)
        model = AhnLabLGBMRankingModel()
        model.fit(X_train, y_train, X_valid, y_valid, train_groups, valid_groups)
        predictions = model.predict(X_test)

        # Full panel-based training (reproduces AhnLab pipeline)
        model = AhnLabLGBMRankingModel()
        model.fit_with_panel(panel, pred_year=2024, train_start="2010-01-01")
        predictions = model.predict(X_test)
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_estimators: int = NUM_BOOST_ROUND,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
        seed: int = SEED,
        feature_cols: Optional[List[str]] = None,
        device: Literal["auto", "gpu", "cpu"] = "auto",
    ):
        """
        Initialize AhnLab LightGBM Ranking Model.

        Args:
            params: LightGBM parameters (uses LGB_PARAMS from constants if None)
            n_estimators: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
            seed: Random seed for reproducibility
            feature_cols: Feature column names (uses ALL_FEATURE_COLS if None)
            device: Compute device ('auto', 'gpu', 'cpu'). 'auto' detects GPU availability.
        """
        super().__init__()

        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        self.params = {**LGB_PARAMS, **(params or {})}
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.feature_cols = feature_cols or ALL_FEATURE_COLS

        # Set device (GPU/CPU)
        if device == "auto":
            self.device = "gpu" if _detect_gpu_available() else "cpu"
        else:
            self.device = device

        # Configure GPU parameters
        if self.device == "gpu":
            self.params["device"] = "gpu"
            self.params["gpu_platform_id"] = 0
            self.params["gpu_device_id"] = 0
            print(f"[AhnLabLGBM] GPU mode enabled")
        else:
            self.params["device"] = "cpu"
            print(f"[AhnLabLGBM] CPU mode")

        # Ensemble of models from rolling CV
        self.models_: List[lgb.LGBMRanker] = []
        self.evals_results_: List[Dict] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        train_groups: Optional[List[int]] = None,
        valid_groups: Optional[List[int]] = None,
        **kwargs
    ) -> "AhnLabLGBMRankingModel":
        """
        Train a single LambdaRank model.

        For full reproduction of AhnLab pipeline with rolling CV,
        use fit_with_panel() instead.

        Args:
            X_train: Training features
            y_train: Training relevance labels (0 to RELEVANCE_BINS-1)
            X_valid: Validation features
            y_valid: Validation relevance labels
            feature_names: Feature names for importance tracking
            train_groups: Group sizes for training (samples per date)
            valid_groups: Group sizes for validation
            **kwargs: Additional arguments (ignored)

        Returns:
            self: Trained model instance
        """
        if feature_names is None:
            feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

        # Clean input data
        X_train_clean = self._clean_features(X_train)
        X_valid_clean = self._clean_features(X_valid) if X_valid is not None else None

        evals_result: Dict = {}
        model = lgb.LGBMRanker(
            **self.params,
            n_estimators=self.n_estimators,
            random_state=self.seed,
        )

        eval_set = [(X_train_clean, y_train)]
        eval_group = [train_groups] if train_groups else None
        eval_names = ["train"]

        if X_valid_clean is not None and y_valid is not None:
            eval_set.append((X_valid_clean, y_valid))
            eval_names.append("valid")
            if valid_groups:
                eval_group.append(valid_groups)

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ]

        model.fit(
            X_train_clean,
            y_train,
            group=train_groups,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_names=eval_names,
            eval_at=[TOP_K],
            callbacks=callbacks,
        )

        self.models_ = [model]
        self.evals_results_ = [evals_result]
        self.is_fitted_ = True
        self.feature_names_ = feature_names
        self._compute_feature_importance()

        return self

    def fit_with_panel(
        self,
        panel: pd.DataFrame,
        pred_year: int,
        train_start: str = "2010-01-01",
        target_col: str = "target_3m",
        apply_feature_shift: bool = True,
    ) -> "AhnLabLGBMRankingModel":
        """
        Train ensemble of models using rolling 2-fold cross-validation.

        This method reproduces the full AhnLab training pipeline:
        1. Shift features by 1 day (if apply_feature_shift=True)
        2. For each fold in rolling CV:
           - Prepare train/valid windows
           - Add relevance labels
           - Train LGBMRanker
        3. Store ensemble of models for prediction averaging

        Args:
            panel: Full panel DataFrame with columns:
                   - ticker: Stock symbol
                   - date: Trading date
                   - target_3m: 3-month future return
                   - All feature columns
            pred_year: Target prediction year (e.g., 2024)
            train_start: Start date for training data
            target_col: Name of target column
            apply_feature_shift: Whether to shift features by 1 day

        Returns:
            self: Trained model with ensemble
        """
        train_start_ts = pd.Timestamp(train_start)
        train_end = pd.Timestamp(f"{pred_year - 1}-12-31")

        # Shift features to prevent leakage
        if apply_feature_shift:
            panel = shift_features_for_prediction(panel, self.feature_cols)

        # Get rolling fold configuration
        rolling_folds = get_rolling_folds(pred_year)

        self.models_ = []
        self.evals_results_ = []

        for idx, fold in enumerate(rolling_folds, start=1):
            # Prepare train window
            train_df = self._prepare_window(
                panel,
                start=train_start_ts,
                end=fold["train_end"],
                train_end=train_end,
                target_col=target_col,
            )

            # Prepare validation window
            valid_df = self._prepare_window(
                panel,
                start=fold["valid_start"],
                end=fold["valid_end"],
                train_end=train_end,
                target_col=target_col,
            )

            if train_df.empty or valid_df.empty:
                print(
                    f"Skipping fold {idx}: train rows {len(train_df)}, "
                    f"valid rows {len(valid_df)}"
                )
                continue

            print(
                f"Training fold {idx}: train_end {fold['train_end'].date()}, "
                f"valid {fold['valid_start'].date()} ~ {fold['valid_end'].date()}"
            )

            # Train single model for this fold
            model, evals_result = self._train_single_ranker(
                train_df, valid_df, seed=self.seed
            )

            self.models_.append(model)
            self.evals_results_.append(evals_result)

            best_iter = getattr(model, "best_iteration_", None)
            print(
                f"Fold {idx} done: train rows {len(train_df)}, "
                f"valid rows {len(valid_df)}, best_iter {best_iter}"
            )

        if not self.models_:
            raise RuntimeError(
                "No models were trained; check fold definitions or data availability"
            )

        self.is_fitted_ = True
        self.feature_names_ = self.feature_cols
        self._compute_feature_importance()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble averaging.

        Args:
            X: Features DataFrame

        Returns:
            Averaged predictions from all models in ensemble
        """
        if not self.is_fitted_ or not self.models_:
            raise ValueError("Model not trained. Call fit() or fit_with_panel() first.")

        X_clean = self._clean_features(X)

        # Ensemble prediction: average across all fold models
        preds = np.zeros(len(X_clean), dtype=float)
        for model in self.models_:
            num_iter = getattr(model, "best_iteration_", None)
            preds += model.predict(X_clean, num_iteration=num_iter)
        preds /= len(self.models_)

        return preds

    def predict_daily_topk(
        self,
        panel: pd.DataFrame,
        tickers: List[str],
        pred_start: pd.Timestamp,
        pred_end: pd.Timestamp,
        k: int = TOP_K,
    ) -> pd.DataFrame:
        """
        Predict daily Top-K stocks for a date range.

        Args:
            panel: Panel DataFrame with features
            tickers: List of valid tickers to consider
            pred_start: Start date for predictions
            pred_end: End date for predictions
            k: Number of top stocks per day

        Returns:
            DataFrame with columns: date, rank, ticker
        """
        pred_mask = (panel["date"] >= pred_start) & (panel["date"] <= pred_end)
        pred_df = panel.loc[pred_mask & panel["ticker"].isin(set(tickers))].copy()

        if pred_df.empty:
            raise RuntimeError("No rows found in prediction window")

        # Clean features
        feature_cols = [c for c in self.feature_cols if c in pred_df.columns]
        pred_df[feature_cols] = pred_df[feature_cols].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        pred_df.sort_values(["date", "ticker"], inplace=True)

        # Get predictions
        X_pred = pred_df[feature_cols]
        preds = self.predict(X_pred)
        pred_df = pred_df.assign(pred=preds)

        # Select Top-K per day
        outputs = []
        for date, group in pred_df.groupby("date"):
            top = group.nlargest(k, "pred")
            outputs.append(
                pd.DataFrame({
                    "date": date.strftime("%Y-%m-%d"),
                    "rank": np.arange(1, len(top) + 1, dtype=int),
                    "ticker": top["ticker"].astype(str).values,
                })
            )

        return pd.concat(outputs, ignore_index=True)

    def predict_top_k_for_year(
        self,
        panel: pd.DataFrame,
        year: int,
        k: int = TOP_K,
        universe_dir: str = "data",
    ) -> pd.DataFrame:
        """
        Predict daily Top-K stocks for a given year.

        Wrapper method that loads universe tickers and calculates date range.

        Args:
            panel: Panel DataFrame with features
            year: Prediction year
            k: Number of top stocks per day
            universe_dir: Directory containing universe CSV files

        Returns:
            DataFrame with columns: date, rank, ticker
        """
        from pathlib import Path

        # Load tickers from universe file
        universe_file = Path(universe_dir) / f"{year}_final_universe.csv"
        if not universe_file.exists():
            raise FileNotFoundError(f"Universe file not found: {universe_file}")

        ticker_df = pd.read_csv(universe_file)
        tickers = ticker_df["ticker"].str.strip().tolist()

        # Calculate prediction date range for the year
        pred_start = pd.Timestamp(f"{year}-01-01")
        pred_end = pd.Timestamp(f"{year}-12-31")

        return self.predict_daily_topk(panel, tickers, pred_start, pred_end, k=k)

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            "params": self.params,
            "n_estimators": self.n_estimators,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "device": self.device,
            "n_models": len(self.models_),
        }

    def _prepare_window(
        self,
        panel: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        train_end: pd.Timestamp,
        target_col: str = "target_3m",
    ) -> pd.DataFrame:
        """
        Prepare a data window for training or validation.

        Args:
            panel: Full panel DataFrame
            start: Window start date
            end: Window end date
            train_end: Latest allowed target date
            target_col: Target column name

        Returns:
            Prepared DataFrame with relevance labels
        """
        # Filter by date range and valid targets
        target_date_col = "target_date" if "target_date" in panel.columns else None

        mask = (panel["date"] >= start) & (panel["date"] <= end)

        if target_date_col and target_date_col in panel.columns:
            mask &= panel[target_date_col].notna()
            mask &= panel[target_date_col] <= train_end

        df = panel.loc[mask].copy()

        # Filter by minimum history
        if "days_since_start" in df.columns:
            df = df[df["days_since_start"] >= MIN_HISTORY_DAYS]

        # Clean data
        feature_cols = [c for c in self.feature_cols if c in df.columns]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=[target_col], inplace=True)
        df[feature_cols] = df[feature_cols].fillna(0)

        # Add relevance labels
        df = add_relevance_labels(df)

        return df

    def _train_single_ranker(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        seed: int,
    ) -> Tuple[lgb.LGBMRanker, Dict]:
        """
        Train a single LGBMRanker model.

        Args:
            train_df: Training DataFrame with relevance column
            valid_df: Validation DataFrame with relevance column
            seed: Random seed

        Returns:
            Tuple of (trained model, evaluation results)
        """
        # Prepare model inputs
        feature_cols = [c for c in self.feature_cols if c in train_df.columns]

        train_df_sorted = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)
        valid_df_sorted = valid_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        X_train = train_df_sorted[feature_cols]
        y_train = train_df_sorted["relevance"]
        train_groups = train_df_sorted.groupby("date").size().astype(int).tolist()

        X_valid = valid_df_sorted[feature_cols]
        y_valid = valid_df_sorted["relevance"]
        valid_groups = valid_df_sorted.groupby("date").size().astype(int).tolist()

        evals_result: Dict = {}
        model = lgb.LGBMRanker(
            **self.params,
            n_estimators=self.n_estimators,
            random_state=seed,
        )

        model.fit(
            X_train,
            y_train,
            group=train_groups,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_group=[train_groups, valid_groups],
            eval_names=["train", "valid"],
            eval_at=[TOP_K],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(period=50),
                lgb.record_evaluation(evals_result),
            ],
        )

        return model, evals_result

    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean feature values (handle inf, nan)."""
        if X is None:
            return None
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X_clean

    def _compute_feature_importance(self) -> None:
        """Compute aggregated feature importance from ensemble."""
        if not self.models_:
            return

        # Average importance across all models
        importance_sum = None
        feature_names = None

        for model in self.models_:
            booster = model.booster_
            importance = booster.feature_importance(importance_type="gain")
            names = booster.feature_name()

            if importance_sum is None:
                importance_sum = importance.astype(float)
                feature_names = names
            else:
                importance_sum += importance

        importance_avg = importance_sum / len(self.models_)

        self.feature_importance_ = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_avg,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        total = self.feature_importance_["importance"].sum()
        if total > 0:
            self.feature_importance_["importance_pct"] = (
                self.feature_importance_["importance"] / total * 100
            )
        else:
            self.feature_importance_["importance_pct"] = 0.0

    def save(self, path: str) -> None:
        """
        Save ensemble models to files.

        Creates files: {path}_fold{i}.txt for each model
        """
        if not self.models_:
            raise ValueError("No models to save")

        for i, model in enumerate(self.models_):
            model_path = f"{path}_fold{i}.txt"
            model.booster_.save_model(model_path)
            print(f"Saved model to {model_path}")

    def load(self, path: str, n_folds: int = 2) -> "AhnLabLGBMRankingModel":
        """
        Load ensemble models from files.

        Args:
            path: Base path (expects {path}_fold{i}.txt files)
            n_folds: Number of fold models to load

        Returns:
            self with loaded models
        """
        self.models_ = []

        for i in range(n_folds):
            model_path = f"{path}_fold{i}.txt"
            booster = lgb.Booster(model_file=model_path)

            # Wrap booster in LGBMRanker-like object
            model = _BoosterWrapper(booster)
            self.models_.append(model)
            print(f"Loaded model from {model_path}")

        self.is_fitted_ = True
        return self


class _BoosterWrapper:
    """Wrapper to make lgb.Booster behave like LGBMRanker for prediction."""

    def __init__(self, booster: lgb.Booster):
        self.booster_ = booster
        self.best_iteration_ = None

    def predict(self, X: pd.DataFrame, num_iteration: Optional[int] = None) -> np.ndarray:
        return self.booster_.predict(X, num_iteration=num_iteration)
