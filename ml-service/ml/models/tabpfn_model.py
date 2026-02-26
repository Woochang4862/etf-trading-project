"""
TabPFN model wrapper for ETF Stock Prediction Competition
Date-by-date training strategy leveraging TabPFN's few-shot learning
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
import gc

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    TabPFNRegressor = None  # Placeholder for type hints
    warnings.warn("TabPFN not installed. Install with: pip install tabpfn")

from ..config import config


class TabPFNRankingModel:
    """
    TabPFN-based model for stock ranking prediction

    Key design: Train a new model for each prediction date using
    recent historical cross-sectional data (few-shot learning approach)
    """

    def __init__(
        self,
        n_estimators: int = 8,
        device: str = "auto",
        ignore_pretraining_limits: bool = True,
        memory_saving_mode: bool = True,
        inference_precision: str = "autocast",
        random_state: int = 42
    ):
        """
        Initialize TabPFN model wrapper

        Args:
            n_estimators: Number of ensemble estimators
            device: Device for inference ("auto", "mps", "cuda", "cpu")
            ignore_pretraining_limits: Allow > 50k samples
            memory_saving_mode: Enable automatic batching
            inference_precision: Precision mode ("autocast" for speed)
            random_state: Random seed
        """
        if not HAS_TABPFN:
            raise ImportError("TabPFN is required. Install with: pip install tabpfn")

        self.n_estimators = n_estimators
        self.device = self._detect_device() if device == "auto" else device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.memory_saving_mode = memory_saving_mode
        self.inference_precision = inference_precision
        self.random_state = random_state

        self.model = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def _detect_device(self) -> str:
        """
        Auto-detect best available device

        Priority: MPS > CUDA > CPU

        Returns:
            Device string for TabPFN
        """
        if not HAS_TORCH:
            return "cpu"

        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
        except AttributeError:
            pass

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _create_model(self):
        """Create a new TabPFN regressor instance"""
        # Use specified device directly (cpu, mps, cuda)
        device = self.device

        model_kwargs = {
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'device': device,
        }

        # Add optional parameters safely
        try:
            if self.ignore_pretraining_limits:
                model_kwargs['ignore_pretraining_limits'] = True
            if self.memory_saving_mode:
                model_kwargs['memory_saving_mode'] = True
        except Exception:
            pass

        return TabPFNRegressor(**model_kwargs)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'TabPFNRankingModel':
        """
        Train the model

        Args:
            X_train: Training features (cross-sectional data)
            y_train: Training targets (3-month forward returns)
            X_valid: Validation features (unused - TabPFN is pretrained)
            y_valid: Validation targets (unused - TabPFN is pretrained)
            feature_names: Feature names for tracking
            **kwargs: Additional parameters (unused)

        Returns:
            self
        """
        # Note: TabPFN is a pretrained model and does not use validation data
        # X_valid and y_valid are accepted for API compatibility but ignored
        # Store feature names
        if feature_names is None:
            if hasattr(X_train, 'columns'):
                self.feature_names_ = list(X_train.columns)
            else:
                self.feature_names_ = [f'feature_{i}' for i in range(X_train.shape[1])]
        else:
            self.feature_names_ = feature_names

        # Convert to numpy if needed
        X = X_train.values if hasattr(X_train, 'values') else X_train
        y = y_train.values if hasattr(y_train, 'values') else y_train

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict future returns

        Args:
            X: Features for prediction

        Returns:
            Predicted returns
        """
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert to numpy if needed
        X_arr = X.values if hasattr(X, 'values') else X

        # Handle NaN values
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return self.model.predict(X_arr)

    def predict_top_k(
        self,
        X: pd.DataFrame,
        tickers: List[str],
        k: int = 100
    ) -> Tuple[List[str], np.ndarray]:
        """
        Predict Top-K stocks

        Args:
            X: Features
            tickers: Ticker symbols (same order as X)
            k: Number of top stocks

        Returns:
            Tuple of (top_k_tickers, top_k_predictions)
        """
        predictions = self.predict(X)

        # Get top-k indices
        k = min(k, len(predictions))
        top_indices = np.argsort(-predictions)[:k]

        top_tickers = [tickers[i] for i in top_indices]
        top_predictions = predictions[top_indices]

        return top_tickers, top_predictions


class DateByDateTrainer:
    """
    Trainer that creates a new TabPFN model for each prediction date

    For each prediction date:
    1. Collect cross-sectional data from past lookback_days
    2. Train TabPFN on this data
    3. Predict on current day's stocks
    4. Select Top-100
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = 'target_3m',
        lookback_days: int = 20,
        min_train_samples: int = 500,
        max_train_samples: int = 10000,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize date-by-date trainer

        Args:
            feature_cols: Feature column names
            target_col: Target column name
            lookback_days: Days of historical data for training
            min_train_samples: Minimum samples required
            max_train_samples: Maximum samples to use (sampling if exceeded)
            model_params: TabPFN model parameters
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback_days = lookback_days
        self.min_train_samples = min_train_samples
        self.max_train_samples = max_train_samples
        self.model_params = model_params or {}

        # Cache
        self._last_prediction = None
        self._date_cache = {}

    def _get_training_dates(
        self,
        panel: pd.DataFrame,
        pred_date: pd.Timestamp,
        target_offset_days: int = 63
    ) -> List[pd.Timestamp]:
        """
        Get training dates for a prediction date

        Training data should be from dates where target is available
        (at least target_offset_days before pred_date)

        Args:
            panel: Full panel data
            pred_date: Prediction date
            target_offset_days: Target horizon in days

        Returns:
            List of valid training dates
        """
        unique_dates = sorted(panel['date'].unique())

        # Find dates before pred_date where target would be available
        # target_3m is computed as close.shift(-63) / close - 1
        # So for date D, target is known at D+63
        # For training, we need dates where target is not NaN

        pred_date_ts = pd.Timestamp(pred_date)

        # Get dates strictly before pred_date
        valid_dates = [d for d in unique_dates if pd.Timestamp(d) < pred_date_ts]

        # Take last lookback_days dates
        if len(valid_dates) > self.lookback_days:
            valid_dates = valid_dates[-self.lookback_days:]

        return valid_dates

    def _prepare_training_data(
        self,
        panel: pd.DataFrame,
        pred_date: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for a specific prediction date

        Args:
            panel: Full panel data
            pred_date: Date to predict

        Returns:
            Tuple of (X_train, y_train)
        """
        train_dates = self._get_training_dates(panel, pred_date)

        if not train_dates:
            raise ValueError(f"No training dates available for {pred_date}")

        # Filter data
        mask = (
            panel['date'].isin(train_dates) &
            panel[self.target_col].notna()
        )
        train_data = panel[mask].copy()

        if len(train_data) < self.min_train_samples:
            raise ValueError(
                f"Insufficient training data: {len(train_data)} < {self.min_train_samples}"
            )

        # Sample if too large
        if len(train_data) > self.max_train_samples:
            train_data = self._sample_data(train_data, self.max_train_samples)

        # Extract features and target
        X = train_data[self.feature_cols].fillna(0)
        y = train_data[self.target_col]

        return X, y

    def _sample_data(
        self,
        data: pd.DataFrame,
        max_samples: int
    ) -> pd.DataFrame:
        """
        Sample data if it exceeds max_samples

        Uses stratified sampling based on target quantiles

        Args:
            data: Data to sample
            max_samples: Maximum samples

        Returns:
            Sampled data
        """
        if len(data) <= max_samples:
            return data

        # Create quantile bins for stratified sampling
        try:
            data = data.copy()
            data['_quantile'] = pd.qcut(
                data[self.target_col],
                q=10,
                labels=False,
                duplicates='drop'
            )

            # Sample proportionally from each quantile
            sampled = data.groupby('_quantile', group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), max(1, int(max_samples * len(x) / len(data)))),
                    random_state=42
                )
            )

            # If still too large, random sample
            if len(sampled) > max_samples:
                sampled = sampled.sample(n=max_samples, random_state=42)

            sampled = sampled.drop(columns=['_quantile'])
            return sampled

        except Exception:
            # Fallback to simple random sampling
            return data.sample(n=max_samples, random_state=42)

    def predict_date(
        self,
        panel: pd.DataFrame,
        pred_date: pd.Timestamp,
        verbose: bool = False
    ) -> Tuple[List[str], np.ndarray]:
        """
        Generate Top-100 prediction for a single date

        Args:
            panel: Full panel data
            pred_date: Date to predict
            verbose: Print progress

        Returns:
            Tuple of (top_100_tickers, predictions)
        """
        try:
            # Prepare training data
            X_train, y_train = self._prepare_training_data(panel, pred_date)

            if verbose:
                print(f"  Training samples: {len(X_train)}")

            # Create and train model
            model = TabPFNRankingModel(**self.model_params)
            model.fit(X_train, y_train)

            # Get prediction day data
            pred_date_ts = pd.Timestamp(pred_date)
            day_data = panel[panel['date'] == pred_date_ts]

            if len(day_data) == 0:
                raise ValueError(f"No data for prediction date {pred_date}")

            X_pred = day_data[self.feature_cols].fillna(0)
            tickers = day_data['ticker'].tolist()

            # Predict and select top-100
            top_tickers, predictions = model.predict_top_k(X_pred, tickers, k=100)

            # Cache for fallback
            self._last_prediction = (top_tickers, predictions)

            # Clean up
            del model
            gc.collect()

            return top_tickers, predictions

        except Exception as e:
            if verbose:
                print(f"  Warning: TabPFN failed for {pred_date}: {e}")

            # Fallback strategies
            return self._fallback_prediction(panel, pred_date)

    def _fallback_prediction(
        self,
        panel: pd.DataFrame,
        pred_date: pd.Timestamp
    ) -> Tuple[List[str], np.ndarray]:
        """
        Fallback prediction when TabPFN fails

        Args:
            panel: Full panel data
            pred_date: Prediction date

        Returns:
            Tuple of (tickers, predictions)
        """
        # Fallback 1: Use previous day's prediction
        if self._last_prediction is not None:
            return self._last_prediction

        # Fallback 2: Simple momentum ranking
        pred_date_ts = pd.Timestamp(pred_date)
        day_data = panel[panel['date'] == pred_date_ts]

        if len(day_data) == 0:
            raise ValueError(f"No data for fallback on {pred_date}")

        # Try different momentum features
        for momentum_col in ['ret_20d', 'ret_10d', 'ret_5d', 'ret_1d']:
            if momentum_col in day_data.columns:
                sorted_df = day_data.nlargest(100, momentum_col)
                return sorted_df['ticker'].tolist(), sorted_df[momentum_col].values

        # Fallback 3: Return universe order
        tickers = day_data['ticker'].head(100).tolist()
        return tickers, np.zeros(len(tickers))
