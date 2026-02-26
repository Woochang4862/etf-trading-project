"""
Abstract base class for all ranking models

All models must implement this interface for compatibility with the experiment pipeline.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BaseRankingModel(ABC):
    """
    Abstract base class for stock ranking prediction models

    All ranking models must inherit from this class and implement
    the required methods: fit() and predict().

    The predict_top_k() method has a default implementation that can be overridden.
    """

    def __init__(self):
        """Initialize base model attributes"""
        self.is_fitted_ = False
        self.feature_names_: Optional[List[str]] = None
        self.feature_importance_: Optional[pd.DataFrame] = None

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'BaseRankingModel':
        """
        Train the model

        Args:
            X_train: Training features (N_samples, N_features)
            y_train: Training targets - 3-month future returns (N_samples,)
            X_valid: Validation features (optional)
            y_valid: Validation targets (optional)
            feature_names: Feature names for importance tracking
            **kwargs: Additional model-specific parameters

        Returns:
            self: Trained model instance
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict future returns/scores

        Args:
            X: Features (N_samples, N_features)

        Returns:
            Predictions (N_samples,) - higher values indicate better expected returns
        """
        pass

    def predict_top_k(
        self,
        X: pd.DataFrame,
        tickers: List[str],
        k: int = 100
    ) -> Tuple[List[str], np.ndarray]:
        """
        Predict Top-K stocks with highest expected returns

        Args:
            X: Features (N_samples, N_features)
            tickers: List of ticker symbols (same order as X rows)
            k: Number of top stocks to select

        Returns:
            Tuple of:
                - top_k_tickers: List[str] of length k
                - top_k_predictions: np.ndarray of shape (k,)
        """
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        predictions = self.predict(X)

        # Handle k larger than available samples
        k = min(k, len(predictions))

        # Get indices of top-k predictions (highest first)
        top_indices = np.argsort(-predictions)[:k]

        top_tickers = [tickers[i] for i in top_indices]
        top_predictions = predictions[top_indices]

        return top_tickers, top_predictions

    def predict_with_rank(
        self,
        X: pd.DataFrame,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Predict with ranking information

        Args:
            X: Features
            tickers: List of ticker symbols

        Returns:
            DataFrame with columns: ticker, prediction, rank
        """
        predictions = self.predict(X)

        result = pd.DataFrame({
            'ticker': tickers,
            'prediction': predictions
        })

        result['rank'] = result['prediction'].rank(ascending=False, method='first').astype(int)
        result = result.sort_values('rank')

        return result

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters

        Returns:
            Dictionary of parameter names and values
        """
        pass

    def get_feature_importance(
        self,
        top_n: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if available)

        Args:
            top_n: Return only top N features

        Returns:
            DataFrame with columns: feature, importance, importance_pct
            Returns None if model doesn't support feature importance
        """
        if self.feature_importance_ is None:
            return None

        if top_n is not None:
            return self.feature_importance_.head(top_n)
        return self.feature_importance_

    def _prepare_input(
        self,
        X: pd.DataFrame,
        fit: bool = False
    ) -> np.ndarray:
        """
        Prepare input data for model

        Args:
            X: Input features DataFrame
            fit: Whether this is for fitting (True) or prediction (False)

        Returns:
            Cleaned numpy array
        """
        # Convert to numpy if needed
        X_arr = X.values if hasattr(X, 'values') else np.array(X)

        # Handle NaN, inf values
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Store feature names during fit
        if fit and hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)

        return X_arr

    def _prepare_target(self, y: pd.Series) -> np.ndarray:
        """
        Prepare target values

        Args:
            y: Target Series

        Returns:
            Cleaned numpy array
        """
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return y_arr

    def __repr__(self) -> str:
        """String representation of model"""
        class_name = self.__class__.__name__
        fitted = "fitted" if self.is_fitted_ else "not fitted"
        return f"{class_name}({fitted})"
