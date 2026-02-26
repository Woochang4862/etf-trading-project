"""
Linear models for ETF Stock Prediction

Includes:
- RidgeRankingModel: Ridge regression (L2 regularization)
- LassoRankingModel: Lasso regression (L1 regularization)
- ElasticNetRankingModel: ElasticNet (L1 + L2 regularization)
- SVRRankingModel: Support Vector Regression

Note: All linear models include StandardScaler for feature normalization
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from .base_model import BaseRankingModel


class RidgeRankingModel(BaseRankingModel):
    """
    Ridge Regression ranking model

    Uses L2 regularization to prevent overfitting.
    Includes StandardScaler for feature normalization.
    """

    DEFAULT_PARAMS = {
        'alpha': 1.0,
        'fit_intercept': True,
        'copy_X': True,
        'max_iter': None,
        'tol': 1e-4,
        'solver': 'auto',
        'random_state': 42,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Ridge model

        Args:
            params: Hyperparameters (alpha is the regularization strength)
        """
        super().__init__()
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.scaler = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'RidgeRankingModel':
        """Train Ridge model with feature scaling"""
        X_train_arr = self._prepare_input(X_train, fit=True)
        y_train_arr = self._prepare_target(y_train)

        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        self.feature_names_ = feature_names

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train_arr)

        # Train model
        self.model = Ridge(**self.params)
        self.model.fit(X_scaled, y_train_arr)

        self._compute_feature_importance(feature_names)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with Ridge"""
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        X_arr = self._prepare_input(X)
        X_scaled = self.scaler.transform(X_arr)

        return self.model.predict(X_scaled)

    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute feature importance from coefficients"""
        if self.model is None:
            return

        importance = np.abs(self.model.coef_)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]

        self.feature_importance_ = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        total = self.feature_importance_['importance'].sum()
        if total > 0:
            self.feature_importance_['importance_pct'] = (
                self.feature_importance_['importance'] / total * 100
            )

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.params.copy()


class LassoRankingModel(BaseRankingModel):
    """
    Lasso Regression ranking model

    Uses L1 regularization for sparse feature selection.
    Includes StandardScaler for feature normalization.
    """

    DEFAULT_PARAMS = {
        'alpha': 0.01,
        'fit_intercept': True,
        'precompute': False,
        'copy_X': True,
        'max_iter': 10000,
        'tol': 1e-4,
        'warm_start': False,
        'positive': False,
        'random_state': 42,
        'selection': 'cyclic',
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Lasso model

        Args:
            params: Hyperparameters (alpha is the regularization strength)
        """
        super().__init__()
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.scaler = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'LassoRankingModel':
        """Train Lasso model with feature scaling"""
        X_train_arr = self._prepare_input(X_train, fit=True)
        y_train_arr = self._prepare_target(y_train)

        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        self.feature_names_ = feature_names

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train_arr)

        # Train model
        self.model = Lasso(**self.params)
        self.model.fit(X_scaled, y_train_arr)

        self._compute_feature_importance(feature_names)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with Lasso"""
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        X_arr = self._prepare_input(X)
        X_scaled = self.scaler.transform(X_arr)

        return self.model.predict(X_scaled)

    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute feature importance from coefficients"""
        if self.model is None:
            return

        importance = np.abs(self.model.coef_)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]

        self.feature_importance_ = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        total = self.feature_importance_['importance'].sum()
        if total > 0:
            self.feature_importance_['importance_pct'] = (
                self.feature_importance_['importance'] / total * 100
            )

    def get_n_selected_features(self) -> int:
        """Get number of non-zero coefficients (selected features)"""
        if self.model is None:
            return 0
        return np.sum(self.model.coef_ != 0)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.params.copy()


class ElasticNetRankingModel(BaseRankingModel):
    """
    ElasticNet Regression ranking model

    Combines L1 and L2 regularization.
    l1_ratio controls the mix (1.0 = Lasso, 0.0 = Ridge).
    Includes StandardScaler for feature normalization.
    """

    DEFAULT_PARAMS = {
        'alpha': 0.01,
        'l1_ratio': 0.5,
        'fit_intercept': True,
        'precompute': False,
        'max_iter': 10000,
        'copy_X': True,
        'tol': 1e-4,
        'warm_start': False,
        'positive': False,
        'random_state': 42,
        'selection': 'cyclic',
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize ElasticNet model

        Args:
            params: Hyperparameters
                - alpha: Regularization strength
                - l1_ratio: L1/L2 mix (0-1)
        """
        super().__init__()
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.scaler = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'ElasticNetRankingModel':
        """Train ElasticNet model with feature scaling"""
        X_train_arr = self._prepare_input(X_train, fit=True)
        y_train_arr = self._prepare_target(y_train)

        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        self.feature_names_ = feature_names

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train_arr)

        # Train model
        self.model = ElasticNet(**self.params)
        self.model.fit(X_scaled, y_train_arr)

        self._compute_feature_importance(feature_names)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with ElasticNet"""
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        X_arr = self._prepare_input(X)
        X_scaled = self.scaler.transform(X_arr)

        return self.model.predict(X_scaled)

    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute feature importance from coefficients"""
        if self.model is None:
            return

        importance = np.abs(self.model.coef_)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]

        self.feature_importance_ = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        total = self.feature_importance_['importance'].sum()
        if total > 0:
            self.feature_importance_['importance_pct'] = (
                self.feature_importance_['importance'] / total * 100
            )

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.params.copy()


class SVRRankingModel(BaseRankingModel):
    """
    Support Vector Regression ranking model

    Uses RBF kernel by default.
    Includes StandardScaler for feature normalization.

    Note: SVR can be slow for large datasets. Consider subsampling.
    """

    DEFAULT_PARAMS = {
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.1,
        'gamma': 'scale',
        'shrinking': True,
        'cache_size': 500,
        'verbose': False,
        'max_iter': -1,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None, max_samples: int = 50000):
        """
        Initialize SVR model

        Args:
            params: Hyperparameters
                - C: Regularization parameter
                - epsilon: Epsilon in epsilon-SVR
                - kernel: Kernel type ('rbf', 'linear', 'poly')
            max_samples: Maximum training samples (SVR is O(n^2))
        """
        super().__init__()
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.max_samples = max_samples
        self.model = None
        self.scaler = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'SVRRankingModel':
        """Train SVR model with feature scaling"""
        X_train_arr = self._prepare_input(X_train, fit=True)
        y_train_arr = self._prepare_target(y_train)

        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        self.feature_names_ = feature_names

        # Subsample if too large (SVR is slow)
        n_samples = len(X_train_arr)
        if n_samples > self.max_samples:
            print(f"SVR: Subsampling from {n_samples} to {self.max_samples} samples")
            np.random.seed(42)
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            X_train_arr = X_train_arr[indices]
            y_train_arr = y_train_arr[indices]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train_arr)

        # Train model
        self.model = SVR(**self.params)
        self.model.fit(X_scaled, y_train_arr)

        # SVR doesn't have feature importance
        self.feature_importance_ = None
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with SVR"""
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        X_arr = self._prepare_input(X)
        X_scaled = self.scaler.transform(X_arr)

        return self.model.predict(X_scaled)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.params.copy()
