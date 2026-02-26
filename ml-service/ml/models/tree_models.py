"""
Tree-based models for ETF Stock Prediction

Includes:
- XGBRankingModel: XGBoost gradient boosting
- CatBoostRankingModel: CatBoost gradient boosting
- RandomForestRankingModel: Random Forest regressor
- ExtraTreesRankingModel: Extremely Randomized Trees regressor
"""
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_model import BaseRankingModel


# Check available libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


def _detect_device() -> str:
    """Detect available compute device (CUDA > MPS > CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


class XGBRankingModel(BaseRankingModel):
    """
    XGBoost-based ranking model

    Supports GPU acceleration via tree_method='gpu_hist'
    Grid Search optimized params (2020): depth=8, lr=0.005, n=500
    """

    # Grid Search optimized parameters (2020년 최적)
    DEFAULT_PARAMS = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.005,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'min_child_weight': 3,
        'random_state': 42,
        'n_jobs': -1,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        """
        Initialize XGBoost model

        Args:
            params: Hyperparameters (merged with defaults)
            device: Compute device ('auto', 'cuda', 'cpu')
        """
        super().__init__()

        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        # Set device
        if device == 'auto':
            device = _detect_device()

        if device == 'cuda':
            self.params['tree_method'] = 'gpu_hist'
            self.params['device'] = 'cuda'
        else:
            self.params['tree_method'] = 'hist'

        # FIX: On macOS, default OpenMP implementation often conflicts with Joblib (scikit-learn)
        # leading to segmentation faults when running multiple models sequentially.
        # Defaulting to 1 thread is safer unless explicitly overridden.
        import platform
        if platform.system() == 'Darwin' and self.params.get('n_jobs', -1) == -1:
            self.params['n_jobs'] = 1

        self.device = device
        self.model = None
        self.best_iteration_ = None
        self.early_stopping_rounds = self.params.pop('early_stopping_rounds', 100)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'XGBRankingModel':
        """Train XGBoost model"""
        # Prepare data
        X_train_arr = self._prepare_input(X_train, fit=True)
        y_train_arr = self._prepare_target(y_train)

        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_arr, label=y_train_arr, feature_names=feature_names)

        evals = [(dtrain, 'train')]
        if X_valid is not None and y_valid is not None:
            X_valid_arr = self._prepare_input(X_valid)
            y_valid_arr = self._prepare_target(y_valid)
            dvalid = xgb.DMatrix(X_valid_arr, label=y_valid_arr, feature_names=feature_names)
            evals.append((dvalid, 'valid'))

        # Extract training params
        n_estimators = self.params.pop('n_estimators', 1000)
        n_jobs = self.params.get('n_jobs', 1)
        
        train_params = {k: v for k, v in self.params.items()
                        if k not in ['n_jobs']}
        
        # Explicitly set threads for XGBoost to avoid conflicts
        if n_jobs != -1:
            train_params['nthread'] = n_jobs

        # Train
        self.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_valid is not None else None,
            verbose_eval=100
        )

        # Restore n_estimators to params
        self.params['n_estimators'] = n_estimators

        self.best_iteration_ = self.model.best_iteration
        self._compute_feature_importance(feature_names)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with XGBoost"""
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        X_arr = self._prepare_input(X)
        dtest = xgb.DMatrix(X_arr, feature_names=self.feature_names_)

        return self.model.predict(dtest, iteration_range=(0, self.best_iteration_ + 1))

    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute feature importance"""
        if self.model is None:
            return

        importance = self.model.get_score(importance_type='gain')

        if feature_names is None:
            feature_names = list(importance.keys())

        # Create DataFrame
        imp_list = [(f, importance.get(f, 0)) for f in feature_names]
        self.feature_importance_ = pd.DataFrame(imp_list, columns=['feature', 'importance'])
        self.feature_importance_ = self.feature_importance_.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)

        total = self.feature_importance_['importance'].sum()
        if total > 0:
            self.feature_importance_['importance_pct'] = (
                self.feature_importance_['importance'] / total * 100
            )

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.params.copy()


class CatBoostRankingModel(BaseRankingModel):
    """
    CatBoost-based ranking model

    Supports GPU acceleration via task_type='GPU'
    Grid Search optimized params (2020): depth=4, lr=0.01, n=500
    """

    # Grid Search optimized parameters (2020년 최적)
    DEFAULT_PARAMS = {
        'iterations': 500,
        'depth': 4,
        'learning_rate': 0.01,
        'l2_leaf_reg': 5.0,
        'subsample': 0.8,
        'border_count': 254,
        'random_strength': 0.5,
        'random_seed': 42,
        'verbose': 100,
        'allow_writing_files': False,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        """
        Initialize CatBoost model

        Args:
            params: Hyperparameters
            device: Compute device ('auto', 'cuda', 'cpu')
        """
        super().__init__()

        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        # Set device
        if device == 'auto':
            device = _detect_device()

        if device == 'cuda':
            self.params['task_type'] = 'GPU'
        else:
            self.params['task_type'] = 'CPU'

        self.device = device
        self.model = None
        self.early_stopping_rounds = self.params.pop('early_stopping_rounds', 100)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'CatBoostRankingModel':
        """Train CatBoost model"""
        X_train_arr = self._prepare_input(X_train, fit=True)
        y_train_arr = self._prepare_target(y_train)

        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)

        self.model = CatBoostRegressor(**self.params)

        eval_set = None
        if X_valid is not None and y_valid is not None:
            X_valid_arr = self._prepare_input(X_valid)
            y_valid_arr = self._prepare_target(y_valid)
            eval_set = (X_valid_arr, y_valid_arr)

        self.model.fit(
            X_train_arr, y_train_arr,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            use_best_model=True if eval_set else False
        )

        self._compute_feature_importance(feature_names)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with CatBoost"""
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        X_arr = self._prepare_input(X)
        return self.model.predict(X_arr)

    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute feature importance"""
        if self.model is None:
            return

        importance = self.model.get_feature_importance()

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


class RandomForestRankingModel(BaseRankingModel):
    """
    Random Forest-based ranking model

    Uses sklearn RandomForestRegressor
    Grid Search optimized params (2020): depth=15, max_features=0.1, n=1500
    """

    # Grid Search optimized parameters (2020년 최적)
    DEFAULT_PARAMS = {
        'n_estimators': 1500,
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 0.1,
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': 1,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest model

        Args:
            params: Hyperparameters
        """
        super().__init__()
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'RandomForestRankingModel':
        """Train Random Forest model"""
        X_train_arr = self._prepare_input(X_train, fit=True)
        y_train_arr = self._prepare_target(y_train)

        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train_arr, y_train_arr)

        self._compute_feature_importance(feature_names)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with Random Forest"""
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        X_arr = self._prepare_input(X)
        return self.model.predict(X_arr)

    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute feature importance"""
        if self.model is None:
            return

        importance = self.model.feature_importances_

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


class ExtraTreesRankingModel(BaseRankingModel):
    """
    Extra Trees (Extremely Randomized Trees) ranking model

    Uses sklearn ExtraTreesRegressor - faster than RF with similar performance
    """

    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': 1,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Extra Trees model

        Args:
            params: Hyperparameters
        """
        super().__init__()
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'ExtraTreesRankingModel':
        """Train Extra Trees model"""
        X_train_arr = self._prepare_input(X_train, fit=True)
        y_train_arr = self._prepare_target(y_train)

        if feature_names is None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)

        self.model = ExtraTreesRegressor(**self.params)
        self.model.fit(X_train_arr, y_train_arr)

        self._compute_feature_importance(feature_names)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with Extra Trees"""
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        X_arr = self._prepare_input(X)
        return self.model.predict(X_arr)

    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute feature importance"""
        if self.model is None:
            return

        importance = self.model.feature_importances_

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
