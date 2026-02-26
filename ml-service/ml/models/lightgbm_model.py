"""
LightGBM model for ETF Stock Prediction Competition
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

from ..config import config


class ETFRankingModel:
    """
    LightGBM-based model for stock ranking prediction

    Predicts future returns and ranks stocks to select Top-K
    Supports both regression and LambdaRank objectives
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, use_lambdarank: bool = True):
        """
        Initialize model

        Args:
            params: LightGBM parameters (uses config defaults if None)
            use_lambdarank: Use LambdaRank objective for ranking optimization
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        self.default_params = config.model.params.copy()
        self.use_lambdarank = use_lambdarank

        # LambdaRank 파라미터 설정
        if use_lambdarank:
            lambdarank_params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [10, 50, 100],  # Top-100 최적화
                'lambdarank_truncation_level': 100,
                'label_gain': list(range(1, 101)),  # 높은 relevance(99)에 높은 gain(100)
                'learning_rate': 0.05,  # 랭킹에 맞게 조정
                'num_leaves': 63,
            }
            self.params = {**self.default_params, **lambdarank_params, **(params or {})}
            # LambdaRank에서는 metric 변경
            self.params.pop('metric', None)
            self.params['metric'] = 'ndcg'
        else:
            self.params = {**self.default_params, **(params or {})}

        self.model = None
        self.feature_importance_ = None
        self.best_iteration_ = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        train_groups: Optional[List[int]] = None,
        valid_groups: Optional[List[int]] = None
    ) -> 'ETFRankingModel':
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training targets (3-month returns)
            X_valid: Validation features
            y_valid: Validation targets
            feature_names: Feature names for importance tracking
            train_groups: Group sizes for LambdaRank (number of samples per query/date)
            valid_groups: Group sizes for validation set

        Returns:
            self
        """
        # Handle feature names
        if feature_names is None:
            feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None

        # LambdaRank용 레이블 변환 (수익률을 relevance score로)
        if self.use_lambdarank and train_groups is not None:
            # 수익률을 0-100 범위의 relevance score로 변환
            y_train_label = self._convert_to_relevance(y_train, train_groups)
            y_valid_label = self._convert_to_relevance(y_valid, valid_groups) if y_valid is not None else None
        else:
            y_train_label = y_train
            y_valid_label = y_valid

        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train_label,
            feature_name=feature_names,
            group=train_groups if self.use_lambdarank else None
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_valid is not None and y_valid_label is not None:
            valid_data = lgb.Dataset(
                X_valid,
                label=y_valid_label,
                reference=train_data,
                feature_name=feature_names,
                group=valid_groups if self.use_lambdarank else None
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')

        # Training callbacks
        callbacks = [
            lgb.log_evaluation(period=100),
        ]

        if X_valid is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=config.model.early_stopping_rounds,
                    verbose=True
                )
            )

        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=config.model.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        self.best_iteration_ = self.model.best_iteration
        self._compute_feature_importance(feature_names)

        return self

    def _convert_to_relevance(self, y: pd.Series, groups: List[int]) -> np.ndarray:
        """
        Convert continuous returns to discrete relevance scores for LambdaRank

        Args:
            y: Target returns
            groups: Group sizes (samples per date)

        Returns:
            Relevance scores (0-99, higher is better)
        """
        if y is None or groups is None:
            return None

        y_arr = np.array(y)
        result = np.zeros_like(y_arr)

        start = 0
        for size in groups:
            end = start + size
            group_y = y_arr[start:end]

            # 그룹 내에서 퍼센타일 랭크 계산 (0-99)
            ranks = pd.Series(group_y).rank(pct=True, na_option='keep').fillna(0.5).values
            result[start:end] = np.clip((ranks * 99).astype(int), 0, 99)  # 0-99 범위로 제한

            start = end

        return result

    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute and store feature importance"""
        if self.model is None:
            return

        importance = self.model.feature_importance(importance_type='gain')

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]

        self.feature_importance_ = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        # Add percentage
        total = self.feature_importance_['importance'].sum()
        self.feature_importance_['importance_pct'] = (
            self.feature_importance_['importance'] / total * 100
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict future returns

        Args:
            X: Features

        Returns:
            Predicted returns
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X, num_iteration=self.best_iteration_)

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
            tickers: List of ticker symbols (same order as X)
            k: Number of top stocks to select

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
            DataFrame with ticker, prediction, rank
        """
        predictions = self.predict(X)

        result = pd.DataFrame({
            'ticker': tickers,
            'prediction': predictions
        })

        result['rank'] = result['prediction'].rank(ascending=False, method='first').astype(int)
        result = result.sort_values('rank')

        return result

    def get_feature_importance(
        self,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Return only top N features

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        if top_n is not None:
            return self.feature_importance_.head(top_n)
        return self.feature_importance_

    def save(self, path: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(path)

    def load(self, path: str):
        """Load model from file"""
        self.model = lgb.Booster(model_file=path)
        return self


class EnsembleRankingModel:
    """
    Ensemble of multiple models with different seeds

    Note: Check competition rules - this may violate "single model" constraint
    """

    def __init__(
        self,
        n_models: int = 5,
        base_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ensemble

        Args:
            n_models: Number of models in ensemble
            base_params: Base parameters for all models
        """
        self.n_models = n_models
        self.base_params = base_params or config.model.params.copy()
        self.models: List[ETFRankingModel] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> 'EnsembleRankingModel':
        """Train all models in ensemble"""
        self.models = []

        for i in range(self.n_models):
            # Vary random seed and feature fraction
            params = self.base_params.copy()
            params['random_state'] = config.seed + i * 100
            params['feature_fraction'] = 0.7 + 0.05 * i  # Vary slightly

            model = ETFRankingModel(params)
            model.fit(X_train, y_train, X_valid, y_valid)
            self.models.append(model)

            print(f"Trained model {i+1}/{self.n_models}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with ensemble (average of all models)"""
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return predictions.mean(axis=1)

    def predict_top_k(
        self,
        X: pd.DataFrame,
        tickers: List[str],
        k: int = 100
    ) -> Tuple[List[str], np.ndarray]:
        """Predict Top-K stocks using ensemble"""
        predictions = self.predict(X)

        k = min(k, len(predictions))
        top_indices = np.argsort(-predictions)[:k]

        top_tickers = [tickers[i] for i in top_indices]
        top_predictions = predictions[top_indices]

        return top_tickers, top_predictions
