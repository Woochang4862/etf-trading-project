"""
Ensemble Ranking Model for combining multiple models

Supports various combination strategies:
- prediction_avg: Average raw predictions
- rank_avg: Average ranks (robust to scale differences)
- weighted: Weight by validation performance
- borda: Borda count voting
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .base_model import BaseRankingModel
from .factory import create_model


class EnsembleRankingModel(BaseRankingModel):
    """
    Ensemble of multiple ranking models

    Combines predictions from multiple models using various strategies.
    All models are trained on the same data with shared feature selection.
    """

    STRATEGIES = ['prediction_avg', 'rank_avg', 'weighted', 'borda']

    def __init__(
        self,
        model_names: List[str],
        model_params: Optional[Dict[str, Dict]] = None,
        strategy: str = 'rank_avg',
        weights: Optional[List[float]] = None,
        params: Optional[Dict] = None
    ):
        """
        Initialize ensemble model

        Args:
            model_names: List of model names to ensemble (e.g., ['xgboost', 'catboost'])
            model_params: Per-model hyperparameters {model_name: {param: value}}
            strategy: Combination strategy ('prediction_avg', 'rank_avg', 'weighted', 'borda')
            weights: Manual model weights (for 'weighted' strategy)
            params: Alternative way to pass parameters (for factory compatibility)
        """
        super().__init__()

        # Handle params from factory
        if params is not None:
            model_names = params.get('model_names', model_names)
            model_params = params.get('model_params', model_params)
            strategy = params.get('strategy', strategy)
            weights = params.get('weights', weights)

        if not model_names:
            raise ValueError("model_names must not be empty")

        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        self.model_names = model_names
        self.model_params = model_params or {}
        self.strategy = strategy
        self.manual_weights = weights

        self.models: List[BaseRankingModel] = []
        self.validation_scores: List[float] = []
        self.computed_weights: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs
    ) -> 'EnsembleRankingModel':
        """
        Train all component models

        Args:
            X_train: Training features
            y_train: Training targets
            X_valid: Validation features
            y_valid: Validation targets
            feature_names: Feature names
            verbose: Print progress

        Returns:
            self
        """
        self.models = []
        self.validation_scores = []

        for model_name in self.model_names:
            if verbose:
                print(f"  Training {model_name}...")

            # Get model-specific params
            params = self.model_params.get(model_name, {})

            # Create and train model
            model = create_model(model_name, params)
            model.fit(X_train, y_train, X_valid, y_valid, feature_names)
            self.models.append(model)

            # Compute validation score if available
            if X_valid is not None and y_valid is not None:
                try:
                    val_preds = model.predict(X_valid)
                    corr, _ = spearmanr(val_preds, y_valid)
                    corr = corr if not np.isnan(corr) else 0.0
                    self.validation_scores.append(max(0.0, corr))
                    if verbose:
                        print(f"    Validation correlation: {corr:.4f}")
                except Exception as e:
                    if verbose:
                        print(f"    Validation failed: {e}")
                    self.validation_scores.append(0.5)  # Default weight
            else:
                self.validation_scores.append(1.0)  # Equal weight

        # Compute weights for weighted strategy
        if self.strategy == 'weighted':
            if self.manual_weights is not None:
                self.computed_weights = np.array(self.manual_weights)
            else:
                self.computed_weights = np.array(self.validation_scores)

            # Normalize weights
            weight_sum = self.computed_weights.sum()
            if weight_sum > 0:
                self.computed_weights = self.computed_weights / weight_sum
            else:
                self.computed_weights = np.ones(len(self.models)) / len(self.models)

            if verbose:
                print(f"  Computed weights: {dict(zip(self.model_names, self.computed_weights))}")

        self.is_fitted_ = True
        self.feature_names_ = feature_names

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble strategy

        Args:
            X: Features

        Returns:
            Combined predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model not trained. Call fit() first.")

        if self.strategy == 'prediction_avg':
            return self._predict_avg(X)
        elif self.strategy == 'rank_avg':
            return self._predict_rank_avg(X)
        elif self.strategy == 'weighted':
            return self._predict_weighted(X)
        elif self.strategy == 'borda':
            return self._predict_borda(X)
        else:
            return self._predict_avg(X)

    def _predict_avg(self, X: pd.DataFrame) -> np.ndarray:
        """Simple prediction averaging"""
        all_preds = []
        for model in self.models:
            preds = model.predict(X)
            all_preds.append(preds)

        return np.mean(all_preds, axis=0)

    def _predict_rank_avg(self, X: pd.DataFrame) -> np.ndarray:
        """
        Average ranks (return negative for sorting compatibility)

        Lower rank = better prediction, so we return negative average rank
        to maintain "higher = better" convention for top-k selection.
        """
        all_ranks = []
        for model in self.models:
            preds = model.predict(X)
            ranks = rankdata(-preds)  # 1 = highest prediction
            all_ranks.append(ranks)

        avg_ranks = np.mean(all_ranks, axis=0)
        return -avg_ranks  # Negative so higher = better for top-k

    def _predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted prediction by validation performance"""
        if self.computed_weights is None:
            return self._predict_avg(X)

        weighted_preds = np.zeros(len(X))
        for i, model in enumerate(self.models):
            preds = model.predict(X)
            weighted_preds += self.computed_weights[i] * preds

        return weighted_preds

    def _predict_borda(self, X: pd.DataFrame) -> np.ndarray:
        """Borda count voting (rank-based scoring)"""
        n = len(X)
        scores = np.zeros(n)

        for model in self.models:
            preds = model.predict(X)
            ranks = rankdata(-preds)  # 1 = highest prediction
            borda_scores = n - ranks + 1  # Higher = better
            scores += borda_scores

        return scores

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters"""
        return {
            'model_names': self.model_names,
            'model_params': self.model_params,
            'strategy': self.strategy,
            'weights': self.manual_weights,
            'validation_scores': self.validation_scores,
            'computed_weights': self.computed_weights.tolist() if self.computed_weights is not None else None
        }

    def get_feature_importance(
        self,
        top_n: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get averaged feature importance across all models

        Only includes models that support feature importance.
        """
        if not self.is_fitted_:
            return None

        importance_dfs = []
        for i, model in enumerate(self.models):
            imp = model.get_feature_importance()
            if imp is not None:
                imp = imp.copy()
                imp['model'] = self.model_names[i]
                importance_dfs.append(imp)

        if not importance_dfs:
            return None

        # Combine and average
        combined = pd.concat(importance_dfs, ignore_index=True)
        avg_importance = combined.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)

        total = avg_importance['importance'].sum()
        if total > 0:
            avg_importance['importance_pct'] = avg_importance['importance'] / total * 100
        else:
            avg_importance['importance_pct'] = 0

        if top_n is not None:
            return avg_importance.head(top_n)
        return avg_importance

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted_ else "not fitted"
        models_str = ", ".join(self.model_names)
        return f"EnsembleRankingModel([{models_str}], strategy='{self.strategy}', {fitted})"
