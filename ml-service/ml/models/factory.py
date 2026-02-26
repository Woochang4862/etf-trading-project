"""
Model Factory for creating ranking models by name

Usage:
    from src.models.factory import create_model, list_models

    model = create_model('xgboost', params={'max_depth': 8})
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base_model import BaseRankingModel


# Registry for all available models
# Maps model name -> (model class, default params, description)
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model(
    name: str,
    model_class: Type[BaseRankingModel],
    default_params: Optional[Dict[str, Any]] = None,
    description: str = ""
):
    """
    Register a model class in the registry

    Args:
        name: Model name (lowercase, e.g., 'xgboost')
        model_class: Model class that inherits from BaseRankingModel
        default_params: Default hyperparameters for the model
        description: Brief description of the model
    """
    _MODEL_REGISTRY[name.lower()] = {
        'class': model_class,
        'default_params': default_params or {},
        'description': description
    }


def create_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseRankingModel:
    """
    Create a model instance by name

    Args:
        model_name: Name of the model (case-insensitive)
        params: Hyperparameters to override defaults
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model_name is not registered

    Example:
        >>> model = create_model('xgboost', params={'max_depth': 10})
        >>> model = create_model('ridge', params={'alpha': 0.5})
    """
    name = model_name.lower()

    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )

    registry_entry = _MODEL_REGISTRY[name]
    model_class = registry_entry['class']
    default_params = registry_entry['default_params'].copy()

    # Merge default params with user-provided params
    if params:
        default_params.update(params)

    # Create model instance
    try:
        return model_class(params=default_params, **kwargs)
    except TypeError:
        # Some models may not accept 'params' keyword
        return model_class(**default_params, **kwargs)


def list_models() -> List[Dict[str, Any]]:
    """
    List all registered models

    Returns:
        List of dictionaries with model info:
            - name: Model name
            - description: Model description
            - default_params: Default hyperparameters
    """
    return [
        {
            'name': name,
            'description': info['description'],
            'default_params': info['default_params']
        }
        for name, info in _MODEL_REGISTRY.items()
    ]


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model info
    """
    name = model_name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")

    info = _MODEL_REGISTRY[name]
    return {
        'name': name,
        'class': info['class'].__name__,
        'description': info['description'],
        'default_params': info['default_params']
    }


def get_available_models() -> List[str]:
    """
    Get list of available model names

    Returns:
        List of model names
    """
    return list(_MODEL_REGISTRY.keys())


# Import and register models
# Note: Imports are done here to avoid circular imports
def _register_all_models():
    """Register all available models in the registry"""

    # LightGBM (existing)
    try:
        from .lightgbm_model import ETFRankingModel
        register_model(
            name='lightgbm',
            model_class=ETFRankingModel,
            default_params={'use_lambdarank': False},
            description='LightGBM gradient boosting (regression objective)'
        )
        register_model(
            name='lightgbm_lambdarank',
            model_class=ETFRankingModel,
            default_params={'use_lambdarank': True},
            description='LightGBM with LambdaRank objective for ranking optimization'
        )
    except ImportError:
        pass

    # TabPFN (existing)
    try:
        from .tabpfn_model import TabPFNRankingModel
        register_model(
            name='tabpfn',
            model_class=TabPFNRankingModel,
            default_params={},
            description='TabPFN transformer-based few-shot learning model'
        )
    except ImportError:
        pass

    # Tree-based models
    try:
        from .tree_models import (
            XGBRankingModel,
            CatBoostRankingModel,
            RandomForestRankingModel,
            ExtraTreesRankingModel
        )
        # Grid Search 최적 파라미터 (2020년 기준)
        register_model(
            name='xgboost',
            model_class=XGBRankingModel,
            default_params={
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.005,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2,
                'min_child_weight': 3,
            },
            description='XGBoost gradient boosting (Grid Search optimized)'
        )
        register_model(
            name='catboost',
            model_class=CatBoostRankingModel,
            default_params={
                'iterations': 500,
                'depth': 4,
                'learning_rate': 0.01,
                'l2_leaf_reg': 5.0,
                'subsample': 0.8,
            },
            description='CatBoost gradient boosting (Grid Search optimized)'
        )
        register_model(
            name='random_forest',
            model_class=RandomForestRankingModel,
            default_params={
                'n_estimators': 1500,
                'max_depth': 15,
                'max_features': 0.1,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
            },
            description='Random Forest regressor (Grid Search optimized)'
        )
        register_model(
            name='extra_trees',
            model_class=ExtraTreesRankingModel,
            default_params={
                'n_estimators': 500,
                'max_depth': 15,
            },
            description='Extra Trees (Extremely Randomized Trees) regressor'
        )
    except ImportError:
        pass

    # Linear models
    try:
        from .linear_models import (
            RidgeRankingModel,
            LassoRankingModel,
            ElasticNetRankingModel,
            SVRRankingModel
        )
        register_model(
            name='ridge',
            model_class=RidgeRankingModel,
            default_params={'alpha': 1.0},
            description='Ridge regression (L2 regularization)'
        )
        register_model(
            name='lasso',
            model_class=LassoRankingModel,
            default_params={'alpha': 0.01},
            description='Lasso regression (L1 regularization)'
        )
        register_model(
            name='elasticnet',
            model_class=ElasticNetRankingModel,
            default_params={'alpha': 0.01, 'l1_ratio': 0.5},
            description='ElasticNet regression (L1 + L2 regularization)'
        )
        register_model(
            name='svr',
            model_class=SVRRankingModel,
            default_params={'C': 1.0, 'epsilon': 0.1},
            description='Support Vector Regression'
        )
    except ImportError:
        pass

    # Ensemble model
    try:
        from .ensemble_model import EnsembleRankingModel
        register_model(
            name='ensemble',
            model_class=EnsembleRankingModel,
            default_params={
                'model_names': ['xgboost', 'catboost', 'random_forest'],
                'strategy': 'rank_avg'
            },
            description='Ensemble of multiple models with rank averaging'
        )
    except ImportError:
        pass

    # AhnLab LGBM Ranking Model
    try:
        from .ahnlab_lgbm import AhnLabLGBMRankingModel
        register_model(
            name='ahnlab_lgbm',
            model_class=AhnLabLGBMRankingModel,
            default_params={},
            description='AhnLab LGBM ranking model with pre-trained weights (rank 0.19231)'
        )
    except ImportError:
        pass


# Register models when module is imported
_register_all_models()
