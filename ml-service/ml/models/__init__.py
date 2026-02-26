"""
ML model modules

Available models:
- ETFRankingModel: LightGBM-based model
- TabPFNRankingModel: TabPFN transformer model
- XGBRankingModel: XGBoost model
- CatBoostRankingModel: CatBoost model
- RandomForestRankingModel: Random Forest model
- ExtraTreesRankingModel: Extra Trees model
- RidgeRankingModel: Ridge regression
- LassoRankingModel: Lasso regression
- ElasticNetRankingModel: ElasticNet regression
- SVRRankingModel: Support Vector Regression

Use factory.create_model() for easy model creation.
"""
from .base_model import BaseRankingModel
from .lightgbm_model import ETFRankingModel
from .trainer import WalkForwardTrainer
from .factory import create_model, list_models, get_available_models

# Tree-based models
try:
    from .tree_models import (
        XGBRankingModel,
        CatBoostRankingModel,
        RandomForestRankingModel,
        ExtraTreesRankingModel
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
except ImportError:
    pass

# TabPFN
try:
    from .tabpfn_model import TabPFNRankingModel
except ImportError:
    pass

__all__ = [
    "BaseRankingModel",
    "ETFRankingModel",
    "WalkForwardTrainer",
    "create_model",
    "list_models",
    "get_available_models",
    # Tree models
    "XGBRankingModel",
    "CatBoostRankingModel",
    "RandomForestRankingModel",
    "ExtraTreesRankingModel",
    # Linear models
    "RidgeRankingModel",
    "LassoRankingModel",
    "ElasticNetRankingModel",
    "SVRRankingModel",
    # TabPFN
    "TabPFNRankingModel",
]
