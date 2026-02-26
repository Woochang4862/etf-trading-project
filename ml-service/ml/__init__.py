"""
ML Core Module - Model factory, feature pipeline, and configuration.

This module provides the core ML functionality for the ETF trading project.
It includes model definitions, feature engineering, and data loading utilities.

Main exports:
    - Config classes: DataConfig, FeatureConfig, ModelConfig, TabPFNConfig, AhnLabConfig, Config
    - Model factory: create_model, list_models, get_model_info, get_available_models
    - Base model: BaseRankingModel
    - Feature pipeline: FeaturePipeline, create_feature_panel
    - Data utilities: DataLoader, Preprocessor

Usage:
    from ml import create_model, FeaturePipeline, Config

    # Create a model
    model = create_model('xgboost', params={'max_depth': 8})

    # Create features
    pipeline = FeaturePipeline(data_provider="yfinance")
    panel = pipeline.create_panel(tickers=["AAPL", "MSFT"], ...)

    # Access config
    config = Config()
    print(config.data.train_start_year)
"""

# Config
from .config import (
    DataConfig,
    FeatureConfig,
    ModelConfig,
    TabPFNConfig,
    AhnLabConfig,
    Config,
    config,
    PROJECT_ROOT,
    DATA_DIR,
    SUBMISSIONS_DIR,
)

# Model factory
from .models.factory import (
    create_model,
    list_models,
    get_model_info,
    get_available_models,
    register_model,
)

# Base model
from .models.base_model import BaseRankingModel

# Feature pipeline
from .features.pipeline import FeaturePipeline, create_feature_panel

# Data utilities
from .data.loader import DataLoader
from .data.preprocessor import Preprocessor

__version__ = "2.0.0"

__all__ = [
    # Config
    "DataConfig",
    "FeatureConfig",
    "ModelConfig",
    "TabPFNConfig",
    "AhnLabConfig",
    "Config",
    "config",
    "PROJECT_ROOT",
    "DATA_DIR",
    "SUBMISSIONS_DIR",
    # Model factory
    "create_model",
    "list_models",
    "get_model_info",
    "get_available_models",
    "register_model",
    # Base model
    "BaseRankingModel",
    # Feature pipeline
    "FeaturePipeline",
    "create_feature_panel",
    # Data utilities
    "DataLoader",
    "Preprocessor",
    "__version__",
]
