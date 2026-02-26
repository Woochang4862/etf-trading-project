"""
ETF Stock Prediction Competition - Configuration
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os

# Project Paths
# When in ml-service/ml/, project root is ml-service/
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = DATA_DIR / "models"  # For trained model storage

# Ensure directories exist
SUBMISSIONS_DIR.mkdir(exist_ok=True)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    # Training data period
    train_start_year: int = 2010
    train_years: int = 10  # Rolling window size

    # Prediction years
    pred_years: List[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023, 2024])

    # Universe files
    universe_pattern: str = "{year}_final_universe.csv"

    # Target definition
    target_horizon: int = 63  # 3 months ~ 63 trading days

    # Data quality thresholds
    min_history_days: int = 300  # Minimum history required
    max_daily_return: float = 5.0  # 500% daily return = anomaly
    max_price_ratio: float = 1000  # Max/min price ratio threshold


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # RSI periods
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Moving average periods
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 63])
    ema_periods: List[int] = field(default_factory=lambda: [20, 63])

    # Return periods
    return_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 40, 63, 126, 252])

    # Volatility periods
    volatility_periods: List[int] = field(default_factory=lambda: [10, 20, 63])

    # ROC periods
    roc_periods: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14

    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    # ADX
    adx_period: int = 14

    # Aroon
    aroon_period: int = 25

    # Volume MA
    volume_ma_periods: List[int] = field(default_factory=lambda: [10, 20])

    # MFI
    mfi_period: int = 14

    # Cross-sectional features to rank
    rank_features: List[str] = field(default_factory=lambda: [
        'ret_1d', 'ret_5d', 'ret_20d', 'ret_63d',
        'rsi_14', 'volume_ratio',
        'price_to_sma_20', 'bb_position'
    ])


@dataclass
class ModelConfig:
    """LightGBM model configuration - optimized for ranking"""
    # LightGBM parameters (최적화됨)
    params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'num_leaves': 127,           # 증가: 더 복잡한 패턴 학습
        'max_depth': 8,              # 추가: 과적합 방지
        'learning_rate': 0.02,       # 감소: 더 세밀한 학습
        'feature_fraction': 0.7,     # 감소: 다양성 증가
        'bagging_fraction': 0.7,     # 감소: 다양성 증가
        'bagging_freq': 3,           # 감소: 더 자주 배깅
        'min_child_samples': 50,     # 증가: 과적합 방지
        'min_child_weight': 0.001,   # 추가: 리프 최소 가중치
        'reg_alpha': 0.3,            # 증가: L1 정규화 강화
        'reg_lambda': 0.3,           # 증가: L2 정규화 강화
        'max_bin': 255,              # 추가: 히스토그램 빈 수
        'min_gain_to_split': 0.01,   # 추가: 분할 최소 이득
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    })

    # Training parameters
    n_estimators: int = 2000         # 증가: 더 많은 트리
    early_stopping_rounds: int = 100  # 증가: 더 인내심 있게

    # Prediction
    top_k: int = 100  # Top-100 stocks to select


@dataclass
class TabPFNConfig:
    """TabPFN model configuration for date-by-date training"""
    # Training data configuration
    lookback_days: int = 20           # Days of cross-sectional data for training
    min_train_samples: int = 500      # Minimum samples required for training
    max_train_samples: int = 10000    # Maximum samples (TabPFN limit consideration)

    # Model parameters
    n_estimators: int = 8             # Number of ensemble estimators
    ignore_pretraining_limits: bool = True  # Allow > 50k samples
    memory_saving_mode: bool = True   # Enable automatic batching
    inference_precision: str = "autocast"  # Fast inference

    # Feature selection
    max_features: int = 100           # TabPFN works best with fewer features
    feature_selection_method: str = "importance"  # "importance" or "correlation"

    # Device configuration
    device: str = "auto"              # "auto", "mps", "cuda", "cpu"

    # Prediction
    top_k: int = 100                  # Top-100 stocks to select

    # Caching
    cache_features: bool = True       # Cache computed features
    cache_dir: str = "cache/tabpfn"


@dataclass
class AhnLabConfig:
    """AhnLab LGBM LambdaRank configuration for ranking task"""
    # API configuration
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))

    # Data provider
    data_provider: str = "yfinance_fred"

    # Training period
    train_start: str = "2010-01-01"

    # Target definition
    target_horizon: int = 63  # 3 months ~ 63 trading days

    # Model selection
    top_k: int = 100  # Top-100 stocks to select

    # Data quality
    validation_days: int = 90
    min_history_days: int = 126  # Minimum 6 months

    # Feature engineering
    relevance_bins: int = 50  # Binning for relevance features

    # LightGBM LambdaRank parameters
    num_boost_round: int = 5000
    early_stopping_rounds: int = 150
    n_folds: int = 2  # Cross-validation folds

    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'lambdarank',
        'boosting_type': 'gbdt',
        'metric': 'ndcg',
        'ndcg_eval_at': [100],
        'num_leaves': 127,
        'max_depth': 8,
        'learning_rate': 0.02,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 3,
        'min_child_samples': 50,
        'min_child_weight': 0.001,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'max_bin': 255,
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    })


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tabpfn: TabPFNConfig = field(default_factory=TabPFNConfig)
    ahnlab: AhnLabConfig = field(default_factory=AhnLabConfig)

    # Random seed for reproducibility
    seed: int = 42


# Default configuration instance
config = Config()
