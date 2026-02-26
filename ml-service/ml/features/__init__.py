"""
Feature engineering modules
"""

from .technical import add_technical_features
from .momentum import add_momentum_features
from .volatility import add_volatility_features
from .volume import add_volume_features
from .returns import add_return_features
from .cross_sectional import add_cross_sectional_features
from .enhanced import add_enhanced_features, add_enhanced_cross_sectional
from .patterns import add_pattern_features
from .regime import add_regime_features
from .decomposition import add_decomposition_features
from .interaction import add_interaction_features
from .autocorr import add_autocorr_features
from .portfolio import add_portfolio_features
from .advanced_technical import (
    add_advanced_technical_features,
    ADVANCED_TECHNICAL_FEATURES,
)
from .new_interactions import add_new_interaction_features, NEW_INTERACTION_FEATURES
from .advanced_decomposition import (
    add_advanced_decomposition_features,
    ADVANCED_DECOMPOSITION_FEATURES,
)

__all__ = [
    "add_technical_features",
    "add_momentum_features",
    "add_volatility_features",
    "add_volume_features",
    "add_return_features",
    "add_cross_sectional_features",
    "add_enhanced_features",
    "add_enhanced_cross_sectional",
    "add_pattern_features",
    "add_regime_features",
    "add_decomposition_features",
    "add_interaction_features",
    "add_autocorr_features",
    "add_portfolio_features",
    "add_advanced_technical_features",
    "add_new_interaction_features",
    "add_advanced_decomposition_features",
    "ADVANCED_TECHNICAL_FEATURES",
    "NEW_INTERACTION_FEATURES",
    "ADVANCED_DECOMPOSITION_FEATURES",
]
