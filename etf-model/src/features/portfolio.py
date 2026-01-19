"""
Portfolio-level features for ETF Stock Prediction

Captures market-wide information and breadth indicators.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def add_portfolio_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add portfolio-level features to panel data

    These features capture market-wide information that helps
    understand individual stocks' performance in context.

    Args:
        panel: Panel DataFrame with columns: date, ticker, returns...

    Returns:
        Panel with added portfolio features
    """
    panel = panel.copy()

    if "date" not in panel.columns or "ticker" not in panel.columns:
        raise ValueError("Panel must have 'date' and 'ticker' columns")

    # === Market Breadth ===
    if "ret_1d" in panel.columns:
        # Advance/Decline ratio
        adv = panel.groupby("date")["ret_1d"].apply(lambda x: (x > 0).sum())
        dec = panel.groupby("date")["ret_1d"].apply(lambda x: (x < 0).sum())

        adv_dec_ratio = adv / (dec + 1e-10)
        panel["adv_dec_ratio"] = panel["date"].map(adv_dec_ratio)

        # Advance/Decline line
        ad_line = (adv - dec).cumsum()
        panel["ad_line"] = panel["date"].map(ad_line)

        # Breadth percentage (stocks with positive return)
        pos_count = adv
        total_count = panel.groupby("date").size()
        breadth_pct = pos_count / total_count
        panel["market_breadth_pct"] = panel["date"].map(breadth_pct)

        # Strong breadth day
        panel["strong_breadth_day"] = (breadth_pct > 0.6).map(
            lambda x: 1 if pd.notna(x) else 0
        )
        panel["weak_breadth_day"] = (breadth_pct < 0.4).map(
            lambda x: 1 if pd.notna(x) else 0
        )

    # === Participation Rate ===
    # How much did this stock move vs market?
    if "ret_1d" in panel.columns:
        market_return = panel.groupby("date")["ret_1d"].mean()
        market_return_std = panel.groupby("date")["ret_1d"].std()

        # Participation = stock return / market return
        panel["participation_rate"] = panel["ret_1d"] / (
            panel["date"].map(market_return) + 1e-10
        )

        # Normalized participation (relative to market std)
        panel["participation_zscore"] = (
            panel["ret_1d"] - panel["date"].map(market_return)
        ) / (panel["date"].map(market_return_std) + 1e-10)

        # High participation stock
        panel["high_participation"] = (panel["participation_rate"] > 1.5).astype(int)
        panel["low_participation"] = (panel["participation_rate"] < 0.5).astype(int)
        panel["negative_participation"] = (panel["participation_rate"] < 0).astype(int)

    # === Market Concentration ===
    # Top 10 stocks' market cap / total market cap
    if "volume" in panel.columns and "close" in panel.columns:
        panel["market_cap"] = panel["close"] * panel["volume"]

        top10_cap = panel.groupby("date")["market_cap"].apply(
            lambda x: x.nlargest(10).sum()
        )
        total_cap = panel.groupby("date")["market_cap"].sum()

        concentration = top10_cap / (total_cap + 1e-10)
        panel["market_concentration"] = panel["date"].map(concentration)

        # High concentration day
        panel["high_concentration_day"] = (concentration > 0.5).map(
            lambda x: 1 if pd.notna(x) else 0
        )

    # === Cross-Sectional Volatility ===
    # How dispersed are returns across stocks?
    if "ret_1d" in panel.columns:
        cs_vol = panel.groupby("date")["ret_1d"].std()
        panel["cross_sectional_vol"] = panel["date"].map(cs_vol)

        # Rolling cross-sectional volatility
        cs_vol_ma20 = cs_vol.rolling(20).mean()
        panel["cs_vol_ma20"] = panel["date"].map(cs_vol_ma20)

        # High vs low cross-sectional vol
        panel["high_cs_vol_day"] = (
            panel["cross_sectional_vol"] > panel["cs_vol_ma20"]
        ).map(lambda x: 1 if pd.notna(x) else 0)

    # === Momentum Leadership ===
    # Are winners and losers consistent across timeframes?
    if all(col in panel.columns for col in ["ret_5d", "ret_20d"]):
        # Calculate correlation between 5d and 20d winners
        is_winner_5d = panel.groupby("date")["ret_5d"].transform(
            lambda x: x > x.quantile(0.8)
        )
        is_winner_20d = panel.groupby("date")["ret_20d"].transform(
            lambda x: x > x.quantile(0.8)
        )

        # Winners in both timeframes
        panel["consistent_winner"] = (is_winner_5d & is_winner_20d).astype(int)

        # Losers in both timeframes
        is_loser_5d = panel.groupby("date")["ret_5d"].transform(
            lambda x: x < x.quantile(0.2)
        )
        is_loser_20d = panel.groupby("date")["ret_20d"].transform(
            lambda x: x < x.quantile(0.2)
        )
        panel["consistent_loser"] = (is_loser_5d & is_loser_20d).astype(int)

    # === Relative Strength vs Market ===
    if all(col in panel.columns for col in ["ret_20d", "volume"]):
        # Dollar-volume weighted market return
        panel["dollar_ret_20d"] = panel["close"] * panel["volume"] * panel["ret_20d"]

        market_dw_ret = panel.groupby("date")["dollar_ret_20d"].sum() / panel.groupby(
            "date"
        )["dollar_ret_20d"].apply(lambda x: x.abs().sum())

        # Stock's contribution to market return
        # Get total absolute dollar return per date for normalization
        total_abs_dollar_ret = panel.groupby("date")["dollar_ret_20d"].apply(
            lambda x: x.abs().sum()
        )
        panel["market_contribution"] = panel["dollar_ret_20d"] / panel["date"].map(
            total_abs_dollar_ret
        )

        # Is this a market leader (positive contribution)?
        panel["market_leader"] = (panel["market_contribution"] > 0).astype(int)

        # Is this a market lagger (negative contribution)?
        panel["market_lagger"] = (panel["market_contribution"] < 0).astype(int)

    # === Volatility Leadership ===
    # Which stocks are driving volatility?
    if "volatility_20" in panel.columns:
        market_vol = panel.groupby("date")["volatility_20"].mean()

        # Volatility relative to market
        panel["vol_relative_to_market"] = panel["volatility_20"] / (
            panel["date"].map(market_vol) + 1e-10
        )

        # High volatility stocks
        panel["high_vol_stock"] = (
            panel.groupby("date")["volatility_20"]
            .transform(lambda x: x > x.quantile(0.7))
            .astype(int)
        )

        # Low volatility stocks
        panel["low_vol_stock"] = (
            panel.groupby("date")["volatility_20"]
            .transform(lambda x: x < x.quantile(0.3))
            .astype(int)
        )

    # === Market State from Portfolio ===
    # Derive market state from portfolio features
    if "adv_dec_ratio" in panel.columns and "cross_sectional_vol" in panel.columns:
        # Adv/Dec trend
        adv_dec_trend = panel["adv_dec_ratio"].rolling(20).mean()

        # Define market states
        strong_up = (panel["adv_dec_ratio"] > 1.5) & (
            panel["adv_dec_ratio"] > adv_dec_trend
        )
        strong_down = (panel["adv_dec_ratio"] < 0.67) & (
            panel["adv_dec_ratio"] < adv_dec_trend
        )

        panel["portfolio_bull_market"] = strong_up.map(
            lambda x: 1 if pd.notna(x) else 0
        )
        panel["portfolio_bear_market"] = strong_down.map(
            lambda x: 1 if pd.notna(x) else 0
        )

    # === Momentum Regime ===
    # Are winners continuing to win?
    if all(col in panel.columns for col in ["ret_5d", "ret_20d"]):
        # Past winners' current performance
        winners_5d = panel.groupby("date")["ret_5d"].transform(
            lambda x: x > x.quantile(0.7)
        )
        panel["winner_current_ret"] = winners_5d * panel["ret_5d"]

        # Winners' average current return
        avg_winner_ret = panel.groupby("date")["winner_current_ret"].transform("mean")
        panel["avg_winner_ret"] = avg_winner_ret

        # Momentum continuation
        panel["momentum_continuing"] = (avg_winner_ret > 0).map(
            lambda x: 1 if pd.notna(x) else 0
        )
        panel["momentum_reversing"] = (avg_winner_ret < 0).map(
            lambda x: 1 if pd.notna(x) else 0
        )

    # === Market Efficiency ===
    # How well is information being priced in?
    if "ret_1d" in panel.columns:
        # Serial correlation in market returns
        market_ret = panel.groupby("date")["ret_1d"].mean()
        market_ret_lag1 = market_ret.shift(1)

        market_autocorr = market_ret.rolling(60).corr(market_ret_lag1)
        panel["market_efficiency"] = 1 - np.abs(market_autocorr)
        panel["market_efficiency"] = panel["date"].map(panel["market_efficiency"])

        # Low efficiency = predictable/opportunity
        panel["low_efficiency"] = (panel["market_efficiency"] < 0.8).map(
            lambda x: 1 if pd.notna(x) else 0
        )

    # === Dispersion Correlation with Returns ===
    # High dispersion days: stock picking more important
    if "cross_sectional_vol" in panel.columns and "ret_20d" in panel.columns:
        # Is current dispersion higher than average?
        panel["high_dispersion_regime"] = (
            panel["cross_sectional_vol"]
            > panel["cross_sectional_vol"].rolling(60).mean()
        ).map(lambda x: 1 if pd.notna(x) else 0)

        # In high dispersion, focus on individual stock quality
        panel["dispersion_interaction"] = (
            panel["high_dispersion_regime"] * panel["ret_20d"]
        )

    # === Market Phase ===
    # Combine breadth, volatility, and efficiency
    if all(
        col in panel.columns
        for col in ["market_breadth_pct", "cross_sectional_vol", "market_efficiency"]
    ):
        breadth_score = (panel["market_breadth_pct"] - 0.5) * 2  # -1 to 1
        efficiency_score = panel["market_efficiency"]  # 0 to 1

        # Market phase: 1=efficient-bull, -1=inefficient-bear, 0=neutral
        panel["market_phase"] = (breadth_score + (efficiency_score - 0.5)) / 2

        # Strong phases
        panel["strong_bull_phase"] = (panel["market_phase"] > 0.3).map(
            lambda x: 1 if pd.notna(x) else 0
        )
        panel["strong_bear_phase"] = (panel["market_phase"] < -0.3).map(
            lambda x: 1 if pd.notna(x) else 0
        )

    return panel


# List of features added by this module
PORTFOLIO_FEATURES = [
    # Market breadth
    "adv_dec_ratio",
    "ad_line",
    "market_breadth_pct",
    "strong_breadth_day",
    "weak_breadth_day",
    # Participation
    "participation_rate",
    "participation_zscore",
    "high_participation",
    "low_participation",
    "negative_participation",
    # Concentration
    "market_cap",
    "market_concentration",
    "high_concentration_day",
    # Cross-sectional vol
    "cross_sectional_vol",
    "cs_vol_ma20",
    "high_cs_vol_day",
    # Momentum leadership
    "consistent_winner",
    "consistent_loser",
    # Market contribution
    "dollar_ret_20d",
    "market_contribution",
    "market_leader",
    "market_lagger",
    # Volatility leadership
    "vol_relative_to_market",
    "high_vol_stock",
    "low_vol_stock",
    # Portfolio market state
    "portfolio_bull_market",
    "portfolio_bear_market",
    # Momentum regime
    "winner_current_ret",
    "avg_winner_ret",
    "momentum_continuing",
    "momentum_reversing",
    # Market efficiency
    "market_efficiency",
    "low_efficiency",
    # Dispersion
    "high_dispersion_regime",
    "dispersion_interaction",
    # Market phase
    "market_phase",
    "strong_bull_phase",
    "strong_bear_phase",
]
