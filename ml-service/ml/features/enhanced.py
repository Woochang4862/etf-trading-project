"""
Enhanced features for improved prediction accuracy
"""
import pandas as pd
import numpy as np
from typing import List


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add enhanced predictive features

    Args:
        df: DataFrame with OHLCV and existing features

    Returns:
        DataFrame with enhanced features
    """
    df = df.copy()

    close = df['close']
    high = df['high'] if 'high' in df.columns else close
    low = df['low'] if 'low' in df.columns else close
    volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

    # === 52-Week High/Low Proximity (강력한 모멘텀 시그널) ===
    rolling_252_high = high.rolling(window=252, min_periods=63).max()
    rolling_252_low = low.rolling(window=252, min_periods=63).min()

    df['near_52w_high'] = close / rolling_252_high
    df['near_52w_low'] = close / rolling_252_low
    df['range_52w_position'] = (close - rolling_252_low) / (rolling_252_high - rolling_252_low + 1e-10)

    # 신고가 돌파 여부
    df['new_52w_high'] = (close >= rolling_252_high).astype(int)

    # === Price Level Features ===
    # 가격 수준 (저가주 vs 고가주 효과)
    df['log_price'] = np.log(close + 1)

    # === Momentum Quality (모멘텀의 질) ===
    if 'ret_20d' in df.columns and 'volatility_20' in df.columns:
        # Risk-adjusted momentum
        df['momentum_quality'] = df['ret_20d'] / (df['volatility_20'] + 1e-10)

    if 'ret_63d' in df.columns and 'volatility_63' in df.columns:
        df['momentum_quality_63'] = df['ret_63d'] / (df['volatility_63'] + 1e-10)

    # === Mean Reversion Signals ===
    # RSI 극단값에서의 반전 확률
    if 'rsi_14' in df.columns:
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        # RSI 정규화 (0-100 → -1~1)
        df['rsi_normalized'] = (df['rsi_14'] - 50) / 50

    # === Volume-Momentum Interaction ===
    if 'volume_ratio' in df.columns:
        # 거래량 동반 상승 (강한 모멘텀)
        if 'ret_5d' in df.columns:
            df['vol_momentum_5d'] = df['ret_5d'] * np.log1p(df['volume_ratio'])
        if 'ret_20d' in df.columns:
            df['vol_momentum_20d'] = df['ret_20d'] * np.log1p(df['volume_ratio'])

    # === Trend Consistency ===
    daily_ret = close.pct_change()

    # 최근 N일 중 상승일 비율
    for period in [5, 10, 20]:
        df[f'up_days_ratio_{period}'] = (daily_ret > 0).rolling(window=period).mean()

    # === Gap Analysis ===
    if 'open' in df.columns:
        # 갭 방향과 크기
        gap = (df['open'] - close.shift(1)) / close.shift(1)
        df['gap_size'] = gap
        df['gap_up'] = (gap > 0.02).astype(int)
        df['gap_down'] = (gap < -0.02).astype(int)

        # 갭 이후 추세 지속성
        df['gap_filled'] = ((gap > 0) & (close < df['open'])).astype(int)

    # === Multi-timeframe Momentum ===
    # 단기, 중기, 장기 모멘텀 조합
    if all(f'ret_{p}d' in df.columns for p in [5, 20, 63]):
        # 모든 시간대 양의 모멘텀
        df['momentum_alignment'] = (
            (df['ret_5d'] > 0).astype(int) +
            (df['ret_20d'] > 0).astype(int) +
            (df['ret_63d'] > 0).astype(int)
        )

        # 모멘텀 가속도
        df['momentum_accel'] = df['ret_5d'] - df['ret_20d'] / 4

    # === Relative Strength vs Recent Performance ===
    # 최근 성과 대비 현재 위치
    if 'ret_63d' in df.columns and 'ret_5d' in df.columns:
        df['short_vs_long_mom'] = df['ret_5d'] - df['ret_63d'] / 12.6

    # === Volatility-Adjusted Features ===
    if 'atr_14' in df.columns:
        # ATR 대비 가격 이동
        df['move_vs_atr'] = daily_ret.abs() / (df['atr_14'] / close + 1e-10)

        # ATR 대비 거래 범위
        if 'high' in df.columns and 'low' in df.columns:
            df['range_vs_atr'] = (high - low) / (df['atr_14'] + 1e-10)

    # === Technical Confluence ===
    # 여러 지표의 일치 여부
    signals = 0
    if 'rsi_14' in df.columns:
        signals += (df['rsi_14'] > 50).astype(int)
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        signals += (df['macd'] > df['macd_signal']).astype(int)
    if 'price_to_sma_20' in df.columns:
        signals += (df['price_to_sma_20'] > 0).astype(int)
    if 'bb_position' in df.columns:
        signals += (df['bb_position'] > 0.5).astype(int)

    df['tech_confluence'] = signals

    # === Lagged Features (과거 성과의 지속성) ===
    for lag in [1, 5, 10]:
        if 'ret_5d' in df.columns:
            df[f'ret_5d_lag{lag}'] = df['ret_5d'].shift(lag)
        if 'volume_ratio' in df.columns:
            df[f'volume_ratio_lag{lag}'] = df['volume_ratio'].shift(lag)

    # === Rolling Z-Score of Returns ===
    for period in [20, 63]:
        ret_mean = daily_ret.rolling(window=period).mean()
        ret_std = daily_ret.rolling(window=period).std()
        df[f'ret_zscore_{period}'] = (daily_ret - ret_mean) / (ret_std + 1e-10)

    # === Breakout Detection ===
    # 최근 고점 돌파
    for period in [20, 50]:
        recent_high = high.rolling(window=period).max().shift(1)
        df[f'breakout_{period}'] = (close > recent_high).astype(int)

    # === Volatility Regime Features (시장 변동성 적응) ===
    # 현재 변동성 vs 장기 평균 변동성
    vol_20 = daily_ret.rolling(window=20).std()
    vol_63 = daily_ret.rolling(window=63).std()
    vol_252 = daily_ret.rolling(window=252, min_periods=63).std()

    df['vol_ratio_short_long'] = vol_20 / (vol_63 + 1e-10)  # 단기/중기 변동성 비율
    df['vol_ratio_vs_annual'] = vol_20 / (vol_252 + 1e-10)  # 현재/연간 변동성 비율
    df['vol_regime'] = (vol_20 > vol_252).astype(int)  # 고변동성 레짐

    # 변동성 추세 (증가 vs 감소)
    df['vol_trend'] = vol_20 - vol_20.shift(10)
    df['vol_acceleration'] = df['vol_trend'] - df['vol_trend'].shift(5)

    # === Drawdown Features ===
    # 최근 고점 대비 하락률 (드로다운)
    rolling_max = close.rolling(window=252, min_periods=63).max()
    df['drawdown'] = (close - rolling_max) / rolling_max
    df['drawdown_20'] = (close - close.rolling(window=20).max()) / close.rolling(window=20).max()

    # 드로다운에서 회복 중인지
    df['recovering'] = ((df['drawdown'] < -0.1) & (df['ret_5d'] > 0)).astype(int) if 'ret_5d' in df.columns else 0

    # === Volatility-Adjusted Returns (변동성 조정 수익률) ===
    for period in [5, 20, 63]:
        ret_col = f'ret_{period}d'
        if ret_col in df.columns:
            vol_period = daily_ret.rolling(window=period).std()
            df[f'sharpe_{period}d'] = df[ret_col] / (vol_period * np.sqrt(period) + 1e-10)

    # === Downside Risk Features ===
    # 하방 변동성 (음의 수익률만 사용)
    negative_ret = daily_ret.where(daily_ret < 0, 0)
    df['downside_vol_20'] = negative_ret.rolling(window=20).std()
    df['sortino_ratio'] = (daily_ret.rolling(window=20).mean()) / (df['downside_vol_20'] + 1e-10)

    # === Tail Risk Features ===
    # 극단적 움직임 빈도
    df['extreme_up_freq'] = (daily_ret > daily_ret.rolling(252).quantile(0.95)).rolling(20).sum()
    df['extreme_down_freq'] = (daily_ret < daily_ret.rolling(252).quantile(0.05)).rolling(20).sum()

    # === Price Distance from Moving Averages ===
    for period in [50, 200]:
        ma = close.rolling(window=period, min_periods=period//2).mean()
        df[f'dist_from_ma_{period}'] = (close - ma) / ma

    # MA 크로스 시그널
    ma_50 = close.rolling(window=50, min_periods=25).mean()
    ma_200 = close.rolling(window=200, min_periods=100).mean()
    df['golden_cross'] = ((ma_50 > ma_200) & (ma_50.shift(1) <= ma_200.shift(1))).astype(int)
    df['death_cross'] = ((ma_50 < ma_200) & (ma_50.shift(1) >= ma_200.shift(1))).astype(int)
    df['above_ma_200'] = (close > ma_200).astype(int)

    # === Trend Strength ===
    # ADX가 없는 경우 간단한 추세 강도 계산
    price_change = close.diff(20)
    price_range = high.rolling(20).max() - low.rolling(20).min()
    df['trend_efficiency'] = price_change.abs() / (price_range + 1e-10)

    return df


def add_enhanced_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add enhanced cross-sectional features

    Args:
        panel: Panel DataFrame with date, ticker columns

    Returns:
        Panel with enhanced cross-sectional features
    """
    panel = panel.copy()

    # === Composite Momentum Score ===
    # 여러 기간 모멘텀의 가중 평균 랭크
    mom_cols = ['ret_5d', 'ret_20d', 'ret_63d']
    available_mom = [c for c in mom_cols if c in panel.columns]

    if available_mom:
        for col in available_mom:
            panel[f'{col}_pct_rank'] = panel.groupby('date')[col].transform(
                lambda x: x.rank(pct=True, na_option='keep')
            )

        # 복합 모멘텀 점수 (단기:중기:장기 = 1:2:1)
        weights = {'ret_5d_pct_rank': 1, 'ret_20d_pct_rank': 2, 'ret_63d_pct_rank': 1}
        score = 0
        total_weight = 0
        for col, w in weights.items():
            if col in panel.columns:
                score += panel[col] * w
                total_weight += w
        if total_weight > 0:
            panel['composite_momentum_score'] = score / total_weight

    # === Relative Strength Index (vs Universe) ===
    if 'ret_20d' in panel.columns:
        market_ret = panel.groupby('date')['ret_20d'].transform('median')
        panel['relative_strength_20d'] = panel['ret_20d'] - market_ret

    # === Sector-like Clustering (based on correlation) ===
    # 유사 종목 대비 성과 (단순 버전: 수익률 사분위 내 상대 성과)
    if 'ret_20d' in panel.columns:
        panel['ret_20d_quartile'] = panel.groupby('date')['ret_20d'].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=4, labels=False, duplicates='drop')
        )
        panel['intra_quartile_rank'] = panel.groupby(['date', 'ret_20d_quartile'])['ret_5d'].transform(
            lambda x: x.rank(pct=True, na_option='keep')
        )

    # === Volume Leadership ===
    if 'volume_ratio' in panel.columns:
        panel['volume_leadership'] = panel.groupby('date')['volume_ratio'].transform(
            lambda x: x.rank(pct=True, na_option='keep')
        )

    # === Combined Rank Score ===
    rank_cols = [
        'ret_20d_pct_rank', 'volume_leadership', 'near_52w_high'
    ]
    available_ranks = [c for c in rank_cols if c in panel.columns]

    if available_ranks:
        panel['combined_rank_score'] = panel[available_ranks].mean(axis=1)

    # === Enhanced Relative Momentum (상대 모멘텀 강화) ===

    # 시장 대비 초과 수익률 (여러 기간)
    for period in [5, 10, 20, 63]:
        ret_col = f'ret_{period}d'
        if ret_col in panel.columns:
            market_median = panel.groupby('date')[ret_col].transform('median')
            market_mean = panel.groupby('date')[ret_col].transform('mean')
            panel[f'excess_ret_{period}d_median'] = panel[ret_col] - market_median
            panel[f'excess_ret_{period}d_mean'] = panel[ret_col] - market_mean

    # 모멘텀 지속성: 과거 상위 종목이 계속 상위인지
    if 'ret_63d_pct_rank' in panel.columns:
        panel['momentum_persistence'] = panel.groupby('ticker')['ret_63d_pct_rank'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )

    # 상대 강도 지수 (RS): 시장 대비 상대 성과의 추세
    if 'ret_20d' in panel.columns:
        market_ret = panel.groupby('date')['ret_20d'].transform('mean')
        relative_perf = panel['ret_20d'] - market_ret
        panel['relative_strength_trend'] = panel.groupby('ticker')['ret_20d'].transform(
            lambda x: x.rolling(window=10, min_periods=5).mean()
        ) - market_ret

    # 섹터 내 순위 (수익률 분위 내 상대 순위)
    for col in ['sharpe_20d', 'momentum_quality']:
        if col in panel.columns:
            panel[f'{col}_rank'] = panel.groupby('date')[col].transform(
                lambda x: x.rank(pct=True, na_option='keep')
            )

    # 변동성 조정 상대 모멘텀
    if 'sharpe_20d' in panel.columns:
        panel['sharpe_rank'] = panel.groupby('date')['sharpe_20d'].transform(
            lambda x: x.rank(pct=True, na_option='keep')
        )

    # 드로다운 대비 회복 순위
    if 'drawdown' in panel.columns:
        panel['drawdown_rank'] = panel.groupby('date')['drawdown'].transform(
            lambda x: x.rank(pct=True, ascending=False, na_option='keep')  # 드로다운 적은 것이 상위
        )

    # 복합 점수: 모멘텀 + 변동성 조정 + 추세
    score_cols = ['ret_20d_pct_rank', 'sharpe_rank', 'near_52w_high']
    available_scores = [c for c in score_cols if c in panel.columns]
    if len(available_scores) >= 2:
        panel['quality_momentum_score'] = panel[available_scores].mean(axis=1)

    # 모멘텀 분산 (일관성 측정)
    if all(f'ret_{p}d_pct_rank' in panel.columns for p in [5, 20, 63]):
        panel['momentum_consistency'] = 1 - panel[['ret_5d_pct_rank', 'ret_20d_pct_rank', 'ret_63d_pct_rank']].std(axis=1)

    return panel


ENHANCED_FEATURES = [
    # 52-week features
    'near_52w_high', 'near_52w_low', 'range_52w_position', 'new_52w_high',
    # Price level
    'log_price',
    # Momentum quality
    'momentum_quality', 'momentum_quality_63',
    # Mean reversion
    'rsi_oversold', 'rsi_overbought', 'rsi_normalized',
    # Volume-momentum
    'vol_momentum_5d', 'vol_momentum_20d',
    # Trend consistency
    'up_days_ratio_5', 'up_days_ratio_10', 'up_days_ratio_20',
    # Gap
    'gap_size', 'gap_up', 'gap_down', 'gap_filled',
    # Multi-timeframe
    'momentum_alignment', 'momentum_accel', 'short_vs_long_mom',
    # Volatility-adjusted
    'move_vs_atr', 'range_vs_atr',
    # Confluence
    'tech_confluence',
    # Lagged
    'ret_5d_lag1', 'ret_5d_lag5', 'ret_5d_lag10',
    'volume_ratio_lag1', 'volume_ratio_lag5', 'volume_ratio_lag10',
    # Z-score
    'ret_zscore_20', 'ret_zscore_63',
    # Breakout
    'breakout_20', 'breakout_50',
    # Volatility regime (NEW)
    'vol_ratio_short_long', 'vol_ratio_vs_annual', 'vol_regime',
    'vol_trend', 'vol_acceleration',
    # Drawdown (NEW)
    'drawdown', 'drawdown_20', 'recovering',
    # Sharpe/Sortino (NEW)
    'sharpe_5d', 'sharpe_20d', 'sharpe_63d',
    'downside_vol_20', 'sortino_ratio',
    # Tail risk (NEW)
    'extreme_up_freq', 'extreme_down_freq',
    # MA distance (NEW)
    'dist_from_ma_50', 'dist_from_ma_200',
    'golden_cross', 'death_cross', 'above_ma_200',
    # Trend efficiency (NEW)
    'trend_efficiency',
]

ENHANCED_CROSS_SECTIONAL_FEATURES = [
    'ret_5d_pct_rank', 'ret_20d_pct_rank', 'ret_63d_pct_rank',
    'composite_momentum_score',
    'relative_strength_20d',
    'ret_20d_quartile', 'intra_quartile_rank',
    'volume_leadership',
    'combined_rank_score',
    # Enhanced relative momentum (NEW)
    'excess_ret_5d_median', 'excess_ret_5d_mean',
    'excess_ret_10d_median', 'excess_ret_10d_mean',
    'excess_ret_20d_median', 'excess_ret_20d_mean',
    'excess_ret_63d_median', 'excess_ret_63d_mean',
    'momentum_persistence', 'relative_strength_trend',
    'sharpe_20d_rank', 'momentum_quality_rank',
    'sharpe_rank', 'drawdown_rank',
    'quality_momentum_score', 'momentum_consistency',
]
