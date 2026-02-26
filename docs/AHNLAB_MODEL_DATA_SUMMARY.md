# AhnLab LGBM Model - Data Structure Summary

## Quick Overview

The AhnLab LightGBM ranking model predicts the top 100 stocks for each trading day using **83 engineered features** derived from historical OHLCV and macro-economic data.

### Key Specs
- **Model Type**: LightGBMRanker (Lambda-Rank objective)
- **Input Features**: 83 total (47 base + 24 engineered + 12 cross-sectional)
- **Target**: 63-day forward return (regression) → Relevance labels (ranking)
- **Output**: Top 100 stocks per trading day
- **Training Period**: 2010-2024
- **Prediction Years**: 2020, 2021, 2022, 2023, 2024

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                │
└─────────────────────────────────────────────────────────────────┘

STEP 1: DATA COLLECTION
├─ yfinance: Daily OHLCV (2010-01-01 to 2025-01-31)
│  └─ ~540 stocks × ~5600 trading days = ~3M rows
├─ pandas_ta: 38 technical indicators
│  └─ MACD, RSI, Bollinger Bands, EMA, SMA, Stochastic, ADX, etc.
└─ FRED API: 10 macro indicators
   └─ VIX, Fed Rate, Unemployment, CPI, Treasury yields, Oil, etc.

STEP 2: SAVE PANEL DATA
└─ Output: stock_panel_data.parquet (1.3 GB)
   ├─ Columns: ticker, date, OHLCV, 38 indicators, 10 macro
   └─ Format: Apache Parquet (columnar, compressed)

STEP 3: FEATURE ENGINEERING
├─ Load parquet
├─ Create 24 engineered features:
│  ├─ Additional returns (10d, 30d)
│  ├─ Volatility (20d, 63d rolling std)
│  ├─ Price ratios (to SMAs, EMAs, 52w high)
│  ├─ Momentum metrics (weighted returns, acceleration)
│  ├─ Volume surge & trend
│  ├─ EMA crossovers & slopes
│  └─ Risk-adjusted returns (return/volatility)
└─ Output: Panel with 51 + 24 = 75 columns

STEP 4: CROSS-SECTIONAL FEATURES
├─ Z-Score Normalization (per date)
│  └─ 7 features: normalize within-date mean/std
│     (vol_63d_zs, volume_sma_20_zs, obv_zs, vwap_zs, ema_200_zs, etc.)
├─ Percentile Ranks (per date)
│  └─ 5 features: 0.0-1.0 percentile rank within date
│     (ret_20d_rank, ret_63d_rank, vol_20d_rank, momentum_rank, volume_rank)
└─ Output: Panel with 75 + 12 = 87 columns

STEP 5: TARGET & METADATA
├─ Add history position: days_since_start = cumcount per ticker
├─ Add target: target_3m = close[t+63] / close[t] - 1 (63-day forward return)
├─ Add target date: date[t+63]
└─ Output: Panel with 87 + 3 = 90 columns

STEP 6: FEATURE SHIFT (CRITICAL!)
├─ Shift ALL 83 features by -1 day
├─ Purpose: Prevent future data leakage
│  └─ Use day t-1 features to predict day t
│  └─ Complies with rule: "2020-03-05 prediction uses only 2020-03-04 data"
├─ Result: All features now contain t-1 data
└─ Output: Same 90 columns, but features are lagged by 1 day

STEP 7: TRAINING DATA PREPARATION
├─ Filter by date range
├─ Filter by minimum history: days_since_start ≥ 126 (6 months)
├─ Filter by target availability: target_date must be non-NaN and ≤ train_end
├─ Clean data:
│  ├─ Replace inf with NaN
│  ├─ Drop rows with NaN target
│  └─ Fill remaining NaN features with 0
├─ Add relevance labels:
│  └─ Per date: Quantile bin (0-49) based on target_3m rank
└─ Output: Training DataFrame with 83 features + labels

STEP 8: CREATE MODEL INPUTS
├─ Sort by [date, ticker]
├─ X = 83 features
├─ y = relevance labels (0-49)
├─ groups = count of stocks per date (for ranking)
└─ Ready for LGBMRanker.fit()

STEP 9: TRAIN ROLLING MODELS
├─ For each prediction year:
│  ├─ Fold 1: train 2010-year-3, validate year-3
│  ├─ Fold 2: train 2010-year-1, validate year-1
│  └─ Average predictions from both folds for stability
└─ Output: Trained LGBMRanker models

STEP 10: PREDICT & RANK
├─ For prediction period (Jan-Dec of year):
│  ├─ Get features for all dates
│  ├─ Predict ranking scores
│  ├─ For each date, rank stocks 1-100 by score
│  └─ Take top 100
└─ Output: Submission CSV (date, rank, ticker)
```

---

## Feature Breakdown (83 Total)

### Category 1: Core OHLCV (7)
- `open, high, low, close, volume` - Raw price data
- `dividends, stock_splits` - Corporate actions

### Category 2: Returns (4)
- `ret_1d, ret_5d, ret_20d, ret_63d` - Daily/5/20/63-day pct changes

### Category 3: MACD (3)
- `macd, macd_signal, macd_hist` - Momentum indicator

### Category 4: RSI (2)
- `rsi_14, rsi_28` - Overbought/oversold oscillators

### Category 5: Bollinger Bands (5)
- `bb_upper, bb_middle, bb_lower, bb_width, bb_position` - Volatility bands

### Category 6: Moving Averages (7)
- `ema_10, ema_20, ema_50, ema_200` - Exponential
- `sma_10, sma_20, sma_50` - Simple

### Category 7: Volume & Momentum (10)
- `obv` - On-Balance Volume
- `stoch_k, stoch_d` - Stochastic oscillator
- `adx, cci, willr, mfi` - Directional/momentum indicators
- `vwap` - Volume Weighted Average Price
- `atr_14` - Average True Range
- `volume_sma_20, volume_ratio` - Volume metrics

### Category 8: Macro Indicators (10)
- `vix, fed_funds_rate, unemployment_rate, cpi`
- `treasury_10y, treasury_2y, yield_curve`
- `oil_price, usd_eur, high_yield_spread`

**Subtotal: 51 Base Features**

---

### Category 9: Engineered Features (24)

#### 9.1 Returns & Volatility (4)
- `ret_10d, ret_30d` - Additional returns
- `vol_20d, vol_63d` - Rolling volatility

#### 9.2 Price Ratios (5)
- `price_to_sma_50, price_to_ema_200, price_to_ema_10, price_to_ema_50` - Price vs. trend
- `close_to_high_52w` - Distance from 52-week high

#### 9.3 Volume & Momentum (7)
- `volume_trend, volume_surge` - Volume anomalies
- `momentum_strength` - Weighted return momentum
- `ret_5d_20d_ratio, ret_vol_ratio_20d, ret_vol_ratio_63d` - Ratios
- `trend_acceleration` - Momentum of momentum

#### 9.4 Price Position (5)
- `close_to_high_20d, close_to_high_63d, close_to_high_126d` - Distance from recent highs

#### 9.5 EMA Features (3)
- `ema_5, ema_100` - Additional EMA periods
- `ema_cross_short, ema_cross_long, ema_slope_20` - Crossovers & trends

**Subtotal: 24 Engineered Features**

---

### Category 10: Cross-Sectional Features (12)

#### 10.1 Z-Score Normalized (7)
Computed **per date** across all stocks:
- `vol_63d_zs, volume_sma_20_zs, obv_zs, vwap_zs`
- `ema_200_zs, price_to_ema_200_zs, close_to_high_52w_zs`

Formula: `(value - date_mean) / date_std`

#### 10.2 Percentile Ranks (5)
Computed **per date** across all stocks:
- `ret_20d_rank, ret_63d_rank, vol_20d_rank`
- `momentum_strength_rank, volume_surge_rank`

Formula: `rank(value) / count(value)` → Range: 0.0 to 1.0

**Subtotal: 12 Cross-Sectional Features**

---

**TOTAL: 51 + 24 + 12 = 83 Features**

---

## Data Types

| Type | Count | Examples |
|------|-------|----------|
| float32 | 81 | All numeric features |
| int32 | 1 | `days_since_start` |
| datetime64[ns] | 1 | `date` |
| string | 1 | `ticker` |

---

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| TRAIN_START | 2010-01-01 | Historical data start |
| PRED_YEARS | [2020, 2021, 2022, 2023, 2024] | Years to predict |
| TARGET_HORIZON | 63 days | Forward return period |
| MIN_HISTORY_DAYS | 126 days | Min trading days before prediction |
| RELEVANCE_BINS | 50 | Quantile bins for ranking labels |
| TOP_K | 100 | Stocks to predict per day |
| NUM_BOOST_ROUND | 5000 | Max boosting iterations |
| EARLY_STOPPING_ROUNDS | 150 | Patience for early stopping |

---

## Data Quality Standards

### Missing Data
| Scenario | Handling |
|----------|----------|
| Macro indicators (rare) | Forward-fill, then backward-fill |
| Features after shift | Fill NaN with 0 |
| Infinite values | Replace with NaN, then fill 0 |
| Target_3m NaN | Drop row (cannot train) |

### Minimum Requirements
- **Ticker History**: ≥ 126 trading days before prediction
- **Cross-Section**: ≥ 2 stocks per date (for z-scores/ranks)
- **Target Availability**: 63-day future return must be available
- **Date Order**: Must be strictly ascending

---

## Training Strategy

### Rolling Folds (Per Year)

For prediction year Y:

| Fold | Train Period | Valid Period | Ratio |
|------|--------------|--------------|-------|
| 1 | 2010-01-01 to (Y-4)-12-31 | (Y-3)-01-01 to (Y-3)-12-31 | 3-year validation |
| 2 | 2010-01-01 to (Y-2)-12-31 | (Y-1)-01-01 to (Y-1)-12-31 | 1-year validation |

### Ensemble
- Train both folds separately
- Average predictions for final ranking
- Improves stability and robustness

### Example: Predicting 2024
- Fold 1: Train 2010-2020, Validate 2021
- Fold 2: Train 2010-2022, Validate 2023
- Final model: Average of both

---

## Output Format

### File: `{year}.submission.csv`

```csv
date,rank,ticker
2020-01-02,1,TSLA
2020-01-02,2,NFLX
2020-01-02,3,MSFT
...
2020-01-02,100,AAPL
2020-01-03,1,NFLX
2020-01-03,2,TSLA
...
```

### Statistics
- **Rows per year**: ~100 stocks × 252 trading days = ~25,000 rows
- **Total rows (2020-2024)**: ~126,000 rows
- **Columns**: date, rank, ticker

---

## Integration with ETF Trading System

### Input Requirements
To make predictions for date T:
1. Historical OHLCV for all stocks (2010-01-01 through T-1)
2. Technical indicators computed up to T-1
3. Macro data through T-1
4. Each stock must have ≥ 126 days of history

### Output Usage
- **Top 100 stocks per day** → Portfolio allocation
- **Rank 1-100** → Weighting (1=most bullish, 100=least bullish)
- **Update frequency**: Daily or monthly retraining

### Data Flow
```
Raw Data ──→ Feature Engineering ──→ Cross-Sectional Transform
                                              ↓
                                      Feature Shift (-1 day)
                                              ↓
                                      Training Windows
                                              ↓
                                      LGBMRanker Model
                                              ↓
                                      Top 100 Predictions
                                              ↓
                                      Portfolio Allocation
```

---

## File Locations

| File | Size | Purpose |
|------|------|---------|
| `AhnLab_LGBM_rank_0.19231/data/stock_panel_data.parquet` | 1.3 GB | Primary input (OHLCV + indicators) |
| `AhnLab_LGBM_rank_0.19231/train.py` | 19 KB | Training script |
| `AhnLab_LGBM_rank_0.19231/download_data.py` | 8.8 KB | Data collection |
| `AhnLab_LGBM_rank_0.19231/DATA_STRUCTURE.md` | 40 KB | Detailed documentation |
| `AhnLab_LGBM_rank_0.19231/{year}.submission.csv` | 460 KB | Predictions (2020-2024) |

---

## Quick Start

### To Regenerate Data:
```bash
cd AhnLab_LGBM_rank_0.19231
export FRED_API_KEY="your_api_key_here"
python3 download_data.py  # Downloads data, creates parquet
```

### To Train Model:
```bash
python3 train.py          # Trains on 2010-2024, generates submissions
```

### To Use in Pipeline:
```python
import pandas as pd
import lightgbm as lgb

# Load trained model
model = lgb.Booster(model_file="path/to/model.txt")

# Make predictions
preds = model.predict(X_test)  # 83 features
top_100 = X_test.nlargest(100, "pred_score")["ticker"]
```

---

## Next Steps for Integration

1. **Load Panel Data**: `pd.read_parquet("stock_panel_data.parquet")`
2. **Apply Feature Engineering**: Add 24 engineered features
3. **Add Cross-Sectional**: Z-scores and ranks per date
4. **Shift Features**: -1 day to prevent leakage
5. **Make Predictions**: Pass 83 features to trained model
6. **Allocate Portfolio**: Use top 100 rankings for weighting

---

## References

- **Model Directory**: `/home/ahnbi2/etf-trading-project/AhnLab_LGBM_rank_0.19231/`
- **Detailed Docs**: `AhnLab_LGBM_rank_0.19231/DATA_STRUCTURE.md`
- **Training Script**: `AhnLab_LGBM_rank_0.19231/train.py` (lines 1-596)
- **Data Script**: `AhnLab_LGBM_rank_0.19231/download_data.py` (lines 1-260)

---

## Summary

The AhnLab LGBM model uses a comprehensive 83-feature engineering pipeline built on 11 years of historical stock data (2010-2024) combined with real-time macro-economic indicators. The model is trained using LightGBM's Lambda-Rank objective to predict stock rankings rather than absolute returns, ensuring stability and robustness in the top 100 stock predictions delivered daily.
