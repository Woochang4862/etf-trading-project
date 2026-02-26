# Train.py & FeaturePipeline Integration

## Summary

AhnLab train.py has been successfully integrated with the new FeaturePipeline system. The integration provides two modes of operation:

1. **Legacy mode** (default): Load pre-generated parquet file
2. **Pipeline mode**: Generate features on-the-fly using FeaturePipeline

## Changes Made

### 1. Updated `etf-model/src/features/ahnlab/__init__.py`

**Added exports:**
- `add_technical_indicators` function
- `MacroDataCollector` class
- `TECHNICAL_FEATURE_COLS` constant (list of technical indicator column names)
- `MACRO_FEATURE_COLS` constant (list of macro feature column names)

**Purpose:** Make AhnLab feature engineering modules accessible from train.py

### 2. Modified `AhnLab_LGBM_rank_0.19231/train.py`

#### Updated `load_panel()` function signature:

**Before:**
```python
def load_panel(path: str) -> pd.DataFrame:
```

**After:**
```python
def load_panel(path: str = None, use_pipeline: bool = False, tickers: List[str] = None) -> pd.DataFrame:
```

**New parameters:**
- `path`: Path to parquet file (used when `use_pipeline=False`)
- `use_pipeline`: If True, generate data via FeaturePipeline
- `tickers`: Required list of tickers when `use_pipeline=True`

**Behavior:**
- `use_pipeline=False` (default): Loads from parquet file (backward compatible)
- `use_pipeline=True`: Generates panel using FeaturePipeline with technical and macro features

#### Updated `run()` function:

Added `USE_PIPELINE` configuration flag at the top of the function:

```python
def run() -> None:
    # Configuration: Set to True to use FeaturePipeline instead of parquet file
    USE_PIPELINE = False
    ...
```

**When `USE_PIPELINE=True`:**
1. Collects all unique tickers from all prediction years (2020-2024)
2. Calls `load_panel(use_pipeline=True, tickers=all_tickers)`
3. FeaturePipeline generates data from 2010 to current date
4. Includes technical indicators and macro features

**When `USE_PIPELINE=False` (default):**
1. Loads from pre-generated parquet file
2. Same behavior as before (backward compatible)

## Usage

### Legacy Mode (Default)
```bash
# No changes needed - works as before
cd AhnLab_LGBM_rank_0.19231
python train.py
```

### Pipeline Mode
```python
# Edit train.py and change:
USE_PIPELINE = False  # Change to True

# Then run:
python train.py
```

### Programmatic Usage
```python
from train import load_panel

# Legacy mode
panel = load_panel(path="data/stock_panel_data.parquet", use_pipeline=False)

# Pipeline mode
tickers = ["AAPL", "GOOGL", "MSFT", ...]
panel = load_panel(use_pipeline=True, tickers=tickers)
```

## Benefits

1. **Flexibility**: Choose between fast parquet loading or fresh feature generation
2. **Consistency**: Both modes produce the same feature set
3. **Backward Compatible**: Default behavior unchanged
4. **Data Freshness**: Pipeline mode can generate up-to-date features
5. **Resource Management**: Can generate data only for needed tickers

## Testing

Run the integration test:
```bash
cd AhnLab_LGBM_rank_0.19231
python test_integration.py
```

**Test Coverage:**
- Module imports (FeaturePipeline, AhnLab modules)
- Function signature validation
- Parameter validation (required args)
- Mock pipeline mode execution

## Feature Set Comparison

Both modes should produce identical features:

**Technical Features (from FeaturePipeline):**
- RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, CCI, Williams %R, MFI
- EMAs, SMAs, VWAP, OBV
- Returns (1d, 5d, 20d, 63d)
- Volume indicators

**Macro Features (from FeaturePipeline):**
- VIX, Fed Funds Rate, Unemployment Rate, CPI
- Treasury yields (2Y, 10Y), Yield Curve
- Oil Price, USD/EUR, High Yield Spread

**Engineered Features (from train.py):**
- Additional returns (10d, 30d)
- Volatility (20d, 63d)
- Price ratios (to SMA, EMA)
- Momentum strength
- Volume surge
- Trend acceleration

**Cross-Sectional Features (from train.py):**
- Z-scores (7 features)
- Percentile ranks (5 features)

**Total: ~150+ features**

## Performance Considerations

| Mode | Speed | Memory | Use Case |
|------|-------|--------|----------|
| Parquet | Fast (seconds) | Low | Production, experiments |
| Pipeline | Slow (minutes) | High | Fresh data, one-off runs |

**Recommendation:** Use parquet mode for development and experiments. Use pipeline mode when you need fresh data or don't have a pre-generated parquet file.

## Notes

1. **Path Setup**: Pipeline mode uses `sys.path.insert(0, '../etf-model')` to import FeaturePipeline
2. **Date Range**: Pipeline mode generates data from `TRAIN_START` (2010-01-01) to current date
3. **Feature Shifting**: Both modes defer feature shifting to later in the pipeline (via `shift_features_for_prediction()`)
4. **Ticker Universe**: Pipeline mode collects all unique tickers across all prediction years to avoid multiple downloads

## Future Enhancements

1. **Caching**: Add option to save pipeline-generated data to parquet for reuse
2. **Incremental Updates**: Load existing parquet and append only new dates
3. **Parallel Processing**: Generate features for different tickers in parallel
4. **Feature Selection**: Allow specifying which feature groups to include
