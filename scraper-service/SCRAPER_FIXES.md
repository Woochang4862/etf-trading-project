# Scraper Service Export Logic Fixes

## Changes Made

### 1. Added `navigate_to_chart()` method
- Uses Korean TradingView URL: `https://kr.tradingview.com/chart/`
- Waits for chart container to load
- Matches original scraper's navigation logic

### 2. Added `search_and_select_symbol()` method
- Uses `#header-toolbar-symbol-search` selector to open search
- Fills search input with symbol name
- Clicks first search result from `[data-role="list-item"]`
- Fallback: Press Enter key
- Includes error handling with ESC key

### 3. Added `change_time_period()` method
- Locates period button by text (e.g., "1Y", "1M", "5D", "1D")
- Clicks button to change timeframe
- Includes fallback method using exact text match
- 2-second wait for chart refresh

### 4. Added `export_chart_data()` method
- **KEY FIX**: Uses JavaScript to click arrow element: `document.querySelectorAll('div[class*="arrow"]')[0].click()`
- Clicks "차트 데이터 다운로드" row
- Clicks "다운로드" button
- Waits for download completion (30s timeout)
- Saves with custom filename
- Includes ESC key error handling

### 5. Updated `scrape_symbol()` method
- Now uses the new helper methods instead of direct navigation
- Flow: search_and_select_symbol → change_time_period → export_chart_data
- Better error messages

### 6. Added `process_single_stock()` method
- Processes ALL 4 timeframes for each symbol:
  - 12달 (1Y)
  - 1달 (1M)
  - 1주 (5D)
  - 1일 (1D)
- Includes retry logic (max 3 attempts per operation)
- Returns dict of {period_name: file_path}

### 7. Updated `scrape_and_upload()` method
- Now processes all 4 timeframes per symbol automatically
- Better progress logging with separator lines
- Updates task status correctly (PARTIAL if some timeframes fail)
- More informative job completion summary

## Key Differences from Old Version

| Aspect | Old (Broken) | New (Fixed) |
|--------|--------------|-------------|
| URL | `www.tradingview.com` | `kr.tradingview.com` |
| Navigation | Direct URL with symbol | Navigate to chart → search symbol |
| Export trigger | `button[aria-label='Export chart data']` | JavaScript click on arrow element |
| Export dialog | Direct click | "차트 데이터 다운로드" row → "다운로드" button |
| Timeframes | Single timeframe | All 4 timeframes |
| Retry logic | None | 3 retries per operation |

## Testing

To test the fixes:

```bash
# Rebuild the container
docker compose build scraper-service

# Run a test scrape
curl -X POST http://localhost:8001/scrape \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "upload": true}'
```

## Expected Behavior

For each symbol, the scraper will:
1. Navigate to Korean TradingView chart
2. Search and select the symbol
3. For each of 4 timeframes:
   - Change the time period
   - Export chart data to CSV
   - Upload to database
4. Report completion status
