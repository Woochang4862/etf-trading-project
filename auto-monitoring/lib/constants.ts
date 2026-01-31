// 101 stock symbols (organized by sector)
export const SYMBOLS = [
  // Technology (30)
  'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'ADBE', 'CRM', 'CSCO',
  'ORCL', 'AMD', 'INTC', 'QCOM', 'TXN', 'NOW', 'INTU', 'AMAT', 'ADI', 'LRCX',
  'KLAC', 'MU', 'PANW', 'CRWD', 'ANET', 'PLTR', 'APP', 'IBM', 'HOOD', 'IBKR',
  // Communication/Consumer Discretionary (4)
  'AMZN', 'TSLA', 'NFLX', 'T',
  // Consumer (9)
  'WMT', 'HD', 'COST', 'MCD', 'LOW', 'TJX', 'BKNG', 'PEP', 'KO',
  // Financials (15)
  'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW',
  'AXP', 'C', 'SPGI', 'COF', 'BX',
  // Healthcare (14)
  'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
  'ISRG', 'GILD', 'BSX', 'SYK',
  // Industrials (12)
  'CAT', 'GE', 'HON', 'UNP', 'BA', 'RTX', 'LMT', 'DE', 'ETN', 'PLD',
  'MDT', 'MMM',
  // Energy (3)
  'XOM', 'CVX', 'COP',
  // Consumer Staples (3)
  'PG', 'PM', 'LIN',
  // Utilities & Others (11)
  'NEE', 'CEG', 'DIS', 'VZ', 'TMUS', 'UBER', 'GEV', 'PGR', 'WELL', 'APH', 'ACN',
] as const;

export const TIMEFRAMES = ['12달', '1달', '1주', '1일'] as const;

export const TIMEFRAME_LABELS: Record<string, string> = {
  '12달': '1Y Daily',
  '1달': '1M Hourly',
  '1주': '1W 15min',
  '1일': '1D 5min',
};

export const TIMEFRAME_TABLES: Record<string, string> = {
  '12달': '_D',
  '1달': '_D',
  '1주': '_D',
  '1일': '_1h',
};

// API endpoints (basePath /monitor가 자동으로 적용됨)
export const API_ENDPOINTS = {
  SCRAPING_STATUS: '/monitor/api/scraping/status',
  TRAINING_STATUS: '/monitor/api/training/status',
  PREDICTION_STATUS: '/monitor/api/prediction/status',
} as const;

// Refresh intervals (ms)
export const REFRESH_INTERVALS = {
  SCRAPING: 5000,  // 5 seconds
  TRAINING: 30000, // 30 seconds
  PREDICTION: 30000, // 30 seconds
} as const;

// Log file paths (server-side only)
// Docker에서는 /app/logs/scraper.log로 마운트됨
// 로컬 개발시에는 원래 경로 사용
export const LOG_PATHS = {
  SCRAPER_LOG: process.env.NODE_ENV === 'production'
    ? '/app/logs/scraper.log'
    : '/home/ahnbi2/etf-trading-project/data-scraping/tradingview_scraper_upload.log',
  DOWNLOADS_DIR: '/home/ahnbi2/etf-trading-project/data-scraping/downloads',
  PIPELINE_LOG_DIR: process.env.NODE_ENV === 'production'
    ? '/app/logs/pipeline'
    : '/home/ahnbi2/etf-trading-project/logs',
} as const;

// Status colors for UI
export const STATUS_COLORS = {
  idle: 'bg-gray-500',
  running: 'bg-blue-500',
  completed: 'bg-green-500',
  error: 'bg-red-500',
  partial: 'bg-yellow-500',
  pending: 'bg-gray-400',
} as const;

// Symbol status colors
export const SYMBOL_STATUS_COLORS = {
  pending: 'bg-gray-200 dark:bg-gray-700',
  in_progress: 'bg-blue-200 dark:bg-blue-800',
  completed: 'bg-green-200 dark:bg-green-800',
  partial: 'bg-yellow-200 dark:bg-yellow-800',
  failed: 'bg-red-200 dark:bg-red-800',
} as const;
