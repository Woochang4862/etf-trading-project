// Job status types
export type JobStatus = 'idle' | 'running' | 'completed' | 'error' | 'partial';
export type TimeframeCode = '12달' | '1달' | '1주' | '1일';
export type SignalType = 'BUY' | 'SELL' | 'HOLD';

// Scraping types
export interface ScrapingSession {
  startTime: string;
  headlessMode: boolean;
  dbUploadEnabled: boolean;
  sshTunnelActive: boolean;
}

export interface TimeframeResult {
  status: 'pending' | 'downloading' | 'success' | 'failed';
  rows?: number;
  table?: string;
  error?: string;
  downloadedAt?: string;
}

export interface SymbolScrapingStatus {
  symbol: string;
  status: 'pending' | 'in_progress' | 'completed' | 'partial' | 'failed';
  timeframes: Record<TimeframeCode, TimeframeResult>;
  startedAt?: string;
  completedAt?: string;
}

export interface ScrapingProgress {
  totalSymbols: number;
  completedSymbols: number;
  currentSymbol: string | null;
  currentTimeframe: TimeframeCode | null;
  percentage: number;
}

export interface ScrapingStatistics {
  totalDownloads: number;
  successfulUploads: number;
  failedDownloads: number;
  totalRowsUploaded: number;
}

export interface ScrapingError {
  timestamp: string;
  symbol: string;
  timeframe: TimeframeCode;
  type: 'timeout' | 'download' | 'upload' | 'parse' | 'unknown';
  message: string;
}

export interface ScrapingStatus {
  status: JobStatus;
  lastRun: string | null;
  currentSession: ScrapingSession | null;
  progress: ScrapingProgress;
  statistics: ScrapingStatistics;
  symbols: SymbolScrapingStatus[];
  errors: ScrapingError[];
}

// Training types (dummy)
export interface TrainingModel {
  name: string;
  status: 'pending' | 'training' | 'trained' | 'failed';
  accuracy: number;
  lastUpdated: string;
  symbols: number;
}

export interface TrainingHistoryEntry {
  date: string;
  duration: string;
  status: 'success' | 'failed';
  metrics: {
    accuracy: number;
    loss: number;
  };
}

export interface TrainingStatus {
  status: 'idle' | 'training' | 'completed';
  lastTraining: string | null;
  nextScheduled: string;
  models: TrainingModel[];
  history: TrainingHistoryEntry[];
}

// Prediction types (dummy)
export interface PredictionSignal {
  symbol: string;
  signal: SignalType;
  confidence: number;
  rsi: number;
  macd: number;
  timestamp?: string;
}

export interface PredictionSummary {
  totalSymbols: number;
  buySignals: number;
  sellSignals: number;
  holdSignals: number;
}

export interface PredictionHistoryEntry {
  date: string;
  buy: number;
  sell: number;
  hold: number;
}

export interface PredictionStatus {
  status: 'idle' | 'running' | 'completed';
  lastPrediction: string | null;
  nextScheduled: string;
  summary: PredictionSummary;
  topSignals: PredictionSignal[];
  history: PredictionHistoryEntry[];
}

// Log parsing types
export interface ParsedLogEntry {
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  parsed?: {
    type: 'session_start' | 'symbol_start' | 'timeframe_start' | 'download' | 'parse' | 'upload' | 'error' | 'result' | 'all_complete';
    symbol?: string;
    timeframe?: TimeframeCode;
    rows?: number;
    table?: string;
    filename?: string;
    error?: string;
  };
}
