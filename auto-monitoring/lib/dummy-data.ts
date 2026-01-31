import {
  TrainingStatus,
  TrainingModel,
  TrainingHistoryEntry,
  PredictionStatus,
  PredictionSignal,
  PredictionHistoryEntry,
  SignalType,
} from './types';
import { SYMBOLS } from './constants';

// Generate realistic training status (dummy data)
export function generateTrainingStatus(): TrainingStatus {
  const now = new Date();

  // Last training was on Jan 15, 2026
  const lastTrainingDate = new Date('2026-01-15T03:00:00.000Z');

  // Next scheduled is Feb 1, 2026 (monthly)
  const nextScheduledDate = new Date('2026-02-01T03:00:00.000Z');

  const models: TrainingModel[] = [
    {
      name: 'RSI_MACD_Model_v2',
      status: 'trained',
      accuracy: 0.72,
      lastUpdated: '2026-01-15T05:23:00.000Z',
      symbols: 101,
    },
    {
      name: 'LSTM_Price_Predictor_v1',
      status: 'trained',
      accuracy: 0.68,
      lastUpdated: '2026-01-15T06:45:00.000Z',
      symbols: 101,
    },
    {
      name: 'Ensemble_Voter_v1',
      status: 'trained',
      accuracy: 0.75,
      lastUpdated: '2026-01-15T07:12:00.000Z',
      symbols: 101,
    },
  ];

  const history: TrainingHistoryEntry[] = [
    {
      date: '2026-01-15',
      duration: '4h 12m',
      status: 'success',
      metrics: { accuracy: 0.75, loss: 0.28 },
    },
    {
      date: '2025-12-15',
      duration: '3h 58m',
      status: 'success',
      metrics: { accuracy: 0.73, loss: 0.31 },
    },
    {
      date: '2025-11-15',
      duration: '4h 05m',
      status: 'success',
      metrics: { accuracy: 0.71, loss: 0.33 },
    },
    {
      date: '2025-10-15',
      duration: '3h 45m',
      status: 'failed',
      metrics: { accuracy: 0.65, loss: 0.42 },
    },
  ];

  return {
    status: 'idle',
    lastTraining: lastTrainingDate.toISOString(),
    nextScheduled: nextScheduledDate.toISOString(),
    models,
    history,
  };
}

// Generate realistic prediction status (dummy data)
export function generatePredictionStatus(): PredictionStatus {
  const now = new Date();

  // Last prediction was today at 8 AM
  const today = new Date();
  today.setHours(8, 0, 0, 0);

  // Next scheduled is tomorrow at 8 AM
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  // Generate random but realistic signals
  const buyCount = Math.floor(Math.random() * 10) + 8;  // 8-17
  const sellCount = Math.floor(Math.random() * 8) + 5;  // 5-12
  const holdCount = 101 - buyCount - sellCount;

  // Generate top signals with realistic RSI/MACD values
  const topSignals: PredictionSignal[] = [
    { symbol: 'NVDA', signal: 'BUY', confidence: 0.89, rsi: 28.5, macd: 3.2 },
    { symbol: 'AMD', signal: 'BUY', confidence: 0.85, rsi: 31.2, macd: 2.8 },
    { symbol: 'META', signal: 'SELL', confidence: 0.82, rsi: 72.4, macd: -2.1 },
    { symbol: 'TSLA', signal: 'BUY', confidence: 0.78, rsi: 35.8, macd: 1.9 },
    { symbol: 'AAPL', signal: 'HOLD', confidence: 0.75, rsi: 52.3, macd: 0.4 },
    { symbol: 'MSFT', signal: 'HOLD', confidence: 0.73, rsi: 48.7, macd: 0.2 },
    { symbol: 'GOOGL', signal: 'SELL', confidence: 0.71, rsi: 68.9, macd: -1.5 },
    { symbol: 'AMZN', signal: 'BUY', confidence: 0.69, rsi: 38.2, macd: 1.2 },
  ];

  // Generate history for last 7 days
  const history: PredictionHistoryEntry[] = [];
  for (let i = 0; i < 7; i++) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const buy = Math.floor(Math.random() * 10) + 8;
    const sell = Math.floor(Math.random() * 8) + 5;
    history.push({
      date: date.toISOString().split('T')[0],
      buy,
      sell,
      hold: 101 - buy - sell,
    });
  }

  return {
    status: 'completed',
    lastPrediction: today.toISOString(),
    nextScheduled: tomorrow.toISOString(),
    summary: {
      totalSymbols: 101,
      buySignals: buyCount,
      sellSignals: sellCount,
      holdSignals: holdCount,
    },
    topSignals,
    history,
  };
}

// Helper to get signal color
export function getSignalColor(signal: SignalType): string {
  switch (signal) {
    case 'BUY':
      return 'text-green-600 dark:text-green-400';
    case 'SELL':
      return 'text-red-600 dark:text-red-400';
    case 'HOLD':
      return 'text-yellow-600 dark:text-yellow-400';
  }
}

// Helper to get signal badge variant
export function getSignalBadgeVariant(signal: SignalType): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (signal) {
    case 'BUY':
      return 'default';  // green-ish
    case 'SELL':
      return 'destructive';  // red
    case 'HOLD':
      return 'secondary';  // gray
  }
}
