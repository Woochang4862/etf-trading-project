import { promises as fs } from 'fs';
import {
  ScrapingStatus,
  ScrapingSession,
  SymbolScrapingStatus,
  ScrapingError,
  TimeframeCode,
  ParsedLogEntry,
  TimeframeResult,
  ScrapingProgress,
  ScrapingStatistics,
  JobStatus,
} from './types';
import { SYMBOLS, TIMEFRAMES, LOG_PATHS } from './constants';

// Regex patterns for parsing
const LOG_PATTERNS = {
  // Main timestamp pattern: 2026-01-30 20:33:17,271 - INFO - message
  timestamp: /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (INFO|WARNING|ERROR) - (.+)$/,

  // Session config patterns
  headlessMode: /^Headless 모드: (True|False)$/,
  dbUpload: /^DB 업로드: (True|False)$/,
  sshTunnel: /^기존 SSH 터널 사용: (True|False)$/,

  // Symbol collection patterns
  symbolStart: /^(\w+(?:\.\w+)?) 데이터 수집 시작$/,
  timeframeStart: /^\[(\w+(?:\.\w+)?)\] (12달|1달|1주|1일) 데이터 수집 중\.\.\.$/,

  // Download/upload patterns
  downloadComplete: /^다운로드 완료: downloads\/(\w+(?:\.\w+)?)_(12달|1달|1주|1일)_(\d{8})\.csv$/,
  parseResult: /^Parsed (\d+) rows from (\w+(?:\.\w+)?)_(12달|1달|1주|1일)_(\d{8})\.csv$/,
  uploadComplete: /^Uploaded (\d+) rows to (\w+(?:\.\w+)?)_(D|1h)$/,
  // Note: confirmation line has leading spaces: "  [SYMBOL - TIMEFRAME] ✓ Uploaded N rows to DB"
  // Also handles: "[SCHW - 12달] ✗ DB upload failed: ..."
  uploadConfirmation: /^\s*\[(\w+(?:\.\w+)?) - (12달|1달|1주|1일)\] ([✓✗]) (?:Uploaded (\d+) rows|DB upload failed|Failed)/,

  // Error patterns
  timeframeChangeFailed: /^시간 단위 변경 최종 실패: (1Y|1M|5D|1D)$/,
  downloadFailed: /^다운로드 최종 실패: (\w+(?:\.\w+)?) - (12달|1달|1주|1일)$/,

  // Completion pattern
  allComplete: /^모든 작업 완료!$/,
};

// Map timeframe text codes to TimeframeCode
const TIMEFRAME_MAP: Record<string, TimeframeCode> = {
  '1Y': '12달',
  '1M': '1달',
  '5D': '1주',
  '1D': '1일',
};

// Helper to parse timestamp to ISO string (local time, not UTC)
function parseTimestamp(ts: string): string {
  // Convert "2026-01-30 20:33:17,271" to ISO format
  // Note: Do NOT append 'Z' as the log timestamps are in local time (KST), not UTC
  const [date, time] = ts.split(' ');
  const [timePart, ms] = time.split(',');
  return `${date}T${timePart}.${ms}`;
}

// Parse a single log line
export function parseLogLine(line: string): ParsedLogEntry | null {
  const match = line.match(LOG_PATTERNS.timestamp);
  if (!match) return null;

  const [, timestamp, level, message] = match;
  const entry: ParsedLogEntry = {
    timestamp: parseTimestamp(timestamp),
    level: level as 'INFO' | 'WARNING' | 'ERROR',
    message,
  };

  // Try to parse specific patterns

  // Session config patterns
  if (LOG_PATTERNS.headlessMode.test(message)) {
    entry.parsed = { type: 'session_start' };
  } else if (LOG_PATTERNS.dbUpload.test(message)) {
    entry.parsed = { type: 'session_start' };
  } else if (LOG_PATTERNS.sshTunnel.test(message)) {
    entry.parsed = { type: 'session_start' };
  }

  // Symbol start
  else if (LOG_PATTERNS.symbolStart.test(message)) {
    const symbolMatch = message.match(LOG_PATTERNS.symbolStart);
    if (symbolMatch) {
      entry.parsed = {
        type: 'symbol_start',
        symbol: symbolMatch[1],
      };
    }
  }

  // Timeframe start
  else if (LOG_PATTERNS.timeframeStart.test(message)) {
    const tfMatch = message.match(LOG_PATTERNS.timeframeStart);
    if (tfMatch) {
      entry.parsed = {
        type: 'timeframe_start',
        symbol: tfMatch[1],
        timeframe: tfMatch[2] as TimeframeCode,
      };
    }
  }

  // Download complete
  else if (LOG_PATTERNS.downloadComplete.test(message)) {
    const dlMatch = message.match(LOG_PATTERNS.downloadComplete);
    if (dlMatch) {
      entry.parsed = {
        type: 'download',
        symbol: dlMatch[1],
        timeframe: dlMatch[2] as TimeframeCode,
        filename: `downloads/${dlMatch[1]}_${dlMatch[2]}_${dlMatch[3]}.csv`,
      };
    }
  }

  // Parse result
  else if (LOG_PATTERNS.parseResult.test(message)) {
    const parseMatch = message.match(LOG_PATTERNS.parseResult);
    if (parseMatch) {
      entry.parsed = {
        type: 'parse',
        rows: parseInt(parseMatch[1], 10),
        symbol: parseMatch[2],
        timeframe: parseMatch[3] as TimeframeCode,
        filename: `${parseMatch[2]}_${parseMatch[3]}_${parseMatch[4]}.csv`,
      };
    }
  }

  // Upload complete
  else if (LOG_PATTERNS.uploadComplete.test(message)) {
    const uploadMatch = message.match(LOG_PATTERNS.uploadComplete);
    if (uploadMatch) {
      entry.parsed = {
        type: 'upload',
        rows: parseInt(uploadMatch[1], 10),
        symbol: uploadMatch[2],
        table: `${uploadMatch[2]}_${uploadMatch[3]}`,
      };
    }
  }

  // Upload confirmation (result)
  else if (LOG_PATTERNS.uploadConfirmation.test(message)) {
    const resultMatch = message.match(LOG_PATTERNS.uploadConfirmation);
    if (resultMatch) {
      const success = resultMatch[3] === '✓';
      entry.parsed = {
        type: 'result',
        symbol: resultMatch[1],
        timeframe: resultMatch[2] as TimeframeCode,
        rows: resultMatch[4] ? parseInt(resultMatch[4], 10) : undefined,
        error: success ? undefined : 'Upload failed',
      };
    }
  }

  // Timeframe change failed
  else if (LOG_PATTERNS.timeframeChangeFailed.test(message)) {
    const tfFailMatch = message.match(LOG_PATTERNS.timeframeChangeFailed);
    if (tfFailMatch) {
      const timeframeCode = TIMEFRAME_MAP[tfFailMatch[1]];
      entry.parsed = {
        type: 'error',
        timeframe: timeframeCode,
        error: `시간 단위 변경 최종 실패: ${tfFailMatch[1]}`,
      };
    }
  }

  // Download failed
  else if (LOG_PATTERNS.downloadFailed.test(message)) {
    const dlFailMatch = message.match(LOG_PATTERNS.downloadFailed);
    if (dlFailMatch) {
      entry.parsed = {
        type: 'error',
        symbol: dlFailMatch[1],
        timeframe: dlFailMatch[2] as TimeframeCode,
        error: `다운로드 최종 실패: ${dlFailMatch[1]} - ${dlFailMatch[2]}`,
      };
    }
  }

  // All complete
  else if (LOG_PATTERNS.allComplete.test(message)) {
    entry.parsed = {
      type: 'all_complete',
    };
  }

  return entry;
}

// Get last N lines from log file (for efficiency)
export async function getLastLogLines(n: number = 1000): Promise<string[]> {
  try {
    const content = await fs.readFile(LOG_PATHS.SCRAPER_LOG, 'utf-8');
    const lines = content.split('\n').filter(line => line.trim().length > 0);
    return lines.slice(-n);
  } catch (error) {
    // File not found or read error
    return [];
  }
}

// Determine if scraping is currently running
export function isScrapingRunning(entries: ParsedLogEntry[]): boolean {
  if (entries.length === 0) return false;

  // Check if "모든 작업 완료!" was seen - if so, scraping is done
  const hasCompleted = entries.some(entry => entry.parsed?.type === 'all_complete');
  if (hasCompleted) {
    return false;
  }

  // Check if recent entries (within last 5 minutes) show activity
  const now = new Date();
  const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);

  const recentEntries = entries.filter(entry => {
    const entryTime = new Date(entry.timestamp);
    return entryTime >= fiveMinutesAgo;
  });

  // If there are recent INFO level entries, scraping is likely running
  return recentEntries.some(entry =>
    entry.level === 'INFO' &&
    entry.parsed &&
    ['symbol_start', 'timeframe_start', 'download', 'parse', 'upload'].includes(entry.parsed.type)
  );
}

// Initialize empty symbol status
function createEmptySymbolStatus(symbol: string): SymbolScrapingStatus {
  const timeframes: Record<TimeframeCode, TimeframeResult> = {
    '12달': { status: 'pending' },
    '1달': { status: 'pending' },
    '1주': { status: 'pending' },
    '1일': { status: 'pending' },
  };

  return {
    symbol,
    status: 'pending',
    timeframes,
  };
}

// Main function to parse the entire log file and build ScrapingStatus
export async function parseScrapingLog(): Promise<ScrapingStatus> {
  try {
    // Read the log file - need enough lines to capture full session
    // 101 symbols * 4 timeframes * ~10 lines per timeframe = ~4000 lines minimum
    const lines = await getLastLogLines(15000);

    if (lines.length === 0) {
      // No log file or empty log - return idle status
      return createIdleStatus();
    }

    // Parse all lines
    let entries = lines.map(parseLogLine).filter((e): e is ParsedLogEntry => e !== null);

    if (entries.length === 0) {
      return createIdleStatus();
    }

    // Filter to only include entries from the most recent session
    // Session starts with "Database upload enabled" message
    const sessionStartIndices: number[] = [];
    entries.forEach((entry, index) => {
      if (entry.message.includes('Database upload enabled')) {
        sessionStartIndices.push(index);
      }
    });

    if (sessionStartIndices.length > 0) {
      // Get entries from the last session start onwards
      const lastSessionStart = sessionStartIndices[sessionStartIndices.length - 1];
      entries = entries.slice(lastSessionStart);
    }

    // Check if session completed - "Database service closed" appears after "모든 작업 완료!"
    const hasDbClosed = entries.some(e => e.message.includes('Database service closed'));
    if (hasDbClosed) {
      // Add a synthetic all_complete entry
      const lastEntry = entries[entries.length - 1];
      entries.push({
        timestamp: lastEntry.timestamp,
        level: 'INFO',
        message: '모든 작업 완료!',
        parsed: { type: 'all_complete' },
      });
    }

    // Build session info from config lines
    let session: ScrapingSession | null = null;
    const sessionConfigEntries = entries.filter(e => e.parsed?.type === 'session_start');

    if (sessionConfigEntries.length >= 3) {
      // Find the most recent set of session config entries
      const headlessEntry = entries.find(e => e.message.includes('Headless 모드:'));
      const dbUploadEntry = entries.find(e => e.message.includes('DB 업로드:'));
      const sshTunnelEntry = entries.find(e => e.message.includes('기존 SSH 터널 사용:'));

      if (headlessEntry && dbUploadEntry && sshTunnelEntry) {
        session = {
          startTime: headlessEntry.timestamp,
          headlessMode: headlessEntry.message.includes('True'),
          dbUploadEnabled: dbUploadEntry.message.includes('True'),
          sshTunnelActive: sshTunnelEntry.message.includes('True'),
        };
      }
    }

    // Initialize symbol tracking
    const symbolsMap = new Map<string, SymbolScrapingStatus>();
    for (const symbol of SYMBOLS) {
      symbolsMap.set(symbol, createEmptySymbolStatus(symbol));
    }

    // Track errors
    const errors: ScrapingError[] = [];

    // Statistics
    let totalDownloads = 0;
    let successfulUploads = 0;
    let failedDownloads = 0;
    let totalRowsUploaded = 0;

    // Current tracking for progress
    let currentSymbol: string | null = null;
    let currentTimeframe: TimeframeCode | null = null;

    // Process all entries chronologically
    for (const entry of entries) {
      if (!entry.parsed) continue;

      const { type, symbol, timeframe, rows, table, error } = entry.parsed;

      if (type === 'symbol_start' && symbol) {
        currentSymbol = symbol;
        const symbolStatus = symbolsMap.get(symbol);
        if (symbolStatus) {
          symbolStatus.status = 'in_progress';
          symbolStatus.startedAt = entry.timestamp;
        }
      }

      else if (type === 'timeframe_start' && symbol && timeframe) {
        currentSymbol = symbol;
        currentTimeframe = timeframe;
        const symbolStatus = symbolsMap.get(symbol);
        if (symbolStatus) {
          symbolStatus.timeframes[timeframe].status = 'downloading';
        }
      }

      else if (type === 'download' && symbol && timeframe) {
        totalDownloads++;
        const symbolStatus = symbolsMap.get(symbol);
        if (symbolStatus) {
          symbolStatus.timeframes[timeframe].downloadedAt = entry.timestamp;
        }
      }

      else if (type === 'upload' && symbol && rows !== undefined && table) {
        const symbolStatus = symbolsMap.get(symbol);
        if (symbolStatus) {
          // Find which timeframe this upload corresponds to
          for (const tf of TIMEFRAMES) {
            if (symbolStatus.timeframes[tf].status === 'downloading') {
              symbolStatus.timeframes[tf].status = 'success';
              symbolStatus.timeframes[tf].rows = rows;
              symbolStatus.timeframes[tf].table = table;
              totalRowsUploaded += rows;
              successfulUploads++;
              break;
            }
          }
        }
      }

      else if (type === 'result' && symbol && timeframe) {
        const symbolStatus = symbolsMap.get(symbol);
        if (symbolStatus) {
          if (error) {
            symbolStatus.timeframes[timeframe].status = 'failed';
            symbolStatus.timeframes[timeframe].error = error;
            failedDownloads++;

            errors.push({
              timestamp: entry.timestamp,
              symbol,
              timeframe,
              type: 'upload',
              message: error,
            });
          } else {
            // Success case - update status and statistics
            symbolStatus.timeframes[timeframe].status = 'success';
            if (rows !== undefined) {
              symbolStatus.timeframes[timeframe].rows = rows;
              totalRowsUploaded += rows;
            }
            successfulUploads++;
          }
        }
      }

      else if (type === 'error') {
        if (symbol && timeframe) {
          const symbolStatus = symbolsMap.get(symbol);
          if (symbolStatus) {
            symbolStatus.timeframes[timeframe].status = 'failed';
            symbolStatus.timeframes[timeframe].error = error;
          }

          failedDownloads++;

          errors.push({
            timestamp: entry.timestamp,
            symbol,
            timeframe,
            type: error?.includes('다운로드') ? 'download' : 'unknown',
            message: error || 'Unknown error',
          });
        } else if (timeframe && error) {
          // Timeframe change error without symbol - use currentSymbol
          const errorSymbol = currentSymbol || 'UNKNOWN';

          // Mark the timeframe as failed for the current symbol
          if (currentSymbol) {
            const symbolStatus = symbolsMap.get(currentSymbol);
            if (symbolStatus) {
              symbolStatus.timeframes[timeframe].status = 'failed';
              symbolStatus.timeframes[timeframe].error = error;
            }
            failedDownloads++;
          }

          errors.push({
            timestamp: entry.timestamp,
            symbol: errorSymbol,
            timeframe,
            type: 'timeout',
            message: error,
          });
        }
      }
    }

    // Update symbol completion status
    const symbolsArray = Array.from(symbolsMap.values());
    for (const symbolStatus of symbolsArray) {
      const timeframeStatuses = Object.values(symbolStatus.timeframes).map(tf => tf.status);
      const allSuccess = timeframeStatuses.every(s => s === 'success');
      const anySuccess = timeframeStatuses.some(s => s === 'success');
      const anyFailed = timeframeStatuses.some(s => s === 'failed');

      if (allSuccess) {
        symbolStatus.status = 'completed';
        symbolStatus.completedAt = entries[entries.length - 1].timestamp;
      } else if (anyFailed && anySuccess) {
        symbolStatus.status = 'partial';
      } else if (anyFailed) {
        symbolStatus.status = 'failed';
      } else if (timeframeStatuses.some(s => s === 'downloading')) {
        symbolStatus.status = 'in_progress';
      }
    }

    // Calculate progress
    const symbols = symbolsArray;
    const completedSymbols = symbols.filter(s => s.status === 'completed').length;
    const totalSymbols = symbols.length;
    const percentage = totalSymbols > 0 ? Math.round((completedSymbols / totalSymbols) * 100) : 0;

    // Determine overall status
    const running = isScrapingRunning(entries);
    let status: JobStatus;
    if (running) {
      status = 'running';
    } else if (completedSymbols === totalSymbols) {
      status = 'completed';
    } else if (completedSymbols > 0) {
      status = 'partial';
    } else if (errors.length > 0) {
      status = 'error';
    } else {
      status = 'idle';
    }

    const lastRun = entries.length > 0 ? entries[entries.length - 1].timestamp : null;

    return {
      status,
      lastRun,
      currentSession: session,
      progress: {
        totalSymbols,
        completedSymbols,
        currentSymbol,
        currentTimeframe,
        percentage,
      },
      statistics: {
        totalDownloads,
        successfulUploads,
        failedDownloads,
        totalRowsUploaded,
      },
      symbols,
      errors: errors.slice(-50), // Keep last 50 errors
    };

  } catch (error) {
    // Handle file read errors gracefully
    console.error('Error parsing scraping log:', error);
    return createIdleStatus();
  }
}

// Helper to create idle status
function createIdleStatus(): ScrapingStatus {
  const symbols = SYMBOLS.map(symbol => createEmptySymbolStatus(symbol));

  return {
    status: 'idle',
    lastRun: null,
    currentSession: null,
    progress: {
      totalSymbols: SYMBOLS.length,
      completedSymbols: 0,
      currentSymbol: null,
      currentTimeframe: null,
      percentage: 0,
    },
    statistics: {
      totalDownloads: 0,
      successfulUploads: 0,
      failedDownloads: 0,
      totalRowsUploaded: 0,
    },
    symbols,
    errors: [],
  };
}
