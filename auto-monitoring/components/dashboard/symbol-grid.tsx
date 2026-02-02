'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { SymbolScrapingStatus, TimeframeCode } from '@/lib/types';
import { TIMEFRAMES } from '@/lib/constants';
import { useState } from 'react';
import Link from 'next/link';
import { SymbolModal } from './symbol-modal';

interface SymbolGridProps {
  symbols: SymbolScrapingStatus[];
  totalDuration?: number;
}

export function SymbolGrid({ symbols, totalDuration }: SymbolGridProps) {
  const [filter, setFilter] = useState<'all' | 'in_progress' | 'completed' | 'partial' | 'failed'>('all');
  const [selectedSymbol, setSelectedSymbol] = useState<SymbolScrapingStatus | null>(null);

  const filteredSymbols = symbols.filter(s => {
    if (filter === 'all') return true;
    return s.status === filter;
  });

  const statusCounts = {
    pending: symbols.filter(s => s.status === 'pending').length,
    in_progress: symbols.filter(s => s.status === 'in_progress').length,
    completed: symbols.filter(s => s.status === 'completed').length,
    partial: symbols.filter(s => s.status === 'partial').length,
    failed: symbols.filter(s => s.status === 'failed').length,
  };

  const formatDuration = (seconds: number | undefined) => {
    if (!seconds) return '-';
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins < 60) return `${mins}m ${secs}s`;
    const hours = Math.floor(mins / 60);
    const remainMins = mins % 60;
    return `${hours}h ${remainMins}m`;
  };

  const handleRetry = (symbol: string) => {
    console.log(`Retry initiated for ${symbol}`);
  };

  return (
    <>
      <Card className="col-span-full">
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex items-center gap-4">
              <div>
                <CardTitle className="text-lg font-semibold">Symbol Status Grid</CardTitle>
                <CardDescription>
                  Click on a symbol to view details and logs
                  {totalDuration && (
                    <span className="ml-2">
                      • Total time: <span className="font-medium">{formatDuration(totalDuration)}</span>
                    </span>
                  )}
                </CardDescription>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Link
                href="/logs"
                className="px-3 py-1.5 bg-muted hover:bg-muted/80 rounded-lg text-xs font-medium transition flex items-center gap-1.5"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                View Full Logs
              </Link>
            </div>
          </div>

          {/* Filter Buttons */}
          <div className="flex flex-wrap gap-2 mt-4">
            <FilterButton
              label="All"
              count={symbols.length}
              active={filter === 'all'}
              onClick={() => setFilter('all')}
            />
            <FilterButton
              label="In Progress"
              count={statusCounts.in_progress}
              active={filter === 'in_progress'}
              onClick={() => setFilter('in_progress')}
            />
            <FilterButton
              label="Completed"
              count={statusCounts.completed}
              active={filter === 'completed'}
              onClick={() => setFilter('completed')}
            />
            <FilterButton
              label="Partial"
              count={statusCounts.partial}
              active={filter === 'partial'}
              onClick={() => setFilter('partial')}
            />
            <FilterButton
              label="Failed"
              count={statusCounts.failed}
              active={filter === 'failed'}
              onClick={() => setFilter('failed')}
            />
          </div>

          {/* Status Summary Bar */}
          <div className="flex items-center gap-2 text-xs mt-4">
            <StatusBar count={statusCounts.pending} total={symbols.length} color="bg-gray-400" label="Pending" />
            <StatusBar count={statusCounts.in_progress} total={symbols.length} color="bg-blue-500" label="Running" />
            <StatusBar count={statusCounts.completed} total={symbols.length} color="bg-green-500" label="Done" />
            <StatusBar count={statusCounts.partial} total={symbols.length} color="bg-yellow-500" label="Partial" />
            <StatusBar count={statusCounts.failed} total={symbols.length} color="bg-red-500" label="Failed" />
          </div>
        </CardHeader>

        <CardContent>
          <ScrollArea className="h-[600px] pr-4">
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
              {filteredSymbols.map((symbolData) => (
                <SymbolCard
                  key={symbolData.symbol}
                  data={symbolData}
                  onClick={() => setSelectedSymbol(symbolData)}
                />
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {selectedSymbol && (
        <SymbolModal
          symbol={selectedSymbol}
          isOpen={!!selectedSymbol}
          onClose={() => setSelectedSymbol(null)}
          onRetry={handleRetry}
        />
      )}
    </>
  );
}

function SymbolCard({ data, onClick }: { data: SymbolScrapingStatus; onClick: () => void }) {
  const statusConfig = {
    pending: {
      bg: 'bg-card',
      border: 'border',
      text: 'text-muted-foreground',
      badge: 'bg-gray-500',
    },
    in_progress: {
      bg: 'bg-blue-50 dark:bg-blue-950/30',
      border: 'border-blue-200 dark:border-blue-800',
      text: 'text-blue-700 dark:text-blue-300',
      badge: 'bg-blue-500',
    },
    completed: {
      bg: 'bg-green-50 dark:bg-green-950/30',
      border: 'border-green-200 dark:border-green-800',
      text: 'text-green-700 dark:text-green-300',
      badge: 'bg-green-500',
    },
    partial: {
      bg: 'bg-yellow-50 dark:bg-yellow-950/30',
      border: 'border-yellow-200 dark:border-yellow-800',
      text: 'text-yellow-700 dark:text-yellow-300',
      badge: 'bg-yellow-500',
    },
    failed: {
      bg: 'bg-red-50 dark:bg-red-950/30',
      border: 'border-red-200 dark:border-red-800',
      text: 'text-red-700 dark:text-red-300',
      badge: 'bg-red-500',
    },
  };

  const config = statusConfig[data.status];
  const completedTimeframes = TIMEFRAMES.filter(tf => data.timeframes[tf]?.status === 'success').length;

  const formatDuration = (seconds: number | undefined) => {
    if (!seconds) return null;
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  return (
    <div
      onClick={onClick}
      className={`p-3 rounded-lg border transition-all cursor-pointer hover:shadow-md ${config.bg} ${config.border}`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className={`font-semibold text-sm ${config.text}`}>{data.symbol}</div>
        <div className={`w-2 h-2 rounded-full ${config.badge} ${data.status === 'in_progress' ? 'animate-pulse' : ''}`} />
      </div>

      {/* Timeframe Progress */}
      <div className="grid grid-cols-4 gap-1 mb-2">
        {TIMEFRAMES.map((tf) => (
          <TimeframeIndicator
            key={tf}
            timeframe={tf}
            status={data.timeframes[tf]?.status || 'pending'}
          />
        ))}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>{completedTimeframes}/{TIMEFRAMES.length} done</span>
        {data.duration && (
          <span className="font-medium">{formatDuration(data.duration)}</span>
        )}
      </div>
    </div>
  );
}

function TimeframeIndicator({ timeframe, status }: { timeframe: TimeframeCode; status: string }) {
  const statusColors = {
    pending: 'bg-muted',
    downloading: 'bg-blue-500 animate-pulse',
    success: 'bg-green-500',
    failed: 'bg-red-500',
  };

  const label = timeframe.replace('달', '').replace('주', 'W').replace('일', 'D');

  return (
    <div
      className={`h-6 rounded flex items-center justify-center text-[9px] font-bold text-white ${statusColors[status as keyof typeof statusColors] || statusColors.pending}`}
      title={`${timeframe}: ${status}`}
    >
      {label}
    </div>
  );
}

function FilterButton({
  label,
  count,
  active,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
        active
          ? 'bg-primary text-primary-foreground'
          : 'bg-muted hover:bg-muted/80 text-muted-foreground'
      }`}
    >
      {label} <span className="font-semibold">({count})</span>
    </button>
  );
}

function StatusBar({
  count,
  total,
  color,
  label
}: {
  count: number;
  total: number;
  color: string;
  label: string;
}) {
  const percentage = (count / total) * 100;

  return (
    <div className="flex-1">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] font-medium text-muted-foreground">{label}</span>
        <span className="text-[10px] font-semibold tabular-nums">{count}</span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
