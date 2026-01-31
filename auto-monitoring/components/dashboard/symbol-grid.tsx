'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { SymbolScrapingStatus, TimeframeCode } from '@/lib/types';
import { TIMEFRAMES } from '@/lib/constants';
import { useState } from 'react';

interface SymbolGridProps {
  symbols: SymbolScrapingStatus[];
}

export function SymbolGrid({ symbols }: SymbolGridProps) {
  const [filter, setFilter] = useState<'all' | 'in_progress' | 'completed' | 'failed'>('all');

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

  return (
    <Card className="col-span-full">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <CardTitle className="text-xl font-bold">Symbol Status Grid</CardTitle>
            <CardDescription>Real-time scraping status for all 101 stocks</CardDescription>
          </div>

          {/* Filter Buttons */}
          <div className="flex flex-wrap gap-2">
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
              color="bg-blue-500"
            />
            <FilterButton
              label="Completed"
              count={statusCounts.completed}
              active={filter === 'completed'}
              onClick={() => setFilter('completed')}
              color="bg-emerald-500"
            />
            <FilterButton
              label="Failed"
              count={statusCounts.failed}
              active={filter === 'failed'}
              onClick={() => setFilter('failed')}
              color="bg-red-500"
            />
          </div>
        </div>

        {/* Status Summary Bar */}
        <div className="flex items-center gap-2 text-xs">
          <StatusBar count={statusCounts.pending} total={symbols.length} color="bg-gray-400" label="Pending" />
          <StatusBar count={statusCounts.in_progress} total={symbols.length} color="bg-blue-500" label="Running" />
          <StatusBar count={statusCounts.completed} total={symbols.length} color="bg-emerald-500" label="Done" />
          <StatusBar count={statusCounts.partial} total={symbols.length} color="bg-yellow-500" label="Partial" />
          <StatusBar count={statusCounts.failed} total={symbols.length} color="bg-red-500" label="Failed" />
        </div>
      </CardHeader>

      <CardContent>
        <ScrollArea className="h-[600px] pr-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {filteredSymbols.map((symbolData) => (
              <SymbolCard key={symbolData.symbol} data={symbolData} />
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function SymbolCard({ data }: { data: SymbolScrapingStatus }) {
  const statusConfig = {
    pending: {
      bg: 'bg-gray-100 dark:bg-gray-800',
      border: 'border-gray-300 dark:border-gray-700',
      text: 'text-gray-600 dark:text-gray-400',
      badge: 'bg-gray-500',
    },
    in_progress: {
      bg: 'bg-blue-50 dark:bg-blue-950',
      border: 'border-blue-300 dark:border-blue-700',
      text: 'text-blue-700 dark:text-blue-300',
      badge: 'bg-blue-500',
    },
    completed: {
      bg: 'bg-emerald-50 dark:bg-emerald-950',
      border: 'border-emerald-300 dark:border-emerald-700',
      text: 'text-emerald-700 dark:text-emerald-300',
      badge: 'bg-emerald-500',
    },
    partial: {
      bg: 'bg-yellow-50 dark:bg-yellow-950',
      border: 'border-yellow-300 dark:border-yellow-700',
      text: 'text-yellow-700 dark:text-yellow-300',
      badge: 'bg-yellow-500',
    },
    failed: {
      bg: 'bg-red-50 dark:bg-red-950',
      border: 'border-red-300 dark:border-red-700',
      text: 'text-red-700 dark:text-red-300',
      badge: 'bg-red-500',
    },
  };

  const config = statusConfig[data.status];
  const completedTimeframes = TIMEFRAMES.filter(tf => data.timeframes[tf]?.status === 'success').length;

  return (
    <div className={`p-3 rounded-lg border-2 transition-all hover:shadow-lg ${config.bg} ${config.border}`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className={`font-bold text-sm ${config.text}`}>{data.symbol}</div>
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
      <div className="text-[10px] text-muted-foreground">
        {completedTimeframes}/{TIMEFRAMES.length} timeframes
      </div>
    </div>
  );
}

function TimeframeIndicator({ timeframe, status }: { timeframe: TimeframeCode; status: string }) {
  const statusColors = {
    pending: 'bg-gray-200 dark:bg-gray-700',
    downloading: 'bg-blue-400 animate-pulse',
    success: 'bg-emerald-500',
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
  color = 'bg-foreground'
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
  color?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        px-3 py-1.5 rounded-lg text-xs font-medium transition-all
        ${active
          ? 'bg-foreground text-background'
          : 'bg-muted hover:bg-muted/80 text-muted-foreground'
        }
      `}
    >
      {label} <span className="font-bold">({count})</span>
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
        <span className="text-[10px] font-bold tabular-nums">{count}</span>
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
