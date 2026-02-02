'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { PredictionStatus as PredictionStatusType } from '@/lib/types';

interface PredictionStatusProps {
  data: PredictionStatusType;
}

export function PredictionStatus({ data }: PredictionStatusProps) {
  const { status, lastPrediction, nextScheduled, summary, topSignals } = data;
  const isRunning = status === 'running';

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-lg">
              Predictions
              {isRunning && (
                <span className="w-2 h-2 rounded-full bg-violet-500 animate-pulse" />
              )}
            </CardTitle>
            <CardDescription className="text-sm">Daily RSI/MACD signal generation</CardDescription>
          </div>
          <Badge variant={getPredictionStatusVariant(status)} className="uppercase text-xs">
            {status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Schedule Info */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-muted/50 border">
            <div className="text-xs font-medium text-muted-foreground mb-1">
              Last Run
            </div>
            <div className="text-sm font-medium">
              {lastPrediction ? new Date(lastPrediction).toLocaleDateString() : 'Never'}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-violet-500/10 border border-violet-200 dark:border-violet-800">
            <div className="text-xs font-medium text-muted-foreground mb-1">
              Next Scheduled
            </div>
            <div className="text-sm font-medium text-violet-700 dark:text-violet-400">
              {new Date(nextScheduled).toLocaleDateString()}
            </div>
          </div>
        </div>

        {/* Summary Grid */}
        <div className="grid grid-cols-3 gap-2">
          <SignalSummary
            label="Buy"
            count={summary.buySignals}
            color="text-green-700 dark:text-green-400"
          />
          <SignalSummary
            label="Hold"
            count={summary.holdSignals}
            color="text-blue-700 dark:text-blue-400"
          />
          <SignalSummary
            label="Sell"
            count={summary.sellSignals}
            color="text-red-700 dark:text-red-400"
          />
        </div>

        {/* Top Signals */}
        <div className="space-y-2">
          <h4 className="text-xs font-medium text-muted-foreground uppercase">
            Top Signals (Last 5)
          </h4>
          <div className="space-y-1">
            {topSignals.slice(0, 5).map((signal, i) => (
              <SignalRow key={i} signal={signal} />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SignalSummary({
  label,
  count,
  color
}: {
  label: string;
  count: number;
  color: string;
}) {
  return (
    <div className="p-3 rounded-lg border bg-card text-center">
      <div className={`text-xl font-semibold tabular-nums ${color}`}>{count}</div>
      <div className="text-[10px] font-medium uppercase text-muted-foreground">{label}</div>
    </div>
  );
}

function SignalRow({ signal }: { signal: PredictionStatusType['topSignals'][0] }) {
  const signalColors = {
    BUY: 'bg-green-500/10 border-green-200 dark:border-green-800',
    HOLD: 'bg-blue-500/10 border-blue-200 dark:border-blue-800',
    SELL: 'bg-red-500/10 border-red-200 dark:border-red-800',
  };

  const signalBadgeColors = {
    BUY: 'bg-green-500 text-white',
    HOLD: 'bg-blue-500 text-white',
    SELL: 'bg-red-500 text-white',
  };

  return (
    <div className={`p-2 rounded-md border flex items-center justify-between ${signalColors[signal.signal]}`}>
      <div className="flex items-center gap-2">
        <span className="font-medium text-sm">{signal.symbol}</span>
        <Badge className={`text-[10px] px-2 uppercase ${signalBadgeColors[signal.signal]}`}>
          {signal.signal}
        </Badge>
      </div>
      <div className="flex items-center gap-3 text-xs">
        <div className="text-muted-foreground">
          Confidence: <span className="font-medium text-foreground">{(signal.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="w-px h-4 bg-border" />
        <div className="text-muted-foreground">
          RSI: <span className="font-medium text-foreground">{signal.rsi.toFixed(1)}</span>
        </div>
      </div>
    </div>
  );
}

function getPredictionStatusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case 'running': return 'default';
    case 'completed': return 'secondary';
    default: return 'outline';
  }
}
