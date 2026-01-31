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
    <Card className="border-l-4 border-l-violet-500">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-xl">
              Predictions
              {isRunning && (
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-violet-500"></span>
                </span>
              )}
            </CardTitle>
            <CardDescription>Daily RSI/MACD signal generation</CardDescription>
          </div>
          <Badge variant={getPredictionStatusVariant(status)} className="uppercase text-xs font-bold">
            {status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Schedule Info */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-muted/30 border border-foreground/5">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
              Last Run
            </div>
            <div className="text-sm font-medium">
              {lastPrediction ? new Date(lastPrediction).toLocaleDateString() : 'Never'}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-violet-500/5 border border-violet-500/20">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
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
            color="bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400"
          />
          <SignalSummary
            label="Hold"
            count={summary.holdSignals}
            color="bg-blue-500/10 border-blue-500/30 text-blue-600 dark:text-blue-400"
          />
          <SignalSummary
            label="Sell"
            count={summary.sellSignals}
            color="bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-400"
          />
        </div>

        {/* Top Signals */}
        <div className="space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
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
    <div className={`p-3 rounded-lg border text-center ${color}`}>
      <div className="text-2xl font-bold tabular-nums">{count}</div>
      <div className="text-[10px] font-bold uppercase tracking-wider">{label}</div>
    </div>
  );
}

function SignalRow({ signal }: { signal: PredictionStatusType['topSignals'][0] }) {
  const signalColors = {
    BUY: 'bg-emerald-500/10 border-emerald-500/20',
    HOLD: 'bg-blue-500/10 border-blue-500/20',
    SELL: 'bg-red-500/10 border-red-500/20',
  };

  const signalBadgeColors = {
    BUY: 'bg-emerald-500 text-white',
    HOLD: 'bg-blue-500 text-white',
    SELL: 'bg-red-500 text-white',
  };

  return (
    <div className={`p-2 rounded-md border flex items-center justify-between hover:shadow-sm transition-all ${signalColors[signal.signal]}`}>
      <div className="flex items-center gap-2">
        <span className="font-bold text-sm">{signal.symbol}</span>
        <Badge className={`text-[10px] px-2 uppercase font-bold ${signalBadgeColors[signal.signal]}`}>
          {signal.signal}
        </Badge>
      </div>
      <div className="flex items-center gap-3 text-xs">
        <div className="text-muted-foreground">
          Confidence: <span className="font-bold text-foreground">{(signal.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="w-px h-4 bg-foreground/10" />
        <div className="text-muted-foreground">
          RSI: <span className="font-bold text-foreground">{signal.rsi.toFixed(1)}</span>
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
