'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ScrapingStatus as ScrapingStatusType } from '@/lib/types';
import { TIMEFRAME_LABELS } from '@/lib/constants';

interface ScrapingStatusProps {
  data: ScrapingStatusType;
}

export function ScrapingStatus({ data }: ScrapingStatusProps) {
  const { status, currentSession, progress, statistics, errors } = data;
  const isRunning = status === 'running';

  return (
    <Card className="col-span-full lg:col-span-2">
      <CardHeader className="space-y-3">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-lg">
              Data Scraping Pipeline
              {isRunning && (
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
              )}
            </CardTitle>
            <CardDescription className="text-sm">
              Real-time TradingView data collection across 101 stocks
            </CardDescription>
          </div>
          <Badge variant={getStatusVariant(status)} className="text-xs uppercase">
            {status}
          </Badge>
        </div>

        {/* Statistics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2">
          <StatCard label="Downloads" value={statistics.totalDownloads} />
          <StatCard label="Successful" value={statistics.successfulUploads} />
          <StatCard label="Failed" value={statistics.failedDownloads} />
          <StatCard label="Total Rows" value={statistics.totalRowsUploaded.toLocaleString()} />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Progress Section */}
        <div className="space-y-2 p-3 rounded-lg bg-muted/50 border">
          <div className="flex justify-between items-baseline">
            <span className="text-xs font-medium text-muted-foreground">Progress</span>
            <span className="text-lg font-semibold tabular-nums">
              {progress.percentage.toFixed(1)}%
            </span>
          </div>

          <Progress value={progress.percentage} className="h-2" />

          {progress.currentSymbol && (
            <div className="flex items-center gap-2 pt-1">
              <div className="flex items-center gap-2 px-2 py-1 rounded bg-background border text-xs">
                <span className="text-muted-foreground">Currently:</span>
                <span className="font-medium">{progress.currentSymbol}</span>
              </div>
              <span className="text-muted-foreground">•</span>
              <span className="text-xs text-muted-foreground">
                {TIMEFRAME_LABELS[progress.currentTimeframe || '12달']}
              </span>
            </div>
          )}

          <div className="grid grid-cols-2 gap-2 text-xs pt-1">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Completed:</span>
              <span className="font-medium tabular-nums">{progress.completedSymbols}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Remaining:</span>
              <span className="font-medium tabular-nums">{progress.totalSymbols - progress.completedSymbols}</span>
            </div>
          </div>
        </div>

        {/* Session Info */}
        {currentSession && (
          <div className="flex flex-wrap gap-2">
            <SessionBadge
              icon="🖥️"
              label={currentSession.headlessMode ? 'Headless Mode' : 'Browser Mode'}
              active={currentSession.headlessMode}
            />
            <SessionBadge icon="💾" label="DB Upload" active={currentSession.dbUploadEnabled} />
            <SessionBadge icon="🔐" label="SSH Tunnel" active={currentSession.sshTunnelActive} />
            <div className="text-xs text-muted-foreground px-2 py-1 rounded bg-muted/50 border">
              Started: {new Date(currentSession.startTime).toLocaleTimeString()}
            </div>
          </div>
        )}

        {/* Recent Errors */}
        {errors.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-500" />
                Recent Errors
              </h4>
              <Badge variant="destructive" className="text-xs">
                {errors.length}
              </Badge>
            </div>
            <ScrollArea className="h-[120px] rounded-lg border">
              <div className="p-2 space-y-1">
                {errors.slice(-5).reverse().map((error, i) => (
                  <div key={i} className="text-xs p-2 bg-muted/50 rounded border">
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <span className="font-medium text-red-600 dark:text-red-400">{error.symbol}</span>
                      <span className="text-[10px] text-muted-foreground uppercase">{error.type}</span>
                    </div>
                    <div className="flex items-center gap-2 text-[11px] text-muted-foreground mb-1">
                      <span>{error.timeframe}</span>
                      <span>•</span>
                      <span>{new Date(error.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <p className="text-[11px] text-foreground/80 line-clamp-2">{error.message}</p>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="p-2 rounded-lg border bg-card">
      <div className="text-[10px] font-medium text-muted-foreground uppercase mb-1">
        {label}
      </div>
      <div className="text-lg font-semibold tabular-nums">
        {value}
      </div>
    </div>
  );
}

function SessionBadge({ icon, label, active }: { icon: string; label: string; active: boolean }) {
  return (
    <div
      className={`px-2 py-1 rounded border text-xs ${
        active
          ? 'bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-800 text-green-700 dark:text-green-300'
          : 'bg-muted/50 border-muted text-muted-foreground'
      }`}
    >
      <span className="mr-1">{icon}</span>
      {label}
    </div>
  );
}

function getStatusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case 'running': return 'default';
    case 'completed': return 'secondary';
    case 'error': return 'destructive';
    default: return 'outline';
  }
}
