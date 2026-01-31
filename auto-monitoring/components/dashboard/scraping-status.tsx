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
    <Card className="col-span-full lg:col-span-2 border-2 border-foreground/10 bg-gradient-to-br from-background via-background to-muted/20">
      <CardHeader className="space-y-4">
        <div className="flex items-start justify-between">
          <div className="space-y-1.5">
            <CardTitle className="flex items-center gap-3 text-2xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                Data Scraping Pipeline
              </span>
              {isRunning && (
                <div className="relative flex h-4 w-4">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-4 w-4 bg-emerald-500 shadow-lg shadow-emerald-500/50"></span>
                </div>
              )}
            </CardTitle>
            <CardDescription className="text-base font-medium">
              Real-time TradingView data collection across 101 stocks
            </CardDescription>
          </div>
          <Badge
            variant={getStatusVariant(status)}
            className="text-sm px-4 py-1.5 font-bold uppercase tracking-wider"
          >
            {status}
          </Badge>
        </div>

        {/* Statistics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
          <StatCard
            label="Downloads"
            value={statistics.totalDownloads}
            color="bg-blue-500/10 border-blue-500/20"
            textColor="text-blue-600 dark:text-blue-400"
          />
          <StatCard
            label="Successful"
            value={statistics.successfulUploads}
            color="bg-emerald-500/10 border-emerald-500/20"
            textColor="text-emerald-600 dark:text-emerald-400"
          />
          <StatCard
            label="Failed"
            value={statistics.failedDownloads}
            color="bg-red-500/10 border-red-500/20"
            textColor="text-red-600 dark:text-red-400"
          />
          <StatCard
            label="Total Rows"
            value={statistics.totalRowsUploaded.toLocaleString()}
            color="bg-violet-500/10 border-violet-500/20"
            textColor="text-violet-600 dark:text-violet-400"
          />
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Progress Section */}
        <div className="space-y-3 p-4 rounded-lg bg-muted/30 border border-foreground/5">
          <div className="flex justify-between items-baseline">
            <span className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Progress</span>
            <span className="text-2xl font-bold tabular-nums bg-gradient-to-r from-foreground to-foreground/60 bg-clip-text text-transparent">
              {progress.percentage.toFixed(1)}%
            </span>
          </div>

          <div className="relative">
            <Progress value={progress.percentage} className="h-3" />
            <div
              className="absolute top-0 left-0 h-3 bg-gradient-to-r from-blue-500 via-violet-500 to-emerald-500 opacity-20 blur-sm transition-all"
              style={{ width: `${progress.percentage}%` }}
            />
          </div>

          {progress.currentSymbol && (
            <div className="flex items-center gap-2 pt-1">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-background border border-foreground/10">
                <span className="text-xs font-medium text-muted-foreground">Currently:</span>
                <span className="text-sm font-bold text-foreground">{progress.currentSymbol}</span>
              </div>
              <div className="h-1 w-1 rounded-full bg-foreground/30" />
              <span className="text-xs font-medium text-muted-foreground">
                {TIMEFRAME_LABELS[progress.currentTimeframe || '12달']}
              </span>
            </div>
          )}

          <div className="grid grid-cols-2 gap-2 text-xs pt-2">
            <div className="flex justify-between px-2">
              <span className="text-muted-foreground">Completed:</span>
              <span className="font-bold tabular-nums">{progress.completedSymbols}</span>
            </div>
            <div className="flex justify-between px-2">
              <span className="text-muted-foreground">Remaining:</span>
              <span className="font-bold tabular-nums">{progress.totalSymbols - progress.completedSymbols}</span>
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
            <SessionBadge
              icon="💾"
              label="DB Upload"
              active={currentSession.dbUploadEnabled}
            />
            <SessionBadge
              icon="🔐"
              label="SSH Tunnel"
              active={currentSession.sshTunnelActive}
            />
            <div className="text-xs text-muted-foreground px-3 py-1.5 rounded-md bg-muted/30 border border-foreground/5">
              Started: {new Date(currentSession.startTime).toLocaleTimeString()}
            </div>
          </div>
        )}

        {/* Recent Errors */}
        {errors.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-bold uppercase tracking-wide flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                Recent Errors
              </h4>
              <Badge variant="destructive" className="text-xs">
                {errors.length}
              </Badge>
            </div>
            <ScrollArea className="h-[120px] rounded-lg border border-red-500/20 bg-red-500/5">
              <div className="p-2 space-y-1">
                {errors.slice(-5).reverse().map((error, i) => (
                  <div
                    key={i}
                    className="text-xs p-3 bg-background/80 backdrop-blur-sm rounded-md border border-red-500/20 hover:border-red-500/40 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <span className="font-bold text-red-600 dark:text-red-400">{error.symbol}</span>
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

function StatCard({
  label,
  value,
  color,
  textColor
}: {
  label: string;
  value: string | number;
  color: string;
  textColor: string;
}) {
  return (
    <div className={`p-3 rounded-lg border ${color} backdrop-blur-sm`}>
      <div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-1">
        {label}
      </div>
      <div className={`text-xl font-bold tabular-nums ${textColor}`}>
        {value}
      </div>
    </div>
  );
}

function SessionBadge({
  icon,
  label,
  active
}: {
  icon: string;
  label: string;
  active: boolean;
}) {
  return (
    <div className={`
      px-3 py-1.5 rounded-md border text-xs font-medium transition-all
      ${active
        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-300'
        : 'bg-muted/30 border-foreground/10 text-muted-foreground'
      }
    `}>
      <span className="mr-1.5">{icon}</span>
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
