'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberTicker } from '@/components/ui/number-ticker';
import { ScrapingStatistics, ScrapingProgress } from '@/lib/types';
import { SYMBOLS } from '@/lib/constants';
import { HugeiconsIcon } from '@hugeicons/react';
import {
  Database02Icon,
  Download04Icon,
  File01Icon,
  AlertCircleIcon,
} from '@hugeicons/core-free-icons';

interface StatsOverviewProps {
  progress: ScrapingProgress;
  statistics: ScrapingStatistics;
}

export function StatsOverview({ progress, statistics }: StatsOverviewProps) {
  const stats = [
    {
      title: 'Total Symbols',
      value: SYMBOLS.length,
      iconData: Database02Icon,
      description: `${progress.completedSymbols} completed`,
      gradient: 'from-blue-500/10 to-cyan-500/10',
      iconColor: 'text-blue-500',
      glowColor: 'group-hover:shadow-blue-500/20',
    },
    {
      title: 'Downloads',
      value: statistics.totalDownloads,
      iconData: Download04Icon,
      description: `${statistics.successfulUploads} uploaded`,
      gradient: 'from-green-500/10 to-emerald-500/10',
      iconColor: 'text-green-500',
      glowColor: 'group-hover:shadow-green-500/20',
    },
    {
      title: 'Rows Uploaded',
      value: statistics.totalRowsUploaded,
      iconData: File01Icon,
      description: 'Total records in DB',
      gradient: 'from-purple-500/10 to-pink-500/10',
      iconColor: 'text-purple-500',
      glowColor: 'group-hover:shadow-purple-500/20',
    },
    {
      title: 'Errors',
      value: statistics.failedDownloads,
      iconData: AlertCircleIcon,
      description: 'Failed downloads',
      gradient:
        statistics.failedDownloads > 0
          ? 'from-red-500/10 to-orange-500/10'
          : 'from-gray-500/5 to-slate-500/5',
      iconColor: statistics.failedDownloads > 0 ? 'text-red-500' : 'text-gray-400',
      glowColor:
        statistics.failedDownloads > 0
          ? 'group-hover:shadow-red-500/20'
          : 'group-hover:shadow-gray-500/10',
    },
  ];

  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => (
        <Card
          key={stat.title}
          className={`group relative overflow-hidden border-0 bg-gradient-to-br ${stat.gradient} backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl ${stat.glowColor}`}
        >
          {/* Decorative corner accent */}
          <div className="absolute right-0 top-0 h-32 w-32 -translate-y-12 translate-x-12 rounded-full bg-gradient-to-br from-white/5 to-transparent blur-2xl transition-transform duration-500 group-hover:scale-150" />

          <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-semibold tracking-wide text-muted-foreground">
              {stat.title}
            </CardTitle>
            <div className={`rounded-xl bg-background/50 p-2.5 backdrop-blur-sm ring-1 ring-white/10 transition-all duration-300 group-hover:scale-110 group-hover:ring-2 ${stat.iconColor.replace('text-', 'group-hover:ring-')}/30`}>
              <HugeiconsIcon
                icon={stat.iconData}
                className={`h-5 w-5 ${stat.iconColor}`}
                strokeWidth={2}
              />
            </div>
          </CardHeader>

          <CardContent className="relative space-y-2">
            <div className={`text-4xl font-black tracking-tight ${stat.iconColor}`}>
              <NumberTicker
                value={stat.value}
                className="drop-shadow-lg"
              />
            </div>
            <p className="text-xs font-medium text-muted-foreground/80 tracking-wide">
              {stat.description}
            </p>
          </CardContent>

          {/* Bottom edge highlight */}
          <div className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${stat.gradient.replace(/\/10/g, '/40')} opacity-50 transition-opacity duration-300 group-hover:opacity-100`} />
        </Card>
      ))}
    </div>
  );
}
