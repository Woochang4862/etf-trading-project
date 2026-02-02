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
      iconColor: 'text-blue-600',
    },
    {
      title: 'Downloads',
      value: statistics.totalDownloads,
      iconData: Download04Icon,
      description: `${statistics.successfulUploads} uploaded`,
      iconColor: 'text-green-600',
    },
    {
      title: 'Rows Uploaded',
      value: statistics.totalRowsUploaded,
      iconData: File01Icon,
      description: 'Total records in DB',
      iconColor: 'text-purple-600',
    },
    {
      title: 'Errors',
      value: statistics.failedDownloads,
      iconData: AlertCircleIcon,
      description: 'Failed downloads',
      iconColor: statistics.failedDownloads > 0 ? 'text-red-600' : 'text-muted-foreground',
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => (
        <Card key={stat.title} className="shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {stat.title}
            </CardTitle>
            <HugeiconsIcon
              icon={stat.iconData}
              className={`h-4 w-4 ${stat.iconColor}`}
              strokeWidth={2}
            />
          </CardHeader>

          <CardContent className="space-y-1">
            <div className="text-2xl font-semibold">
              <NumberTicker value={stat.value} />
            </div>
            <p className="text-xs text-muted-foreground">
              {stat.description}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
