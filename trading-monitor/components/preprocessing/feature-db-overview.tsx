'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { DBOverview } from '@/lib/types';

interface FeatureDBOverviewProps {
  title: string;
  overview: DBOverview | null;
  description: string;
}

export function FeatureDBOverview({ title, overview, description }: FeatureDBOverviewProps) {
  if (!overview) {
    return (
      <Card className="shadow-sm">
        <CardHeader><CardTitle className="text-base">{title}</CardTitle></CardHeader>
        <CardContent><p className="text-sm text-muted-foreground">로딩 중...</p></CardContent>
      </Card>
    );
  }

  const healthPercent = overview.totalTables > 0
    ? Math.round((overview.upToDateTables / overview.totalTables) * 100)
    : 0;

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-xs text-muted-foreground">테이블</p>
            <p className="text-xl font-bold">{overview.totalTables.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">총 행 수</p>
            <p className="text-xl font-bold">{overview.totalRows.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">최신</p>
            <p className="text-xl font-bold text-green-500">{overview.upToDateTables}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">지연</p>
            <p className="text-xl font-bold text-red-500">{overview.staleTables}</p>
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">건강도</span>
            <span className={`font-bold ${healthPercent >= 90 ? 'text-green-500' : healthPercent >= 70 ? 'text-yellow-500' : 'text-red-500'}`}>
              {healthPercent}%
            </span>
          </div>
          <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${healthPercent >= 90 ? 'bg-green-500' : healthPercent >= 70 ? 'bg-yellow-500' : 'bg-red-500'}`}
              style={{ width: `${healthPercent}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
