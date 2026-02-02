'use client';

import { Badge } from '@/components/ui/badge';

interface DashboardHeaderProps {
  title: string;
  subtitle: string;
  lastUpdated: string;
}

export function DashboardHeader({
  title,
  subtitle,
  lastUpdated
}: DashboardHeaderProps) {
  const formatLastUpdated = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSeconds = Math.floor(diffMs / 1000);
    const diffMinutes = Math.floor(diffSeconds / 60);

    if (diffSeconds < 60) {
      return `${diffSeconds}s ago`;
    } else if (diffMinutes < 60) {
      return `${diffMinutes}m ago`;
    } else {
      return date.toLocaleTimeString();
    }
  };

  return (
    <header className="border-b bg-card">
      <div className="px-6 py-6">
        <div className="flex items-start justify-between gap-6">
          {/* Title section */}
          <div className="space-y-1">
            <h1 className="text-2xl font-semibold tracking-tight">
              {title}
            </h1>
            <p className="text-sm text-muted-foreground">
              {subtitle}
            </p>
          </div>

          {/* Last updated timestamp */}
          <Badge variant="secondary" className="text-xs">
            Updated {formatLastUpdated(lastUpdated)}
          </Badge>
        </div>
      </div>
    </header>
  );
}
