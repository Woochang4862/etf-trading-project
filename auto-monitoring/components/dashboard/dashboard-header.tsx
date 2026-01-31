'use client';

import { ShimmerButton } from '@/components/ui/shimmer-button';
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
    <header className="relative overflow-hidden border-b-4 border-b-foreground/10 bg-gradient-to-br from-background via-muted/30 to-background">
      {/* Ambient glow effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-violet-500/5 to-emerald-500/5" />

      {/* Diagonal accent line */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-violet-500 to-emerald-500" />

      <div className="relative px-8 py-8">
        <div className="flex items-start justify-between gap-6">
          {/* Title section */}
          <div className="space-y-2">
            <h1 className="text-4xl font-black tracking-tight bg-gradient-to-r from-foreground via-foreground/90 to-foreground/70 bg-clip-text text-transparent">
              {title}
            </h1>
            <p className="text-base text-muted-foreground font-medium max-w-2xl">
              {subtitle}
            </p>
          </div>

          {/* Last updated timestamp */}
          <div className="flex flex-col items-end gap-1 px-4 py-3 rounded-lg bg-muted/30 border border-foreground/10">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Last Updated
            </div>
            <div className="text-sm font-bold tabular-nums">
              {formatLastUpdated(lastUpdated)}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom glow */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-foreground/20 to-transparent" />
    </header>
  );
}
