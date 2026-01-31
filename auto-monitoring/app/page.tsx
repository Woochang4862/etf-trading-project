'use client';

import { DashboardHeader } from '@/components/dashboard/dashboard-header';
import { StatsOverview } from '@/components/dashboard/stats-overview';
import { ScrapingStatus } from '@/components/dashboard/scraping-status';
import { TrainingStatus } from '@/components/dashboard/training-status';
import { PredictionStatus } from '@/components/dashboard/prediction-status';
import { SymbolGrid } from '@/components/dashboard/symbol-grid';
import { useScrapingStatus } from '@/hooks/use-scraping-status';
import { useTrainingStatus } from '@/hooks/use-training-status';
import { usePredictionStatus } from '@/hooks/use-prediction-status';

export default function DashboardPage() {
  const { data: scrapingData, isLoading: scrapingLoading, error: scrapingError } = useScrapingStatus();
  const { data: trainingData, isLoading: trainingLoading } = useTrainingStatus();
  const { data: predictionData, isLoading: predictionLoading } = usePredictionStatus();

  if (scrapingLoading || trainingLoading || predictionLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-muted/20 to-background">
        <div className="text-center space-y-4">
          <div className="relative w-16 h-16 mx-auto">
            <div className="absolute inset-0 border-4 border-foreground/10 rounded-full" />
            <div className="absolute inset-0 border-4 border-foreground border-t-transparent rounded-full animate-spin" />
          </div>
          <p className="text-sm font-medium text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (scrapingError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-muted/20 to-background">
        <div className="text-center space-y-4 p-8 rounded-xl border-2 border-red-500/20 bg-red-500/5">
          <div className="text-4xl">⚠️</div>
          <h2 className="text-xl font-bold">Error Loading Dashboard</h2>
          <p className="text-sm text-muted-foreground max-w-md">
            {scrapingError instanceof Error ? scrapingError.message : 'Failed to load dashboard data'}
          </p>
        </div>
      </div>
    );
  }

  if (!scrapingData || !trainingData || !predictionData) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-muted/10 to-background">
      <div className="container mx-auto px-4 py-8 space-y-8">
        {/* Header */}
        <DashboardHeader
          title="ETF Trading Pipeline Monitor"
          subtitle="Real-time monitoring for data scraping, model training, and predictions"
          lastUpdated={new Date().toISOString()}
        />

        {/* Stats Overview */}
        <StatsOverview
          progress={scrapingData.progress}
          statistics={scrapingData.statistics}
        />

        {/* Main Grid - Scraping, Training, Predictions */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <ScrapingStatus data={scrapingData} />
          <TrainingStatus data={trainingData} />
          <PredictionStatus data={predictionData} />
        </div>

        {/* Symbol Grid */}
        <SymbolGrid symbols={scrapingData.symbols} />

        {/* Footer */}
        <footer className="text-center text-xs text-muted-foreground py-8 border-t border-foreground/5">
          <p>ETF Trading Pipeline • Auto-refresh every 5 seconds</p>
          <p className="mt-1">Monitoring 101 stocks across 4 timeframes</p>
        </footer>
      </div>
    </div>
  );
}
