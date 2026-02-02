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
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <div className="w-8 h-8 mx-auto border-2 border-foreground/20 border-t-foreground rounded-full animate-spin" />
          <p className="text-sm text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (scrapingError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4 max-w-md p-8 rounded-lg border bg-card">
          <div className="text-4xl">⚠️</div>
          <h2 className="text-xl font-semibold">Error Loading Dashboard</h2>
          <p className="text-sm text-muted-foreground">
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
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 space-y-6">
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <ScrapingStatus data={scrapingData} />
          <TrainingStatus data={trainingData} />
          <PredictionStatus data={predictionData} />
        </div>

        {/* Symbol Grid */}
        <SymbolGrid symbols={scrapingData.symbols} totalDuration={scrapingData.totalDuration} />

        {/* Footer */}
        <footer className="text-center text-xs text-muted-foreground py-8 border-t">
          <p>ETF Trading Pipeline • Auto-refresh every 5 seconds</p>
          <p className="mt-1">Monitoring 101 stocks across 4 timeframes</p>
        </footer>
      </div>
    </div>
  );
}
