'use client';

import { useState, useEffect, useCallback } from 'react';
import { ScrapingStatus } from '@/lib/types';
import { API_ENDPOINTS, REFRESH_INTERVALS } from '@/lib/constants';
import { useInterval } from './use-interval';

interface UseScrapingStatusResult {
  data: ScrapingStatus | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  lastUpdated: Date | null;
}

export function useScrapingStatus(): UseScrapingStatusResult {
  const [data, setData] = useState<ScrapingStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(API_ENDPOINTS.SCRAPING_STATUS);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const status = await response.json();
      setData(status);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Polling every 5 seconds
  useInterval(fetchStatus, REFRESH_INTERVALS.SCRAPING);

  return {
    data,
    isLoading,
    error,
    refetch: fetchStatus,
    lastUpdated,
  };
}
