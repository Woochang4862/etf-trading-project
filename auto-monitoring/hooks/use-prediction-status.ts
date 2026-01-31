'use client';

import { useState, useEffect, useCallback } from 'react';
import { PredictionStatus } from '@/lib/types';
import { API_ENDPOINTS, REFRESH_INTERVALS } from '@/lib/constants';
import { useInterval } from './use-interval';

interface UsePredictionStatusResult {
  data: PredictionStatus | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function usePredictionStatus(): UsePredictionStatusResult {
  const [data, setData] = useState<PredictionStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(API_ENDPOINTS.PREDICTION_STATUS);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const status = await response.json();
      setData(status);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useInterval(fetchStatus, REFRESH_INTERVALS.PREDICTION);

  return { data, isLoading, error, refetch: fetchStatus };
}
