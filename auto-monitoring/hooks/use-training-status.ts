'use client';

import { useState, useEffect, useCallback } from 'react';
import { TrainingStatus } from '@/lib/types';
import { API_ENDPOINTS, REFRESH_INTERVALS } from '@/lib/constants';
import { useInterval } from './use-interval';

interface UseTrainingStatusResult {
  data: TrainingStatus | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function useTrainingStatus(): UseTrainingStatusResult {
  const [data, setData] = useState<TrainingStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(API_ENDPOINTS.TRAINING_STATUS);
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

  useInterval(fetchStatus, REFRESH_INTERVALS.TRAINING);

  return { data, isLoading, error, refetch: fetchStatus };
}
