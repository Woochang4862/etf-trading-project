'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface JobStatus {
  status: string;
  message?: string;
  progress?: number;
  totalSymbols?: number;
  completedSymbols?: number;
}

interface PipelineStep {
  id: string;
  label: string;
  description: string;
  schedule: string;
  startUrl: string;
  stopUrl?: string;
  statusUrl: string;
  status: 'idle' | 'running' | 'completed' | 'error';
  message: string;
}

export function AutomationControl() {
  const [steps, setSteps] = useState<PipelineStep[]>([
    {
      id: 'scraping',
      label: '데이터 수집',
      description: 'TradingView 스크래핑 (101종목)',
      schedule: '매일 06:00 KST',
      startUrl: '/trading/api/scraper/start',
      stopUrl: '/trading/api/scraper/stop',
      statusUrl: '/trading/api/scraper/status',
      status: 'idle',
      message: '대기 중',
    },
    {
      id: 'features',
      label: '데이터 정제',
      description: '85개 피처 엔지니어링',
      schedule: '수집 완료 후 자동',
      startUrl: '/trading/api/features/start',
      statusUrl: '/trading/api/features/status',
      status: 'idle',
      message: '대기 중',
    },
    {
      id: 'prediction',
      label: 'ML 예측',
      description: 'LightGBM 랭킹 예측',
      schedule: '정제 완료 후 자동',
      startUrl: '/trading/api/ml/predict',
      statusUrl: '/trading/api/ml/ranking',
      status: 'idle',
      message: '대기 중',
    },
    {
      id: 'trading',
      label: '매매 실행',
      description: 'KIS API 자동 주문',
      schedule: '23:30 KST (APScheduler)',
      startUrl: '/trading/api/trading/automation',
      statusUrl: '/trading/api/trading/automation',
      status: 'idle',
      message: '대기 중',
    },
  ]);

  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [tradingEnabled, setTradingEnabled] = useState(false);

  // Poll statuses
  const fetchStatuses = useCallback(async () => {
    try {
      const [scrapRes, featRes, autoRes] = await Promise.allSettled([
        fetch('/trading/api/scraper/status'),
        fetch('/trading/api/features/status'),
        fetch('/trading/api/trading/automation'),
      ]);

      setSteps(prev => {
        const updated = [...prev];

        // Scraping status
        if (scrapRes.status === 'fulfilled' && scrapRes.value.ok) {
          const d = scrapRes.value.json().then(data => {
            setSteps(p => p.map(s => s.id === 'scraping' ? {
              ...s,
              status: mapStatus(data.status),
              message: data.status === 'running'
                ? `${data.completedSymbols || 0}/${data.totalSymbols || 101} 진행 중`
                : data.status === 'completed' ? '수집 완료' : '대기 중',
            } : s));
          });
        }

        // Feature status
        if (featRes.status === 'fulfilled' && featRes.value.ok) {
          featRes.value.json().then(data => {
            setSteps(p => p.map(s => s.id === 'features' ? {
              ...s,
              status: mapStatus(data.status),
              message: data.status === 'running'
                ? `${data.progress || 0}/${data.total || 101} 처리 중`
                : data.status === 'completed' ? '정제 완료' : data.message || '대기 중',
            } : s));
          });
        }

        // Trading automation
        if (autoRes.status === 'fulfilled' && autoRes.value.ok) {
          autoRes.value.json().then(data => {
            setTradingEnabled(data.enabled);
            setSteps(p => p.map(s => s.id === 'trading' ? {
              ...s,
              status: data.enabled ? 'running' : 'idle',
              message: data.enabled ? `활성 (${data.scheduler_time})` : '비활성',
            } : s));
          });
        }

        return updated;
      });
    } catch {
      // silent
    }
  }, []);

  useEffect(() => {
    fetchStatuses();
    const interval = setInterval(fetchStatuses, 10000);
    return () => clearInterval(interval);
  }, [fetchStatuses]);

  async function handleStart(step: PipelineStep) {
    setLoading(prev => ({ ...prev, [step.id]: true }));
    try {
      if (step.id === 'trading') {
        // Trading은 automation toggle
        await fetch(step.startUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: true, fractional_mode: false }),
        });
      } else {
        await fetch(step.startUrl, { method: 'POST' });
      }
      await fetchStatuses();
    } catch {
      // silent
    } finally {
      setLoading(prev => ({ ...prev, [step.id]: false }));
    }
  }

  async function handleStop(step: PipelineStep) {
    setLoading(prev => ({ ...prev, [step.id]: true }));
    try {
      if (step.id === 'trading') {
        await fetch(step.startUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: false }),
        });
      } else if (step.stopUrl) {
        await fetch(step.stopUrl, { method: 'POST' });
      }
      await fetchStatuses();
    } catch {
      // silent
    } finally {
      setLoading(prev => ({ ...prev, [step.id]: false }));
    }
  }

  async function handleRunAll() {
    // 순차 실행: 수집 → 정제 → 예측
    for (const step of steps.filter(s => s.id !== 'trading')) {
      setLoading(prev => ({ ...prev, [step.id]: true }));
      try {
        if (step.id === 'trading') continue;
        await fetch(step.startUrl, { method: 'POST' });
      } catch {
        // continue
      } finally {
        setLoading(prev => ({ ...prev, [step.id]: false }));
      }
    }
    await fetchStatuses();
  }

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base flex items-center gap-2">
          파이프라인 관리
          <Badge variant="outline" className="text-[10px]">자동화</Badge>
        </CardTitle>
        <button
          onClick={handleRunAll}
          className="text-xs bg-cyan-600 text-white rounded px-3 py-1.5 hover:bg-cyan-700 transition-colors"
        >
          전체 실행 (수집→정제→예측)
        </button>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {steps.map((step, idx) => (
            <div key={step.id} className="flex items-center gap-3 rounded-md border border-border p-3">
              {/* Step number + indicator */}
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-xs text-muted-foreground font-mono w-4">{idx + 1}</span>
                <span className={`h-2.5 w-2.5 rounded-full flex-shrink-0 ${
                  step.status === 'running' ? 'bg-blue-500 animate-pulse'
                  : step.status === 'completed' ? 'bg-green-500'
                  : step.status === 'error' ? 'bg-red-500'
                  : 'bg-zinc-600'
                }`} />
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{step.label}</span>
                  <span className="text-[10px] text-muted-foreground">{step.description}</span>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px] text-muted-foreground">{step.schedule}</span>
                  <span className="text-[10px] text-muted-foreground">·</span>
                  <span className={`text-[10px] font-medium ${
                    step.status === 'running' ? 'text-blue-400'
                    : step.status === 'completed' ? 'text-green-400'
                    : step.status === 'error' ? 'text-red-400'
                    : 'text-zinc-500'
                  }`}>{step.message}</span>
                </div>
              </div>

              {/* Controls */}
              <div className="flex items-center gap-1.5 flex-shrink-0">
                <button
                  onClick={() => handleStart(step)}
                  disabled={loading[step.id] || step.status === 'running'}
                  className={`px-2.5 py-1 text-xs rounded transition-colors ${
                    step.status === 'running'
                      ? 'bg-blue-600/20 text-blue-400 cursor-default'
                      : 'bg-green-600/20 text-green-400 hover:bg-green-600 hover:text-white'
                  } disabled:opacity-50`}
                >
                  {loading[step.id] ? '...' : step.status === 'running' ? '실행 중' : 'Start'}
                </button>
                {(step.stopUrl || step.id === 'trading') && (
                  <button
                    onClick={() => handleStop(step)}
                    disabled={loading[step.id] || (step.status !== 'running' && !(step.id === 'trading' && tradingEnabled))}
                    className="px-2.5 py-1 text-xs rounded bg-red-600/20 text-red-400 hover:bg-red-600 hover:text-white transition-colors disabled:opacity-30"
                  >
                    Stop
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function mapStatus(s: string): 'idle' | 'running' | 'completed' | 'error' {
  switch (s) {
    case 'running': case 'pending': return 'running';
    case 'completed': return 'completed';
    case 'failed': case 'error': case 'stopped': return 'error';
    default: return 'idle';
  }
}
