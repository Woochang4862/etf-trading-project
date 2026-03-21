'use client';

import { useState, useEffect } from 'react';
import { PipelineTimeline } from '@/components/pipeline/pipeline-timeline';
import { PipelineOverview } from '@/components/pipeline/pipeline-overview';
import { ScheduleEditor } from '@/components/pipeline/schedule-editor';
import type { PipelineStatus, ScheduleConfig } from '@/lib/types';

function buildDefaultPipeline(): PipelineStatus {
  const now = new Date().toISOString();
  return {
    isRunning: false,
    currentStep: null,
    steps: [
      { id: 'scraping', name: '데이터 수집', description: 'TradingView 스크래핑', scheduledTime: '06:00', status: 'idle', lastRunAt: null, lastRunDuration: null, lastRunMessage: null, nextRunAt: null },
      { id: 'feature', name: '피처 처리', description: '85개 피처 생성', scheduledTime: '07:00', status: 'idle', lastRunAt: null, lastRunDuration: null, lastRunMessage: null, nextRunAt: null },
      { id: 'prediction', name: 'ML 예측', description: 'LightGBM 랭킹', scheduledTime: '07:30', status: 'idle', lastRunAt: null, lastRunDuration: null, lastRunMessage: null, nextRunAt: null },
      { id: 'trading', name: '매매 실행', description: 'KIS API 주문', scheduledTime: '23:30', status: 'idle', lastRunAt: null, lastRunDuration: null, lastRunMessage: null, nextRunAt: null },
    ],
    lastFullRunAt: null,
    lastFullRunSuccess: false,
  };
}

function buildDefaultSchedule(): ScheduleConfig {
  return {
    scraping: '06:00',
    featureEngineering: '07:00',
    prediction: '07:30',
    tradeDecision: '08:00',
    kisOrder: '23:30',
    monthlyRetrain: '03:00',
  };
}

export default function PipelinePage() {
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [schedule, setSchedule] = useState<ScheduleConfig | null>(null);

  useEffect(() => {
    setPipeline(buildDefaultPipeline());
    setSchedule(buildDefaultSchedule());
  }, []);

  if (!pipeline || !schedule) return null;

  return (
    <div className="space-y-6">
      <PipelineOverview pipeline={pipeline} />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <PipelineTimeline steps={pipeline.steps} />
        </div>
        <div>
          <ScheduleEditor schedule={schedule} onSave={setSchedule} />
        </div>
      </div>
    </div>
  );
}
