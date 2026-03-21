'use client';

import { useState, useEffect, useCallback } from 'react';
import { DBStatsOverview } from '@/components/db-viewer/db-stats-overview';
import { DBTableGrid } from '@/components/db-viewer/db-table-grid';
import { DBTableDetail } from '@/components/db-viewer/db-table-detail';
import { HelpTooltip } from '@/components/ui/help-tooltip';
import type { DBOverview } from '@/lib/types';

const DB_HELP_ITEMS = [
  {
    color: 'bg-green-500/30 border border-green-500/50',
    label: '초록색 셀 (최신)',
    description: '3일 이내 데이터가 있는 테이블. 정상적으로 수집되고 있음.',
  },
  {
    color: 'bg-red-500/30 border border-red-500/50',
    label: '빨간색 셀 (지연)',
    description: '3일 이상 데이터 없음. 스크래핑 실패했거나 해당 종목이 상장폐지됨.',
  },
  {
    color: 'bg-primary/30 border border-primary/50',
    label: '파란 테두리 (선택됨)',
    description: '클릭하면 오른쪽에 테이블 실제 데이터가 표시됨.',
  },
  {
    label: '데이터 건강도',
    description: '(최신 테이블 수 / 전체 테이블 수) × 100. 90% 이상이면 초록, 70~90%이면 노랑, 70% 미만이면 빨강.',
  },
  {
    label: 'etf2_db (원본)',
    description: 'TradingView에서 스크래핑한 OHLCV + RSI/MACD 원본 데이터. 종목별 × 타임프레임별 테이블.',
  },
  {
    label: 'etf2_db_processed (피처)',
    description: '85개 기술지표 + 거시경제 피처가 계산된 ML 학습용 DB. 일봉(D) 데이터만 존재.',
  },
  {
    label: '타임프레임 필터',
    description: 'D=일봉, W=주봉, M=월봉, 1h=1시간봉, 10m=10분봉, 12M=12개월(장기)',
  },
];

export default function DBViewerPage() {
  const [activeDB, setActiveDB] = useState<'etf2_db' | 'etf2_db_processed'>('etf2_db');
  const [data, setData] = useState<DBOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');
  const [selectedTable, setSelectedTable] = useState<string | null>(null);

  const fetchTables = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/trading/api/db/tables?db_name=${activeDB}`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch (e) {
      console.error('Failed to fetch tables:', e);
    } finally {
      setLoading(false);
    }
  }, [activeDB]);

  useEffect(() => {
    fetchTables();
    setSelectedTable(null);
  }, [fetchTables]);

  const filteredTables = data?.tables.filter(t =>
    filter ? t.symbol.toLowerCase().includes(filter.toLowerCase()) ||
             t.tableName.toLowerCase().includes(filter.toLowerCase()) : true
  ) ?? [];

  return (
    <div className="space-y-6">
      {/* 상단 컨트롤 */}
      <div className="flex items-center gap-4">
        <div className="flex rounded-lg border border-border overflow-hidden">
          <button
            onClick={() => setActiveDB('etf2_db')}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeDB === 'etf2_db'
                ? 'bg-primary text-primary-foreground'
                : 'bg-background text-muted-foreground hover:bg-muted'
            }`}
          >
            etf2_db (원본)
          </button>
          <button
            onClick={() => setActiveDB('etf2_db_processed')}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeDB === 'etf2_db_processed'
                ? 'bg-primary text-primary-foreground'
                : 'bg-background text-muted-foreground hover:bg-muted'
            }`}
          >
            etf2_db_processed (피처)
          </button>
        </div>

        <input
          type="text"
          placeholder="종목 검색 (AAPL, NVDA...)"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="flex h-9 w-64 rounded-md border border-input bg-background px-3 py-1 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />

        <button onClick={fetchTables} className="text-xs text-primary hover:underline">
          새로고침
        </button>

        {loading && <span className="text-xs text-muted-foreground">로딩 중...</span>}

        <div className="ml-auto">
          <HelpTooltip title="DB 뷰어 가이드" items={DB_HELP_ITEMS} />
        </div>
      </div>

      {data && <DBStatsOverview overview={data} />}

      <div className="grid gap-6 lg:grid-cols-2">
        <DBTableGrid
          tables={filteredTables}
          onSelectTable={setSelectedTable}
          selectedTable={selectedTable}
        />
        {selectedTable && (
          <DBTableDetail
            tableName={selectedTable}
            dbName={activeDB}
            onClose={() => setSelectedTable(null)}
          />
        )}
      </div>
    </div>
  );
}
