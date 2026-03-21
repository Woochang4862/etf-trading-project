import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

const TRADING_SERVICE_URL = process.env.TRADING_SERVICE_URL || 'http://localhost:8002';

function transformStatus(raw: Record<string, unknown>) {
  const cycle = raw.cycle as Record<string, unknown> | null;
  const today = new Date().toISOString().split('T')[0];

  return {
    mode: raw.trading_mode || 'paper',
    cycle: cycle
      ? {
          currentDay: cycle.current_day_number || 0,
          totalDays: 63,
          cycleType: 'long' as const,
          shortCycleDays: 15,
          longCycleDays: 63,
          startDate: cycle.cycle_start_date || today,
          nextRebalanceDate: '',
        }
      : {
          currentDay: 0,
          totalDays: 63,
          cycleType: 'long' as const,
          shortCycleDays: 15,
          longCycleDays: 63,
          startDate: today,
          nextRebalanceDate: '',
        },
    totalInvestment: (raw.total_invested as number) || 0,
    holdingsCount: (raw.total_holdings as number) || 0,
    todayBuyCount: 0,
    todaySellCount: 0,
    automationStatus: {
      lastRun: null,
      success: true,
      message: (raw.is_trading_day as boolean) ? '거래일' : `다음 거래일: ${raw.next_trading_day || '-'}`,
    },
    automationEnabled: (raw.automation_enabled as boolean) || false,
    fractionalMode: (raw.fractional_mode as boolean) || false,
  };
}

const EMPTY_STATUS = {
  mode: 'paper',
  cycle: { currentDay: 0, totalDays: 63, cycleType: 'long', shortCycleDays: 15, longCycleDays: 63, startDate: new Date().toISOString().split('T')[0], nextRebalanceDate: '' },
  totalInvestment: 0,
  holdingsCount: 0,
  todayBuyCount: 0,
  todaySellCount: 0,
  automationStatus: { lastRun: null, success: false, message: 'trading-service 연결 실패' },
  automationEnabled: false,
  fractionalMode: false,
};

export async function GET() {
  try {
    const response = await fetch(`${TRADING_SERVICE_URL}/api/trading/status`, {
      signal: AbortSignal.timeout(5000),
    });
    if (response.ok) {
      const data = await response.json();
      console.log('[BFF] trading/status: 실서비스 데이터 사용');
      return NextResponse.json(transformStatus(data));
    }
    throw new Error(`Trading service responded with ${response.status}`);
  } catch (error) {
    console.log('[BFF] trading/status: 연결 실패 -', error instanceof Error ? error.message : 'unknown');
    return NextResponse.json(EMPTY_STATUS);
  }
}
