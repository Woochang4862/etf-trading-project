import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

const TRADING_SERVICE_URL = process.env.TRADING_SERVICE_URL || 'http://localhost:8002';

function transformPortfolio(raw: Record<string, unknown>) {
  const holdings = (raw.holdings as Array<Record<string, unknown>>) || [];
  const totalInvested = (raw.total_invested as number) || 0;

  const transformedHoldings = holdings.map((h) => ({
    etfCode: (h.etf_code as string) || '',
    etfName: (h.etf_code as string) || '',
    quantity: (h.quantity as number) || 0,
    buyPrice: (h.price as number) || 0,
    currentPrice: (h.price as number) || 0,
    buyDate: (h.purchase_date as string) || '',
    dDay: (h.trading_day_number as number) || 0,
    profitLoss: 0,
    profitLossPercent: 0,
  }));

  return {
    totalInvestment: totalInvested,
    totalCurrentValue: totalInvested,
    totalProfitLoss: 0,
    totalProfitLossPercent: 0,
    holdings: transformedHoldings,
  };
}

const EMPTY_PORTFOLIO = {
  totalInvestment: 0,
  totalCurrentValue: 0,
  totalProfitLoss: 0,
  totalProfitLossPercent: 0,
  holdings: [],
};

export async function GET() {
  try {
    const response = await fetch(`${TRADING_SERVICE_URL}/api/trading/portfolio`, {
      signal: AbortSignal.timeout(5000),
    });
    if (response.ok) {
      const data = await response.json();
      console.log('[BFF] trading/portfolio: 실서비스 데이터 사용');
      return NextResponse.json(transformPortfolio(data));
    }
    throw new Error(`Trading service responded with ${response.status}`);
  } catch (error) {
    console.log('[BFF] trading/portfolio: 연결 실패 -', error instanceof Error ? error.message : 'unknown');
    return NextResponse.json(EMPTY_PORTFOLIO);
  }
}
