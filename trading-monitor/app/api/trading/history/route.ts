import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

const TRADING_SERVICE_URL = process.env.TRADING_SERVICE_URL || 'http://localhost:8002';

interface RawPurchase {
  purchase_date: string;
  etf_code: string;
  quantity: number;
  price: number;
  total_amount: number;
  sold: boolean;
  sold_price?: number;
  sell_pnl?: number;
}

function transformToSummaries(raw: Record<string, unknown>) {
  const purchases = (raw.purchases as RawPurchase[]) || [];
  if (purchases.length === 0) return [];

  // Group by date
  const byDate: Record<string, { buys: RawPurchase[]; sells: RawPurchase[] }> = {};
  for (const p of purchases) {
    const date = p.purchase_date;
    if (!byDate[date]) byDate[date] = { buys: [], sells: [] };
    if (p.sold) {
      byDate[date].sells.push(p);
    } else {
      byDate[date].buys.push(p);
    }
  }

  return Object.entries(byDate).map(([date, { buys, sells }]) => ({
    date,
    buyCount: buys.length,
    sellCount: sells.length,
    totalProfitLoss: sells.reduce((sum, s) => sum + (s.sell_pnl || 0), 0),
    trades: [...buys, ...sells].map((p, i) => ({
      id: `${date}-${i}`,
      etfCode: p.etf_code,
      etfName: p.etf_code,
      side: p.sold ? 'SELL' : 'BUY',
      quantity: p.quantity,
      price: p.sold ? (p.sold_price || p.price) : p.price,
      executedAt: date,
      profitLoss: p.sell_pnl || undefined,
      profitLossPercent: p.sell_pnl && p.price > 0
        ? Number(((p.sell_pnl / (p.price * p.quantity)) * 100).toFixed(2))
        : undefined,
    })),
  })).sort((a, b) => b.date.localeCompare(a.date));
}

export async function GET() {
  try {
    const response = await fetch(`${TRADING_SERVICE_URL}/api/trading/history?page_size=1000`, {
      signal: AbortSignal.timeout(5000),
    });
    if (response.ok) {
      const data = await response.json();
      console.log('[BFF] trading/history: 실서비스 데이터 사용');
      return NextResponse.json(transformToSummaries(data));
    }
    throw new Error(`Trading service responded with ${response.status}`);
  } catch (error) {
    console.log('[BFF] trading/history: 연결 실패 -', error instanceof Error ? error.message : 'unknown');
    return NextResponse.json([]);
  }
}
