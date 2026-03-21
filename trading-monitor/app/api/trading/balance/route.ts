import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const TRADING_SERVICE_URL = process.env.TRADING_SERVICE_URL || 'http://localhost:8002';

export async function GET() {
  try {
    const response = await fetch(`${TRADING_SERVICE_URL}/api/trading/balance`, {
      signal: AbortSignal.timeout(10000),
    });
    if (response.ok) {
      const data = await response.json();
      console.log('[BFF] trading/balance: 실서비스 데이터 사용');
      return NextResponse.json(data);
    }
    throw new Error(`Trading service responded with ${response.status}`);
  } catch (error) {
    console.log('[BFF] trading/balance: 폴백 -', error instanceof Error ? error.message : 'unknown');
    return NextResponse.json({
      available_cash_usd: 0,
      total_evaluation_usd: 0,
      available_cash_krw: 0,
      total_evaluation_krw: 0,
      exchange_rate: 1350,
      holdings: [],
      kis_connected: false,
      error: 'trading-service 연결 실패',
    });
  }
}
