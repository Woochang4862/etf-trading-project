/**
 * TradingView API를 사용한 과거 주식 데이터 조회
 *
 * 사용법:
 *   node fetch-history.js NASDAQ:AAPL 100
 *   node fetch-history.js BINANCE:BTCUSDT 50 240
 *   node fetch-history.js KRX:005930 200 D
 *
 * 인자:
 *   1. 심볼 (기본값: NASDAQ:AAPL)
 *   2. 캔들 개수 (기본값: 100)
 *   3. 타임프레임 (기본값: D)
 *      - 1, 5, 15, 30, 60, 240 (분)
 *      - D (일), W (주), M (월)
 */

import TradingView from '@mathieuc/tradingview';
import fs from 'fs';
import path from 'path';

const symbol = process.argv[2] || 'NASDAQ:AAPL';
const candleCount = parseInt(process.argv[3]) || 100;
const timeframe = process.argv[4] || 'D';

console.log('='.repeat(60));
console.log('  TradingView 과거 데이터 조회');
console.log('='.repeat(60));
console.log(`  심볼: ${symbol}`);
console.log(`  캔들 개수: ${candleCount}`);
console.log(`  타임프레임: ${timeframe}`);
console.log('='.repeat(60));
console.log('');

// TradingView WebSocket 클라이언트 생성
const client = new TradingView.Client();

// 차트 세션 생성
const chart = new client.Session.Chart();

// 심볼 설정
chart.setMarket(symbol, {
  timeframe: timeframe,
  range: candleCount,
});

// 에러 핸들링
chart.onError((...err) => {
  console.error('에러:', ...err);
  client.end();
  process.exit(1);
});

// 심볼 로드 완료
chart.onSymbolLoaded(() => {
  console.log(`[로드 완료] ${symbol}`);
});

// 데이터 수신
let dataReceived = false;

chart.onUpdate(() => {
  if (dataReceived) return;
  if (!chart.periods || chart.periods.length === 0) return;

  dataReceived = true;
  const periods = chart.periods;

  console.log(`\n총 ${periods.length}개의 캔들 데이터 수신`);
  console.log('');

  // 통계 계산
  const closes = periods.map(p => p.close).filter(c => c != null);
  const highs = periods.map(p => p.high).filter(h => h != null);
  const lows = periods.map(p => p.low).filter(l => l != null);
  const volumes = periods.map(p => p.volume).filter(v => v != null);

  const avgClose = closes.reduce((a, b) => a + b, 0) / closes.length;
  const maxHigh = Math.max(...highs);
  const minLow = Math.min(...lows);
  const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;

  // 기간 정보
  const firstTime = new Date(periods[0].time * 1000);
  const lastTime = new Date(periods[periods.length - 1].time * 1000);

  console.log('데이터 요약');
  console.log('-'.repeat(40));
  console.log(`기간: ${firstTime.toLocaleDateString()} ~ ${lastTime.toLocaleDateString()}`);
  console.log(`평균 종가: ${avgClose.toFixed(2)}`);
  console.log(`최고가: ${maxHigh.toFixed(2)}`);
  console.log(`최저가: ${minLow.toFixed(2)}`);
  console.log(`평균 거래량: ${Math.round(avgVolume).toLocaleString()}`);
  console.log('');

  // 테이블 출력
  console.log('캔들 데이터:');
  console.log('='.repeat(100));
  console.log(
    '날짜'.padEnd(12) +
    '시가'.padStart(12) +
    '고가'.padStart(12) +
    '저가'.padStart(12) +
    '종가'.padStart(12) +
    '거래량'.padStart(15) +
    '변동률'.padStart(10)
  );
  console.log('='.repeat(100));

  for (let i = 0; i < periods.length; i++) {
    const p = periods[i];
    const date = new Date(p.time * 1000).toLocaleDateString();
    const open = p.open?.toFixed(2) || '-';
    const high = p.high?.toFixed(2) || '-';
    const low = p.low?.toFixed(2) || '-';
    const close = p.close?.toFixed(2) || '-';
    const volume = p.volume?.toLocaleString() || '-';

    let changePercent = '-';
    if (i > 0 && periods[i - 1].close && p.close) {
      const change = ((p.close - periods[i - 1].close) / periods[i - 1].close * 100);
      changePercent = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
    }

    console.log(
      date.padEnd(12) +
      open.padStart(12) +
      high.padStart(12) +
      low.padStart(12) +
      close.padStart(12) +
      volume.padStart(15) +
      changePercent.padStart(10)
    );
  }
  console.log('='.repeat(100));

  // JSON 파일로 저장
  const outputDir = path.join(process.cwd(), 'data');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const filename = `${symbol.replace(':', '_')}_${timeframe}_${candleCount}.json`;
  const outputPath = path.join(outputDir, filename);

  const outputData = {
    symbol: symbol,
    timeframe: timeframe,
    fetchedAt: new Date().toISOString(),
    summary: {
      period: {
        start: firstTime.toISOString(),
        end: lastTime.toISOString(),
      },
      avgClose: avgClose,
      maxHigh: maxHigh,
      minLow: minLow,
      avgVolume: avgVolume,
      totalCandles: periods.length,
    },
    candles: periods.map(p => ({
      time: new Date(p.time * 1000).toISOString(),
      timestamp: p.time,
      open: p.open,
      high: p.high,
      low: p.low,
      close: p.close,
      volume: p.volume,
    })),
  };

  fs.writeFileSync(outputPath, JSON.stringify(outputData, null, 2));
  console.log(`\n데이터가 저장되었습니다: ${outputPath}`);

  // 종료
  chart.delete();
  client.end();
  process.exit(0);
});

// 타임아웃
setTimeout(() => {
  console.error('타임아웃: 데이터를 받지 못했습니다.');
  chart.delete();
  client.end();
  process.exit(1);
}, 30000);
