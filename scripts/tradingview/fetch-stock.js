/**
 * TradingView API를 사용한 실시간 주식 데이터 조회
 *
 * 사용법:
 *   node fetch-stock.js NASDAQ:AAPL
 *   node fetch-stock.js BINANCE:BTCUSDT
 *   node fetch-stock.js KRX:005930
 */

import TradingView from '@mathieuc/tradingview';

const symbol = process.argv[2] || 'NASDAQ:AAPL';
const timeframe = process.argv[3] || 'D'; // D=일봉, 60=1시간, 240=4시간

console.log('='.repeat(60));
console.log('  TradingView 실시간 주식 데이터 조회');
console.log('='.repeat(60));
console.log(`  심볼: ${symbol}`);
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
  range: 100, // 최근 100개 캔들
});

// 에러 핸들링
chart.onError((...err) => {
  console.error('차트 에러:', ...err);
});

// 심볼 로드 완료
chart.onSymbolLoaded(() => {
  console.log(`[로드 완료] ${symbol}`);
  console.log('');
});

// 데이터 업데이트
let updateCount = 0;
chart.onUpdate(() => {
  updateCount++;

  if (!chart.periods || chart.periods.length === 0) {
    console.log('데이터 로딩 중...');
    return;
  }

  const periods = chart.periods;
  const latest = periods[periods.length - 1];

  console.log(`[업데이트 #${updateCount}] ${new Date().toLocaleString()}`);
  console.log('-'.repeat(60));

  // 최신 캔들 데이터 출력
  console.log('최신 캔들:');
  console.log(`  시간: ${new Date(latest.time * 1000).toLocaleString()}`);
  console.log(`  시가(Open):  ${latest.open?.toFixed(2) || 'N/A'}`);
  console.log(`  고가(High):  ${latest.high?.toFixed(2) || 'N/A'}`);
  console.log(`  저가(Low):   ${latest.low?.toFixed(2) || 'N/A'}`);
  console.log(`  종가(Close): ${latest.close?.toFixed(2) || 'N/A'}`);
  console.log(`  거래량:      ${latest.volume?.toLocaleString() || 'N/A'}`);

  // 가격 변동 계산
  if (periods.length >= 2) {
    const prev = periods[periods.length - 2];
    const change = latest.close - prev.close;
    const changePercent = (change / prev.close * 100).toFixed(2);
    const sign = change >= 0 ? '+' : '';
    console.log(`  변동:        ${sign}${change.toFixed(2)} (${sign}${changePercent}%)`);
  }

  console.log('');

  // 최근 5개 캔들 요약
  if (updateCount === 1 && periods.length >= 5) {
    console.log('최근 5개 캔들:');
    console.log('-'.repeat(80));
    console.log('시간                        시가       고가       저가       종가       거래량');
    console.log('-'.repeat(80));

    const recentPeriods = periods.slice(-5);
    for (const p of recentPeriods) {
      const time = new Date(p.time * 1000).toLocaleString().padEnd(20);
      const open = (p.open?.toFixed(2) || 'N/A').padStart(10);
      const high = (p.high?.toFixed(2) || 'N/A').padStart(10);
      const low = (p.low?.toFixed(2) || 'N/A').padStart(10);
      const close = (p.close?.toFixed(2) || 'N/A').padStart(10);
      const volume = (p.volume?.toLocaleString() || 'N/A').padStart(12);
      console.log(`${time} ${open} ${high} ${low} ${close} ${volume}`);
    }
    console.log('-'.repeat(80));
    console.log('');
  }
});

// 30초 후 종료
setTimeout(() => {
  console.log('연결 종료...');
  chart.delete();
  client.end();
  process.exit(0);
}, 30000);

// Ctrl+C 핸들링
process.on('SIGINT', () => {
  console.log('\n사용자에 의해 중단됨');
  chart.delete();
  client.end();
  process.exit(0);
});

console.log('실시간 데이터 스트리밍 중... (30초 후 자동 종료, Ctrl+C로 중단)');
console.log('');
