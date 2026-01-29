/**
 * TradingView API 인증을 사용한 실시간 주식 데이터 조회
 *
 * 환경변수 설정 필요:
 *   export TV_SESSION="your_sessionid"
 *   export TV_SIGNATURE="your_signature"
 *
 * 세션/시그니처 가져오는 방법:
 *   1. TradingView 로그인 (유료 계정 권장)
 *   2. 브라우저 개발자도구 > Application > Cookies
 *   3. sessionid, sessionid_sign 값 복사
 *
 * 사용법:
 *   node fetch-stock-auth.js KRX:005930
 *   node fetch-stock-auth.js NASDAQ:AAPL D
 */

import TradingView from '@mathieuc/tradingview';

const symbol = process.argv[2] || 'KRX:005930';
const timeframe = process.argv[3] || 'D';

// 환경변수에서 인증 정보 가져오기
const session = process.env.TV_SESSION;
const signature = process.env.TV_SIGNATURE;

console.log('='.repeat(60));
console.log('  TradingView 인증 데이터 조회');
console.log('='.repeat(60));
console.log(`  심볼: ${symbol}`);
console.log(`  타임프레임: ${timeframe}`);
console.log(`  인증: ${session ? '설정됨' : '미설정 (지연 데이터)'}`);
console.log('='.repeat(60));
console.log('');

if (!session) {
  console.log('경고: TV_SESSION 환경변수가 설정되지 않았습니다.');
  console.log('      한국 주식 등 일부 데이터가 지연될 수 있습니다.');
  console.log('');
  console.log('인증 설정 방법:');
  console.log('  1. TradingView.com 로그인');
  console.log('  2. 브라우저 개발자도구 > Application > Cookies');
  console.log('  3. sessionid, sessionid_sign 값 복사');
  console.log('  4. export TV_SESSION="sessionid값"');
  console.log('     export TV_SIGNATURE="sessionid_sign값"');
  console.log('');
}

// TradingView 클라이언트 생성 (인증 포함)
const clientOptions = session ? {
  token: session,
  signature: signature,
} : {};

const client = new TradingView.Client(clientOptions);
const chart = new client.Session.Chart();

chart.setMarket(symbol, {
  timeframe: timeframe,
  range: 100,
});

chart.onError((...err) => {
  console.error('차트 에러:', ...err);
});

chart.onSymbolLoaded(() => {
  console.log(`[로드 완료] ${symbol}`);
  console.log('');
});

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

  console.log('최신 캔들:');
  console.log(`  시간: ${new Date(latest.time * 1000).toLocaleString()}`);
  console.log(`  시가(Open):  ${latest.open?.toLocaleString() || 'N/A'}`);
  console.log(`  종가(Close): ${latest.close?.toLocaleString() || 'N/A'}`);
  console.log(`  거래량:      ${latest.volume?.toLocaleString() || 'N/A'}`);

  if (periods.length >= 2) {
    const prev = periods[periods.length - 2];
    const change = latest.close - prev.close;
    const changePercent = (change / prev.close * 100).toFixed(2);
    const sign = change >= 0 ? '+' : '';
    console.log(`  변동:        ${sign}${change.toLocaleString()} (${sign}${changePercent}%)`);
  }

  // 데이터 날짜 경고
  const dataDate = new Date(latest.time * 1000);
  const now = new Date();
  const daysDiff = Math.floor((now - dataDate) / (1000 * 60 * 60 * 24));

  if (daysDiff > 7) {
    console.log('');
    console.log(`  ⚠️  데이터가 ${daysDiff}일 전입니다. 인증이 필요할 수 있습니다.`);
  }

  console.log('');

  if (updateCount === 1 && periods.length >= 5) {
    console.log('최근 5개 캔들:');
    console.log('-'.repeat(70));

    const recentPeriods = periods.slice(-5);
    for (const p of recentPeriods) {
      const time = new Date(p.time * 1000).toLocaleDateString();
      const open = p.open?.toLocaleString() || 'N/A';
      const close = p.close?.toLocaleString() || 'N/A';
      const volume = p.volume?.toLocaleString() || 'N/A';
      console.log(`  ${time.padEnd(12)} | 시가: ${open.padStart(10)} | 종가: ${close.padStart(10)} | 거래량: ${volume.padStart(15)}`);
    }
    console.log('-'.repeat(70));
    console.log('');
  }
});

setTimeout(() => {
  console.log('연결 종료...');
  chart.delete();
  client.end();
  process.exit(0);
}, 30000);

process.on('SIGINT', () => {
  console.log('\n사용자에 의해 중단됨');
  chart.delete();
  client.end();
  process.exit(0);
});

console.log('실시간 데이터 스트리밍 중... (30초 후 자동 종료)');
console.log('');
