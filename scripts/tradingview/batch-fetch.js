/**
 * TradingView API를 사용한 다중 종목 일괄 조회
 *
 * 사용법:
 *   node batch-fetch.js
 *   node batch-fetch.js --symbols "NASDAQ:AAPL,NASDAQ:GOOGL,NASDAQ:MSFT"
 *   node batch-fetch.js --file symbols.txt
 *   node batch-fetch.js --count 50 --timeframe 60
 */

import TradingView from '@mathieuc/tradingview';
import fs from 'fs';
import path from 'path';

// 기본 ETF/주식 심볼 목록
const DEFAULT_SYMBOLS = [
  'NASDAQ:AAPL',
  'NASDAQ:GOOGL',
  'NASDAQ:MSFT',
  'NASDAQ:AMZN',
  'NASDAQ:NVDA',
  'NASDAQ:META',
  'NASDAQ:TSLA',
  'NYSE:JPM',
  'NYSE:V',
  'NYSE:JNJ',
  // ETF
  'AMEX:SPY',
  'AMEX:QQQ',
  'AMEX:IWM',
  'AMEX:DIA',
  'AMEX:VTI',
];

// 인자 파싱
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    symbols: DEFAULT_SYMBOLS,
    count: 100,
    timeframe: 'D',
    output: 'data',
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--symbols':
        config.symbols = args[++i].split(',').map(s => s.trim());
        break;
      case '--file':
        const filePath = args[++i];
        const content = fs.readFileSync(filePath, 'utf-8');
        config.symbols = content.split('\n').map(s => s.trim()).filter(s => s && !s.startsWith('#'));
        break;
      case '--count':
        config.count = parseInt(args[++i]);
        break;
      case '--timeframe':
        config.timeframe = args[++i];
        break;
      case '--output':
        config.output = args[++i];
        break;
      case '--help':
        console.log(`
사용법: node batch-fetch.js [옵션]

옵션:
  --symbols "SYM1,SYM2"   쉼표로 구분된 심볼 목록
  --file <path>           심볼 목록 파일 (줄바꿈으로 구분)
  --count <n>             캔들 개수 (기본값: 100)
  --timeframe <tf>        타임프레임 (기본값: D)
                          1, 5, 15, 30, 60, 240 (분)
                          D (일), W (주), M (월)
  --output <dir>          출력 디렉토리 (기본값: data)
  --help                  도움말 표시

예시:
  node batch-fetch.js
  node batch-fetch.js --symbols "NASDAQ:AAPL,NASDAQ:GOOGL"
  node batch-fetch.js --file my-symbols.txt --count 200
`);
        process.exit(0);
    }
  }

  return config;
}

// 단일 심볼 데이터 조회
async function fetchSymbol(symbol, config) {
  return new Promise((resolve, reject) => {
    const client = new TradingView.Client();
    const chart = new client.Session.Chart();
    let resolved = false;

    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        chart.delete();
        client.end();
        reject(new Error('타임아웃'));
      }
    }, 15000);

    chart.onError((...err) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        chart.delete();
        client.end();
        reject(new Error(err.join(' ')));
      }
    });

    chart.setMarket(symbol, {
      timeframe: config.timeframe,
      range: config.count,
    });

    chart.onUpdate(() => {
      if (resolved) return;
      if (!chart.periods || chart.periods.length === 0) return;

      resolved = true;
      clearTimeout(timeout);

      const periods = chart.periods;
      const data = {
        symbol: symbol,
        timeframe: config.timeframe,
        fetchedAt: new Date().toISOString(),
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

      chart.delete();
      client.end();
      resolve(data);
    });
  });
}

// 메인 함수
async function main() {
  const config = parseArgs();

  console.log('='.repeat(60));
  console.log('  TradingView 다중 종목 일괄 조회');
  console.log('='.repeat(60));
  console.log(`  종목 수: ${config.symbols.length}`);
  console.log(`  캔들 개수: ${config.count}`);
  console.log(`  타임프레임: ${config.timeframe}`);
  console.log(`  출력 디렉토리: ${config.output}`);
  console.log('='.repeat(60));
  console.log('');

  // 출력 디렉토리 생성
  if (!fs.existsSync(config.output)) {
    fs.mkdirSync(config.output, { recursive: true });
  }

  const results = {
    success: [],
    failed: [],
  };

  // 순차적으로 데이터 조회 (동시 연결 제한)
  for (let i = 0; i < config.symbols.length; i++) {
    const symbol = config.symbols[i];
    const progress = `[${i + 1}/${config.symbols.length}]`;

    process.stdout.write(`${progress} ${symbol.padEnd(20)} ... `);

    try {
      const data = await fetchSymbol(symbol, config);

      // 파일 저장
      const filename = `${symbol.replace(':', '_')}_${config.timeframe}.json`;
      const outputPath = path.join(config.output, filename);
      fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));

      console.log(`OK (${data.candles.length}개 캔들)`);
      results.success.push({
        symbol,
        candles: data.candles.length,
        file: outputPath,
      });

    } catch (error) {
      console.log(`실패: ${error.message}`);
      results.failed.push({
        symbol,
        error: error.message,
      });
    }

    // API 부하 방지를 위한 딜레이
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  // 결과 요약
  console.log('');
  console.log('='.repeat(60));
  console.log('  결과 요약');
  console.log('='.repeat(60));
  console.log(`  성공: ${results.success.length}개`);
  console.log(`  실패: ${results.failed.length}개`);

  if (results.failed.length > 0) {
    console.log('');
    console.log('실패한 종목:');
    for (const item of results.failed) {
      console.log(`  - ${item.symbol}: ${item.error}`);
    }
  }

  // 요약 파일 저장
  const summaryPath = path.join(config.output, '_summary.json');
  fs.writeFileSync(summaryPath, JSON.stringify({
    fetchedAt: new Date().toISOString(),
    config: {
      timeframe: config.timeframe,
      candleCount: config.count,
    },
    results: results,
  }, null, 2));

  console.log('');
  console.log(`요약 파일: ${summaryPath}`);
}

main().catch(err => {
  console.error('에러:', err);
  process.exit(1);
});
