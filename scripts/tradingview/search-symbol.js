/**
 * TradingView 심볼 검색
 *
 * 사용법:
 *   node search-symbol.js AAPL
 *   node search-symbol.js 삼성전자
 *   node search-symbol.js Bitcoin
 */

import TradingView from '@mathieuc/tradingview';

const query = process.argv[2] || 'AAPL';

console.log('='.repeat(60));
console.log('  TradingView 심볼 검색');
console.log('='.repeat(60));
console.log(`  검색어: ${query}`);
console.log('='.repeat(60));
console.log('');

async function searchSymbol(searchQuery) {
  try {
    const results = await TradingView.searchMarket(searchQuery);

    if (!results || results.length === 0) {
      console.log('검색 결과가 없습니다.');
      return;
    }

    console.log(`${results.length}개의 결과를 찾았습니다:`);
    console.log('');
    console.log('='.repeat(90));
    console.log(
      '심볼'.padEnd(25) +
      '이름'.padEnd(35) +
      '거래소'.padEnd(15) +
      '타입'.padEnd(15)
    );
    console.log('='.repeat(90));

    for (const result of results.slice(0, 20)) { // 상위 20개만 표시
      const symbol = (result.symbol || '-').padEnd(25);
      const description = (result.description || '-').substring(0, 33).padEnd(35);
      const exchange = (result.exchange || '-').padEnd(15);
      const type = (result.type || '-').padEnd(15);

      console.log(`${symbol}${description}${exchange}${type}`);
    }

    console.log('='.repeat(90));
    console.log('');

    // 사용 예시 출력
    if (results.length > 0) {
      const firstResult = results[0];
      const fullSymbol = `${firstResult.exchange}:${firstResult.symbol}`;
      console.log('사용 예시:');
      console.log(`  node fetch-stock.js ${fullSymbol}`);
      console.log(`  node fetch-history.js ${fullSymbol} 100 D`);
    }

  } catch (error) {
    console.error('검색 중 에러 발생:', error.message);
    process.exit(1);
  }
}

searchSymbol(query);
