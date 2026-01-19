// 예측 결과 타입
export interface Prediction {
  symbol: string
  name: string
  signal: "BUY" | "SELL" | "HOLD"
  confidence: number
  rsi: number
  macd: number
  currentPrice: number
  predictedChange: number
  updatedAt: string
}

// 포트폴리오 항목 타입
export interface PortfolioItem {
  symbol: string
  name: string
  quantity: number
  avgPrice: number
  currentPrice: number
  totalValue: number
  profit: number
  profitPercent: number
}

// 수익률 데이터 타입
export interface ReturnData {
  date: string
  portfolioValue: number
  dailyReturn: number
  cumulativeReturn: number
  benchmarkReturn: number
  benchmarkCumulativeReturn: number
}

// 더미 예측 데이터 (실제로는 FastAPI에서 가져올 예정)
export const predictions: Prediction[] = [
  {
    symbol: "AAPL",
    name: "Apple Inc.",
    signal: "BUY",
    confidence: 78,
    rsi: 28.5,
    macd: 1.23,
    currentPrice: 178.50,
    predictedChange: 2.5,
    updatedAt: "2026-01-04 09:00:00",
  },
  {
    symbol: "NVDA",
    name: "NVIDIA Corp.",
    signal: "HOLD",
    confidence: 65,
    rsi: 52.3,
    macd: 0.45,
    currentPrice: 485.20,
    predictedChange: 0.8,
    updatedAt: "2026-01-04 09:00:00",
  },
  {
    symbol: "MSFT",
    name: "Microsoft Corp.",
    signal: "SELL",
    confidence: 72,
    rsi: 74.2,
    macd: -0.89,
    currentPrice: 378.90,
    predictedChange: -1.8,
    updatedAt: "2026-01-04 09:00:00",
  },
  {
    symbol: "GOOGL",
    name: "Alphabet Inc.",
    signal: "BUY",
    confidence: 81,
    rsi: 32.1,
    macd: 2.15,
    currentPrice: 141.20,
    predictedChange: 3.2,
    updatedAt: "2026-01-04 09:00:00",
  },
  {
    symbol: "AMZN",
    name: "Amazon.com Inc.",
    signal: "HOLD",
    confidence: 58,
    rsi: 48.7,
    macd: 0.12,
    currentPrice: 153.80,
    predictedChange: 0.5,
    updatedAt: "2026-01-04 09:00:00",
  },
  {
    symbol: "META",
    name: "Meta Platforms Inc.",
    signal: "BUY",
    confidence: 85,
    rsi: 25.8,
    macd: 3.45,
    currentPrice: 352.40,
    predictedChange: 4.1,
    updatedAt: "2026-01-04 09:00:00",
  },
  {
    symbol: "TSLA",
    name: "Tesla Inc.",
    signal: "SELL",
    confidence: 69,
    rsi: 71.5,
    macd: -1.23,
    currentPrice: 248.90,
    predictedChange: -2.3,
    updatedAt: "2026-01-04 09:00:00",
  },
  {
    symbol: "AMD",
    name: "AMD Inc.",
    signal: "BUY",
    confidence: 76,
    rsi: 31.2,
    macd: 1.87,
    currentPrice: 145.60,
    predictedChange: 2.9,
    updatedAt: "2026-01-04 09:00:00",
  },
]

// 더미 포트폴리오 데이터
export const portfolio: PortfolioItem[] = [
  {
    symbol: "AAPL",
    name: "Apple Inc.",
    quantity: 50,
    avgPrice: 165.20,
    currentPrice: 178.50,
    totalValue: 8925.00,
    profit: 665.00,
    profitPercent: 8.05,
  },
  {
    symbol: "NVDA",
    name: "NVIDIA Corp.",
    quantity: 20,
    avgPrice: 450.00,
    currentPrice: 485.20,
    totalValue: 9704.00,
    profit: 704.00,
    profitPercent: 7.82,
  },
  {
    symbol: "GOOGL",
    name: "Alphabet Inc.",
    quantity: 100,
    avgPrice: 135.80,
    currentPrice: 141.20,
    totalValue: 14120.00,
    profit: 540.00,
    profitPercent: 3.98,
  },
  {
    symbol: "MSFT",
    name: "Microsoft Corp.",
    quantity: 30,
    avgPrice: 390.50,
    currentPrice: 378.90,
    totalValue: 11367.00,
    profit: -348.00,
    profitPercent: -2.97,
  },
  {
    symbol: "META",
    name: "Meta Platforms Inc.",
    quantity: 25,
    avgPrice: 320.00,
    currentPrice: 352.40,
    totalValue: 8810.00,
    profit: 810.00,
    profitPercent: 10.13,
  },
]

// 더미 수익률 데이터 (최근 30일)
// 시드 기반 랜덤 생성기
const seededRandom = (seed: number) => {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
};

// 더미 수익률 데이터 (최근 30일)
export const returns: ReturnData[] = Array.from({ length: 30 }, (_, i) => {
  // 기준일을 고정하여 서버/클라이언트 간 일관성 유지 (2024-01-01 기준)
  const date = new Date("2024-01-01");
  date.setDate(date.getDate() + i);

  const randomValue = seededRandom(i); // 시드를 인덱스로 사용
  const dailyReturn = (randomValue - 0.48) * 4; // -1.92% ~ 2.08%

  const benchmarkRandom = seededRandom(i + 200);
  const benchmarkReturn = (benchmarkRandom - 0.45) * 3; // -1.35% ~ 1.65% (약간 더 안정적)

  return {
    date: date.toISOString().split("T")[0],
    portfolioValue: 50000 + seededRandom(i + 100) * 5000 + i * 100,
    dailyReturn: parseFloat(dailyReturn.toFixed(2)),
    cumulativeReturn: parseFloat(((i / 30) * 8 + dailyReturn).toFixed(2)),
    benchmarkReturn: parseFloat(benchmarkReturn.toFixed(2)),
    benchmarkCumulativeReturn: parseFloat(((i / 30) * 5 + benchmarkReturn).toFixed(2)),
  };
});

// 요약 통계
export const summary = {
  totalValue: portfolio.reduce((sum, item) => sum + item.totalValue, 0),
  totalProfit: portfolio.reduce((sum, item) => sum + item.profit, 0),
  profitPercent: 5.23,
  buySignals: predictions.filter((p) => p.signal === "BUY").length,
  sellSignals: predictions.filter((p) => p.signal === "SELL").length,
  holdSignals: predictions.filter((p) => p.signal === "HOLD").length,
}
