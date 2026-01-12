/**
 * Snowballing AI ETF 정보
 *
 * TabPFN AI 모델 기반 3개월 수익률 예측 Top-10 종목으로 구성된 동일가중 포트폴리오
 */

export interface SnowballingETFInfo {
  name: string
  symbol: string
  description: string
  strategy: string
  benchmark: string
  inceptionDate: string
  rebalanceFrequency: string
  holdingsCount: number
  expenseRatio: string
}

export const SNOWBALLING_ETF: SnowballingETFInfo = {
  name: "Snowballing AI ETF",
  symbol: "SNOW-AI",
  description:
    "TabPFN AI 모델 기반 3개월 수익률 예측 Top-10 종목으로 구성된 동일가중 포트폴리오",
  strategy:
    "매월 말 AI 모델이 예측한 향후 3개월 수익률 상위 10개 종목을 동일 비중(10%)으로 편입",
  benchmark: "S&P 500 Index",
  inceptionDate: "2020-01-01",
  rebalanceFrequency: "Monthly",
  holdingsCount: 10,
  expenseRatio: "N/A",
}

// 월 이름 (한국어)
export const MONTH_NAMES_KO = [
  "1월",
  "2월",
  "3월",
  "4월",
  "5월",
  "6월",
  "7월",
  "8월",
  "9월",
  "10월",
  "11월",
  "12월",
]

// 연도-월 포맷 헬퍼
export function formatYearMonth(year: number, month: number): string {
  return `${year}년 ${month}월`
}

// 월 이름 가져오기
export function getMonthName(month: number): string {
  return MONTH_NAMES_KO[month - 1] || ""
}
