// API URL 설정
// - 포트 3000 (npm run dev): localhost:8000 직접 호출
// - 포트 80 (Docker/Nginx): 상대 경로 사용 (Nginx 프록시)
// - 프로덕션: 상대 경로 사용
function getApiBaseUrl(): string {
  if (typeof window === 'undefined') {
    // 서버 사이드 렌더링
    return process.env.NEXT_PUBLIC_API_URL || 'http://ml-service:8000'
  }

  // 브라우저 환경
  const port = window.location.port
  const hostname = window.location.hostname

  // 포트 3000 (npm run dev) -> localhost:8000 직접 호출
  if (port === '3000' && (hostname === 'localhost' || hostname === '127.0.0.1')) {
    return 'http://localhost:8000'
  }

  // 그 외 (Docker/Nginx 포트 80, 프로덕션) -> 상대 경로
  return ''
}

const API_BASE_URL = getApiBaseUrl()

// API 예측 결과 타입 (FastAPI 응답 형식)
export interface APIPrediction {
  id: number
  symbol: string
  prediction_date: string
  target_date: string
  current_close: number
  predicted_close: number
  predicted_direction: "UP" | "DOWN"
  confidence: number
  rsi_value: number
  macd_value: number
  actual_close: number | null
  is_correct: boolean | null
}

export interface PredictionsResponse {
  count: number
  predictions: APIPrediction[]
}

// 프론트엔드용 변환된 예측 타입
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

// 종목 이름 매핑 (추후 API에서 가져올 수 있음)
const STOCK_NAMES: Record<string, string> = {
  AAPL: "Apple Inc.",
  NVDA: "NVIDIA Corp.",
  MSFT: "Microsoft Corp.",
  GOOGL: "Alphabet Inc.",
  AMZN: "Amazon.com Inc.",
  META: "Meta Platforms Inc.",
  TSLA: "Tesla Inc.",
  AMD: "AMD Inc.",
  XOM: "Exxon Mobil Corp.",
  WMT: "Walmart Inc.",
  WFC: "Wells Fargo & Co.",
  JPM: "JPMorgan Chase & Co.",
  V: "Visa Inc.",
  JNJ: "Johnson & Johnson",
  PG: "Procter & Gamble Co.",
  UNH: "UnitedHealth Group Inc.",
  HD: "Home Depot Inc.",
  MA: "Mastercard Inc.",
  DIS: "Walt Disney Co.",
  PYPL: "PayPal Holdings Inc.",
}

// RSI 기반 신호 결정
function getSignalFromRSI(rsi: number): "BUY" | "SELL" | "HOLD" {
  if (rsi < 30) return "BUY"
  if (rsi > 70) return "SELL"
  return "HOLD"
}

// API 예측 데이터를 프론트엔드 형식으로 변환
export function transformPrediction(apiPred: APIPrediction): Prediction {
  const predictedChange = ((apiPred.predicted_close - apiPred.current_close) / apiPred.current_close) * 100

  return {
    symbol: apiPred.symbol,
    name: STOCK_NAMES[apiPred.symbol] || apiPred.symbol,
    signal: getSignalFromRSI(apiPred.rsi_value),
    confidence: Math.round(apiPred.confidence * 100),
    rsi: apiPred.rsi_value,
    macd: apiPred.macd_value,
    currentPrice: apiPred.current_close,
    predictedChange: parseFloat(predictedChange.toFixed(2)),
    updatedAt: new Date(apiPred.prediction_date).toLocaleString("ko-KR"),
  }
}

// 예측 데이터 가져오기
export async function fetchPredictions(): Promise<Prediction[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predictions`, {
      cache: "no-store",
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    const data: PredictionsResponse = await response.json()
    return data.predictions.map(transformPrediction)
  } catch (error) {
    console.error("Failed to fetch predictions:", error)
    throw error
  }
}

// 헬스 체크
export async function checkHealth(): Promise<{
  status: string
  remote_db: string
  local_db: string
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      cache: "no-store",
    })
    return response.json()
  } catch {
    return {
      status: "error",
      remote_db: "unknown",
      local_db: "unknown",
    }
  }
}


// =====================================================
// Factsheet API - Snowballing AI ETF
// =====================================================

// 팩트시트 구성 종목 타입
export interface ETFComposition {
  rank: number
  ticker: string
  weight: number
  stock_name?: string
  sector?: string
}

// 월별 팩트시트 타입
export interface MonthlyFactsheet {
  id: number
  year: number
  month: number
  snapshot_date: string
  nav?: number
  monthly_return?: number
  ytd_return?: number
  volatility?: number
  sharpe_ratio?: number
  max_drawdown?: number
  compositions: ETFComposition[]
}

// 팩트시트 목록 아이템 타입
export interface FactsheetListItem {
  id: number
  year: number
  month: number
  snapshot_date: string
}

// 팩트시트 목록 응답 타입
export interface FactsheetListResponse {
  count: number
  factsheets: FactsheetListItem[]
}

// 팩트시트 목록 조회
export async function fetchFactsheetList(): Promise<FactsheetListItem[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/factsheet`, {
      cache: "no-store",
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    const data: FactsheetListResponse = await response.json()
    return data.factsheets
  } catch (error) {
    console.error("Failed to fetch factsheet list:", error)
    throw error
  }
}

// 특정 월 팩트시트 조회
export async function fetchFactsheet(year: number, month: number): Promise<MonthlyFactsheet> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/factsheet/${year}/${month}`, {
      cache: "no-store",
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    return response.json()
  } catch (error) {
    console.error(`Failed to fetch factsheet for ${year}-${month}:`, error)
    throw error
  }
}

// 최신 팩트시트 조회
export async function fetchLatestFactsheet(): Promise<MonthlyFactsheet> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/factsheet/latest`, {
      cache: "no-store",
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    return response.json()
  } catch (error) {
    console.error("Failed to fetch latest factsheet:", error)
    throw error
  }
}

// 팩트시트 생성 요청
export async function generateFactsheet(year: number, month: number): Promise<MonthlyFactsheet> {
  const response = await fetch(`${API_BASE_URL}/api/factsheet/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ year, month }),
  })

  if (!response.ok) {
    throw new Error(`Failed to generate factsheet: ${response.status}`)
  }

  const data = await response.json()
  return data.snapshot
}

// 모든 과거 팩트시트 일괄 생성
export async function generateAllFactsheets(years?: number[]): Promise<{
  total: number
  success: number
  failed: number
}> {
  const params = years ? `?years=${years.join("&years=")}` : ""
  const response = await fetch(`${API_BASE_URL}/api/factsheet/generate-all${params}`, {
    method: "POST",
  })

  if (!response.ok) {
    throw new Error(`Failed to generate factsheets: ${response.status}`)
  }

  return response.json()
}
