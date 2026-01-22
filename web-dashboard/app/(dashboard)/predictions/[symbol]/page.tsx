"use client"

import { useEffect, useState, use } from "react"
import Link from "next/link"
import { ArrowLeft, ArrowUp, ArrowDown, TrendingUp, Info } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CandlestickChart } from "@/components/charts/candlestick-chart"
import {
  fetchCandlestickForecast,
  fetchPredictionHistory,
  getStockInfo,
  type CandlestickForecastResponse,
  type PredictionHistoryItem,
} from "@/lib/api"
import { CandlestickData, Time } from "lightweight-charts"

interface PageProps {
  params: Promise<{ symbol: string }>
}

export default function PredictionDetailPage({ params }: PageProps) {
  const resolvedParams = use(params)
  const symbol = resolvedParams.symbol.toUpperCase()
  const stockInfo = getStockInfo(symbol)

  const [forecastData, setForecastData] = useState<CandlestickForecastResponse | null>(null)
  const [historyData, setHistoryData] = useState<PredictionHistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadData() {
      setLoading(true)
      setError(null)

      try {
        // 210일 데이터 요청 (MA120 계산을 위해 최소 120일 + 90일 예측 기간 필요)
        const [forecast, history] = await Promise.all([
          fetchCandlestickForecast(symbol, 210),
          fetchPredictionHistory(symbol, 180),
        ])
        setForecastData(forecast)
        setHistoryData(history)
      } catch (err) {
        console.error("Failed to load data:", err)
        setError("데이터를 불러오는데 실패했습니다.")
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [symbol])

  // Transform API data to lightweight-charts format
  const chartData: CandlestickData<Time>[] = forecastData
    ? forecastData.data.map((d) => ({
      time: d.time as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }))
    : []

  // Get latest prediction for this symbol
  const latestPrediction = historyData[0]

  // Calculate stats from history
  const completedPredictions = historyData.filter((p) => p.has_performance)
  const correctPredictions = completedPredictions.filter((p) => p.is_correct)
  const accuracy = completedPredictions.length > 0
    ? ((correctPredictions.length / completedPredictions.length) * 100).toFixed(1)
    : "N/A"

  // Create reference lines for the chart
  const referenceLines = []

  if (forecastData) {
    referenceLines.push({
      price: forecastData.current_price,
      color: "#3b82f6", // blue-500
      label: "현재가",
    })
  }

  if (latestPrediction) {
    referenceLines.push({
      price: latestPrediction.predicted_close,
      color: latestPrediction.predicted_direction === "UP" ? "#22c55e" : "#ef4444", // green-500 or red-500
      label: "예측가",
    })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link href="/predictions">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <div>
          <h2 className="text-2xl font-bold tracking-tight">{symbol}</h2>
          <p className="text-muted-foreground">{stockInfo.name}</p>
        </div>
        <Badge variant="outline" className="ml-auto">
          {stockInfo.sector}
        </Badge>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-4 pb-4">
            <p className="text-red-600">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">현재가</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <div className="text-2xl font-bold">
                ${forecastData?.current_price.toFixed(2) || "N/A"}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">최신 예측</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-8 w-24" />
            ) : latestPrediction ? (
              <div className="flex items-center gap-2">
                <Badge
                  className={
                    latestPrediction.predicted_direction === "UP"
                      ? "bg-green-600"
                      : "bg-red-600"
                  }
                >
                  {latestPrediction.predicted_direction === "UP" ? "상승" : "하락"}
                </Badge>
                <span className="text-muted-foreground">
                  {(latestPrediction.confidence * 100).toFixed(0)}% 신뢰도
                </span>
              </div>
            ) : (
              <span className="text-muted-foreground">데이터 없음</span>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">RSI</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-8 w-16" />
            ) : latestPrediction ? (
              <div
                className={`text-2xl font-bold ${latestPrediction.rsi_value < 30
                  ? "text-green-600"
                  : latestPrediction.rsi_value > 70
                    ? "text-red-600"
                    : ""
                  }`}
              >
                {latestPrediction.rsi_value.toFixed(1)}
              </div>
            ) : (
              <span className="text-muted-foreground">N/A</span>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">예측 정확도</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-muted-foreground" />
                <span className="text-2xl font-bold">{accuracy}%</span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Tabs for Chart and History */}
      <Tabs defaultValue="forecast" className="space-y-4">
        <TabsList>
          <TabsTrigger value="forecast">향후 3개월 예측</TabsTrigger>
          <TabsTrigger value="history">예측 히스토리</TabsTrigger>
        </TabsList>

        <TabsContent value="forecast">
          <Card>
            <CardHeader>
              <CardTitle>캔들스틱 차트 (210일)</CardTitle>
              <CardDescription className="space-y-2">
                <div className="flex items-center gap-2">
                  <Info className="h-4 w-4" />
                  현재는 더미 데이터입니다. 추후 ML 모델로 실제 예측 데이터로 교체됩니다.
                </div>
                <div className="flex flex-wrap gap-4 text-xs">
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-0.5 bg-blue-500 inline-block"></span>
                    MA5
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-0.5 bg-amber-500 inline-block"></span>
                    MA20
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-0.5 bg-purple-500 inline-block"></span>
                    MA60
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-0.5 bg-pink-500 inline-block"></span>
                    MA120
                  </span>
                </div>
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <Skeleton className="h-[400px] w-full" />
              ) : chartData.length > 0 ? (
                <CandlestickChart
                  data={chartData}
                  height={400}
                  referenceLines={referenceLines}
                />
              ) : (
                <div className="h-[400px] flex items-center justify-center text-muted-foreground">
                  차트 데이터가 없습니다
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>예측 히스토리</CardTitle>
              <CardDescription>
                최근 180일간의 예측 기록. 3개월 경과 예측은 실제 수익률이 표시됩니다.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : historyData.length > 0 ? (
                <div className="space-y-4">
                  {historyData.slice(0, 20).map((prediction) => (
                    <div
                      key={prediction.id}
                      className="flex items-center justify-between border-b pb-4 last:border-0"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">
                            {new Date(prediction.prediction_date).toLocaleDateString("ko-KR")}
                          </span>
                          <Badge
                            variant="outline"
                            className={
                              prediction.predicted_direction === "UP"
                                ? "border-green-500 text-green-500"
                                : "border-red-500 text-red-500"
                            }
                          >
                            {prediction.predicted_direction === "UP" ? (
                              <ArrowUp className="h-3 w-3 mr-1" />
                            ) : (
                              <ArrowDown className="h-3 w-3 mr-1" />
                            )}
                            {prediction.predicted_direction}
                          </Badge>
                          <span className="text-sm text-muted-foreground">
                            신뢰도 {(prediction.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          예측가: ${prediction.predicted_close.toFixed(2)} (현재가:
                          ${prediction.current_close.toFixed(2)})
                        </p>
                      </div>

                      <div className="text-right">
                        {prediction.has_performance ? (
                          <div className="space-y-1">
                            <div
                              className={`font-semibold ${prediction.actual_return && prediction.actual_return >= 0
                                ? "text-green-600"
                                : "text-red-600"
                                }`}
                            >
                              {prediction.actual_return
                                ? `${prediction.actual_return >= 0 ? "+" : ""}${prediction.actual_return.toFixed(2)}%`
                                : "N/A"}
                            </div>
                            <Badge
                              variant={prediction.is_correct ? "default" : "destructive"}
                              className={prediction.is_correct ? "bg-green-600" : ""}
                            >
                              {prediction.is_correct ? "적중" : "빗나감"}
                            </Badge>
                          </div>
                        ) : (
                          <span className="text-sm text-muted-foreground">
                            {prediction.days_elapsed}일 경과
                            <br />
                            (3개월 후 확인)
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  예측 히스토리가 없습니다
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
