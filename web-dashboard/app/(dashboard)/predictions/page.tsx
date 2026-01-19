"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { ArrowDown, ArrowUp, RefreshCw, AlertCircle, ChevronRight, Search } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { fetchPredictions, type Prediction } from "@/lib/api"

export default function PredictionsPage() {
  const router = useRouter()
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<"ALL" | "BUY" | "SELL" | "HOLD">("ALL")
  const [searchTerm, setSearchTerm] = useState("")

  const loadPredictions = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchPredictions()
      setPredictions(data)
    } catch (err) {
      setError("예측 데이터를 불러오는데 실패했습니다.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPredictions()
  }, [])

  const filteredPredictions = predictions.filter(p => {
    const matchesFilter = filter === "ALL" ? true : p.signal === filter
    const matchesSearch = p.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.name.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesFilter && matchesSearch
  })

  const buyCount = predictions.filter(p => p.signal === "BUY").length
  const sellCount = predictions.filter(p => p.signal === "SELL").length
  const holdCount = predictions.filter(p => p.signal === "HOLD").length

  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">예측 결과</h2>
            <p className="text-muted-foreground">RSI/MACD 기반 매매 신호 분석 결과</p>
          </div>
          <Button variant="outline" size="sm" onClick={loadPredictions}>
            <RefreshCw className="h-4 w-4 mr-2" />
            다시 시도
          </Button>
        </div>
        <Card className="border-signal-sell-border bg-signal-sell-bg">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-signal-sell">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
            <p className="text-sm text-muted-foreground mt-2">
              FastAPI 서비스가 실행 중인지 확인해주세요. (http://localhost:8000)
            </p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">예측 결과</h2>
          <p className="text-muted-foreground">
            RSI/MACD 기반 매매 신호 분석 결과 {!loading && `(${predictions.length}개 종목)`}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={loadPredictions} disabled={loading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          새로고침
        </Button>
      </div>

      {/* 요약 카드 */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border-signal-buy-border bg-signal-buy-bg">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-signal-buy">
              매수 신호
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-9 w-16" />
            ) : (
              <>
                <div className="text-3xl font-bold text-signal-buy">
                  {buyCount}
                </div>
                <p className="text-xs text-signal-buy">
                  RSI &lt; 30 과매도 구간
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="border-signal-sell-border bg-signal-sell-bg">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-signal-sell">
              매도 신호
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-9 w-16" />
            ) : (
              <>
                <div className="text-3xl font-bold text-signal-sell">
                  {sellCount}
                </div>
                <p className="text-xs text-signal-sell">
                  RSI &gt; 70 과매수 구간
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="border-signal-hold-border bg-signal-hold-bg">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-signal-hold">
              관망 신호
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-9 w-16" />
            ) : (
              <>
                <div className="text-3xl font-bold text-signal-hold">
                  {holdCount}
                </div>
                <p className="text-xs text-signal-hold">
                  30 &lt; RSI &lt; 70 중립 구간
                </p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 필터 및 검색 */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
        <Tabs defaultValue="ALL" onValueChange={(v) => setFilter(v as typeof filter)} className="w-full sm:w-auto">
          <TabsList>
            <TabsTrigger value="ALL">전체</TabsTrigger>
            <TabsTrigger value="BUY" className="text-signal-buy">매수</TabsTrigger>
            <TabsTrigger value="SELL" className="text-signal-sell">매도</TabsTrigger>
            <TabsTrigger value="HOLD">관망</TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="relative w-full sm:w-64">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="종목명 또는 심볼 검색..."
            className="pl-8"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>예측 상세</CardTitle>
          <CardDescription>
            마지막 업데이트: {predictions[0]?.updatedAt || "-"} | 종목을 클릭하면 상세 차트를 볼 수 있습니다
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="min-h-[500px]">
            {loading ? (
              <div className="space-y-3">
                {Array.from({ length: 10 }).map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>종목</TableHead>
                    <TableHead>현재가</TableHead>
                    <TableHead>RSI</TableHead>
                    <TableHead>MACD</TableHead>
                    <TableHead>신호</TableHead>
                    <TableHead>신뢰도</TableHead>
                    <TableHead className="text-right">예상 변동</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredPredictions.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center h-24 text-muted-foreground">
                        검색 결과가 없습니다.
                      </TableCell>
                    </TableRow>
                  ) : filteredPredictions.map((prediction) => (
                    <TableRow
                      key={prediction.symbol}
                      className="cursor-pointer hover:bg-muted/50"
                      onClick={() => router.push(`/predictions/${prediction.symbol}`)}
                    >
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div>
                            <div className="font-medium">{prediction.symbol}</div>
                            <div className="text-sm text-muted-foreground">{prediction.name}</div>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>${prediction.currentPrice.toFixed(2)}</TableCell>
                      <TableCell>
                        <span className={
                          prediction.rsi < 30 ? "text-signal-buy font-medium" :
                            prediction.rsi > 70 ? "text-signal-sell font-medium" : ""
                        }>
                          {prediction.rsi.toFixed(1)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className={prediction.macd >= 0 ? "text-signal-buy" : "text-signal-sell"}>
                          {prediction.macd >= 0 ? "+" : ""}{prediction.macd.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            prediction.signal === "BUY" ? "default" :
                              prediction.signal === "SELL" ? "destructive" : "secondary"
                          }
                          className={prediction.signal === "BUY" ? "bg-signal-buy" : ""}
                        >
                          {prediction.signal === "BUY" ? "매수" :
                            prediction.signal === "SELL" ? "매도" : "관망"}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className="h-full"
                              style={{
                                width: `${prediction.confidence}%`,
                                background: `linear-gradient(90deg, ${prediction.confidence >= 75 ? "#10B981" :
                                  prediction.confidence >= 50 ? "#F59E0B" : "#EF4444"
                                  } 0%, ${prediction.confidence >= 75 ? "#059669" :
                                    prediction.confidence >= 50 ? "#D97706" : "#DC2626"
                                  } 100%)`
                              }}
                            />
                          </div>
                          <span className="text-sm">{prediction.confidence}%</span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className={`flex items-center gap-1 ${prediction.predictedChange >= 0 ? "text-signal-buy" : "text-signal-sell"
                            }`}>
                            {prediction.predictedChange >= 0 ? (
                              <ArrowUp className="h-4 w-4" />
                            ) : (
                              <ArrowDown className="h-4 w-4" />
                            )}
                            {prediction.predictedChange >= 0 ? "+" : ""}{prediction.predictedChange.toFixed(2)}%
                          </div>
                          <ChevronRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 설명 */}
      <Card>
        <CardHeader>
          <CardTitle>예측 모델 설명</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            <strong>RSI (Relative Strength Index):</strong> 0-100 범위의 모멘텀 지표.
            30 이하는 과매도, 70 이상은 과매수 상태를 나타냅니다.
          </p>
          <p>
            <strong>MACD (Moving Average Convergence Divergence):</strong>
            추세 추종 모멘텀 지표. 양수는 상승 추세, 음수는 하락 추세를 나타냅니다.
          </p>
          <p>
            <strong>신뢰도:</strong> 모델의 예측 확신도. 높을수록 신호가 강합니다.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
