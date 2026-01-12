"use client"

import { useState, useEffect } from "react"
import {
  Info,
  Target,
  TrendingUp,
  PieChart as PieChartIcon,
  Download,
  RefreshCw,
  AlertCircle,
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Skeleton } from "@/components/ui/skeleton"
import {
  fetchFactsheetList,
  fetchFactsheet,
  generateAllFactsheets,
  type FactsheetListItem,
  type MonthlyFactsheet,
} from "@/lib/api"
import {
  SNOWBALLING_ETF,
  formatYearMonth,
  getMonthName,
} from "@/lib/types/snowballing-etf"

export default function FactSheetPage() {
  const [factsheetList, setFactsheetList] = useState<FactsheetListItem[]>([])
  const [selectedYear, setSelectedYear] = useState<number | null>(null)
  const [selectedMonth, setSelectedMonth] = useState<number | null>(null)
  const [factsheet, setFactsheet] = useState<MonthlyFactsheet | null>(null)
  const [loading, setLoading] = useState(true)
  const [listLoading, setListLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)

  // 사용 가능한 연도 (목록에서 추출)
  const availableYears = [...new Set(factsheetList.map((f) => f.year))].sort(
    (a, b) => b - a
  )

  // 선택된 연도의 사용 가능한 월
  const availableMonths = factsheetList
    .filter((f) => f.year === selectedYear)
    .map((f) => f.month)
    .sort((a, b) => b - a)

  // 팩트시트 목록 로드
  useEffect(() => {
    async function loadList() {
      setListLoading(true)
      try {
        const list = await fetchFactsheetList()
        setFactsheetList(list)

        if (list.length > 0) {
          // 가장 최신 팩트시트 선택
          setSelectedYear(list[0].year)
          setSelectedMonth(list[0].month)
        }
        setError(null)
      } catch (err) {
        console.error("Failed to load factsheet list:", err)
        setError("팩트시트 목록을 불러올 수 없습니다. 서버 연결을 확인해주세요.")
      } finally {
        setListLoading(false)
      }
    }
    loadList()
  }, [])

  // 팩트시트 로드 (연도/월 변경 시)
  useEffect(() => {
    async function loadFactsheet() {
      if (!selectedYear || !selectedMonth) return

      setLoading(true)
      setError(null)

      try {
        const data = await fetchFactsheet(selectedYear, selectedMonth)
        setFactsheet(data)
      } catch (err) {
        console.error(`Failed to load factsheet for ${selectedYear}-${selectedMonth}:`, err)
        setError(`${selectedYear}년 ${selectedMonth}월 팩트시트를 불러올 수 없습니다.`)
        setFactsheet(null)
      } finally {
        setLoading(false)
      }
    }
    loadFactsheet()
  }, [selectedYear, selectedMonth])

  // 연도 변경 시 해당 연도의 첫 번째 월 선택
  const handleYearChange = (year: string) => {
    const yearNum = Number(year)
    setSelectedYear(yearNum)

    const monthsForYear = factsheetList
      .filter((f) => f.year === yearNum)
      .map((f) => f.month)
      .sort((a, b) => b - a)

    if (monthsForYear.length > 0) {
      setSelectedMonth(monthsForYear[0])
    }
  }

  // 과거 팩트시트 일괄 생성
  const handleGenerateAll = async () => {
    setGenerating(true)
    try {
      const result = await generateAllFactsheets()
      alert(`팩트시트 생성 완료: ${result.success}개 성공, ${result.failed}개 실패`)
      // 목록 새로고침
      const list = await fetchFactsheetList()
      setFactsheetList(list)
      if (list.length > 0 && !selectedYear) {
        setSelectedYear(list[0].year)
        setSelectedMonth(list[0].month)
      }
    } catch (err) {
      console.error("Failed to generate factsheets:", err)
      alert("팩트시트 생성 중 오류가 발생했습니다.")
    } finally {
      setGenerating(false)
    }
  }

  // PDF 다운로드 (간단한 인쇄 기능)
  const handlePDFDownload = () => {
    window.print()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">{SNOWBALLING_ETF.name}</h2>
          <p className="text-muted-foreground">AI 예측 기반 월별 팩트시트</p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          {/* 연도 선택 */}
          <Select
            value={selectedYear?.toString() || ""}
            onValueChange={(v) => v && handleYearChange(v)}
            disabled={listLoading || availableYears.length === 0}
          >
            <SelectTrigger className="w-30">
              <SelectValue>{selectedYear ? `${selectedYear}년` : "연도"}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {availableYears.map((year) => (
                <SelectItem key={year} value={year.toString()}>
                  {year}년
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* 월 선택 */}
          <Select
            value={selectedMonth?.toString() || ""}
            onValueChange={(v) => v && setSelectedMonth(Number(v))}
            disabled={listLoading || availableMonths.length === 0}
          >
            <SelectTrigger className="w-25">
              <SelectValue>{selectedMonth ? getMonthName(selectedMonth) : "월"}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {availableMonths.map((month) => (
                <SelectItem key={month} value={month.toString()}>
                  {getMonthName(month)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* PDF 다운로드 버튼 */}
          <Button variant="outline" onClick={handlePDFDownload} disabled={!factsheet}>
            <Download className="h-4 w-4 mr-2" />
            PDF
          </Button>
        </div>
      </div>

      {/* 데이터 없음 경고 및 생성 버튼 */}
      {!listLoading && factsheetList.length === 0 && (
        <Card className="border-yellow-200 bg-yellow-50 dark:bg-yellow-950 dark:border-yellow-800">
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-yellow-700 dark:text-yellow-300">
                <AlertCircle className="h-4 w-4" />
                <span className="text-sm">팩트시트 데이터가 없습니다. 과거 데이터를 생성해주세요.</span>
              </div>
              <Button
                onClick={handleGenerateAll}
                disabled={generating}
                variant="outline"
                size="sm"
              >
                {generating ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    생성 중...
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    2020-2024 데이터 생성
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 로딩 상태 */}
      {loading && selectedYear && selectedMonth && (
        <div className="space-y-4">
          <Skeleton className="h-[200px] w-full" />
          <Skeleton className="h-[300px] w-full" />
        </div>
      )}

      {/* 에러 상태 */}
      {error && !loading && (
        <Card className="border-red-200 bg-red-50 dark:bg-red-950 dark:border-red-800">
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 팩트시트 내용 */}
      {factsheet && !loading && (
        <div className="space-y-6 print:space-y-4">
          {/* Section 1: ETF 개요 + 투자 전략 */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* ETF 개요 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  ETF 개요
                </CardTitle>
                <CardDescription>
                  {formatYearMonth(factsheet.year, factsheet.month)} 기준
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">상품명</p>
                    <p className="font-medium">{SNOWBALLING_ETF.name}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">심볼</p>
                    <p className="font-medium">{SNOWBALLING_ETF.symbol}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">리밸런싱 주기</p>
                    <p className="font-medium">{SNOWBALLING_ETF.rebalanceFrequency}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">보유 종목 수</p>
                    <p className="font-medium">{SNOWBALLING_ETF.holdingsCount}개</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">벤치마크</p>
                    <p className="font-medium">{SNOWBALLING_ETF.benchmark}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">기준일</p>
                    <p className="font-medium">
                      {new Date(factsheet.snapshot_date).toLocaleDateString("ko-KR")}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 투자 전략 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  투자 전략
                </CardTitle>
                <CardDescription>AI 기반 종목 선정</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">운용 목표</p>
                  <p className="text-sm">{SNOWBALLING_ETF.description}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">운용 전략</p>
                  <p className="text-sm">{SNOWBALLING_ETF.strategy}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-2">핵심 특징</p>
                  <ul className="space-y-1 text-sm">
                    <li className="flex items-start gap-2">
                      <span className="text-primary mt-1">-</span>
                      TabPFN AI 모델 기반 3개월 수익률 예측
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-primary mt-1">-</span>
                      매월 말 Top-10 종목 동일가중 편입
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-primary mt-1">-</span>
                      242개 기술적 지표 기반 종목 선정
                    </li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Section 2: 운용 성과 (데이터가 있는 경우) */}
          {(factsheet.monthly_return !== null ||
            factsheet.ytd_return !== null ||
            factsheet.nav !== null) && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  운용 성과
                </CardTitle>
                <CardDescription>수익률 및 위험 지표</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  {factsheet.nav !== null && (
                    <div className="p-4 border rounded-lg text-center">
                      <p className="text-xs text-muted-foreground mb-1">NAV</p>
                      <p className="text-xl font-bold">
                        ${factsheet.nav?.toFixed(2)}
                      </p>
                    </div>
                  )}
                  {factsheet.monthly_return !== null && (
                    <div className="p-4 border rounded-lg text-center">
                      <p className="text-xs text-muted-foreground mb-1">월간 수익률</p>
                      <p
                        className={`text-xl font-bold ${
                          (factsheet.monthly_return ?? 0) >= 0
                            ? "text-green-600"
                            : "text-red-600"
                        }`}
                      >
                        {(factsheet.monthly_return ?? 0) >= 0 ? "+" : ""}
                        {factsheet.monthly_return?.toFixed(2)}%
                      </p>
                    </div>
                  )}
                  {factsheet.ytd_return !== null && (
                    <div className="p-4 border rounded-lg text-center">
                      <p className="text-xs text-muted-foreground mb-1">연초대비(YTD)</p>
                      <p
                        className={`text-xl font-bold ${
                          (factsheet.ytd_return ?? 0) >= 0
                            ? "text-green-600"
                            : "text-red-600"
                        }`}
                      >
                        {(factsheet.ytd_return ?? 0) >= 0 ? "+" : ""}
                        {factsheet.ytd_return?.toFixed(2)}%
                      </p>
                    </div>
                  )}
                  {factsheet.volatility !== null && (
                    <div className="p-4 border rounded-lg text-center">
                      <p className="text-xs text-muted-foreground mb-1">변동성</p>
                      <p className="text-xl font-bold">
                        {factsheet.volatility?.toFixed(2)}%
                      </p>
                    </div>
                  )}
                  {factsheet.sharpe_ratio !== null && (
                    <div className="p-4 border rounded-lg text-center">
                      <p className="text-xs text-muted-foreground mb-1">샤프지수</p>
                      <p className="text-xl font-bold">
                        {factsheet.sharpe_ratio?.toFixed(2)}
                      </p>
                    </div>
                  )}
                  {factsheet.max_drawdown !== null && (
                    <div className="p-4 border rounded-lg text-center">
                      <p className="text-xs text-muted-foreground mb-1">최대낙폭</p>
                      <p className="text-xl font-bold text-red-600">
                        {factsheet.max_drawdown?.toFixed(2)}%
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Section 3: Top 10 종목 구성 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChartIcon className="h-5 w-5" />
                Top 10 종목 구성
              </CardTitle>
              <CardDescription>동일 가중치 포트폴리오 (각 10%)</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[60px]">순위</TableHead>
                    <TableHead>종목코드</TableHead>
                    <TableHead>종목명</TableHead>
                    <TableHead className="text-right">비중</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {factsheet.compositions.map((comp) => (
                    <TableRow key={comp.rank}>
                      <TableCell className="font-medium">{comp.rank}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{comp.ticker}</Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {comp.stock_name || comp.ticker}
                      </TableCell>
                      <TableCell className="text-right font-medium">
                        {comp.weight.toFixed(1)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Footer */}
          <div className="text-xs text-muted-foreground text-center py-4 border-t print:mt-8">
            기준일:{" "}
            {new Date(factsheet.snapshot_date).toLocaleDateString("ko-KR")} |
            본 자료는 투자 권유가 아니며, 투자 결정 시 본인의 판단과 책임하에
            결정하시기 바랍니다.
          </div>
        </div>
      )}
    </div>
  )
}
