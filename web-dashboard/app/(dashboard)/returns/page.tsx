"use client"

import { ArrowDown, ArrowUp, TrendingUp, TrendingDown } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { returns, portfolio, summary } from "@/lib/data"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Area, AreaChart, Bar, BarChart, Line, LineChart, XAxis, YAxis, CartesianGrid, ReferenceLine } from "recharts"

const chartConfig = {
  portfolioValue: {
    label: "포트폴리오 가치",
    color: "var(--chart-1)",
  },
  dailyReturn: {
    label: "일일 수익률",
    color: "var(--chart-2)",
  },
  cumulativeReturn: {
    label: "누적 수익률",
    color: "var(--chart-3)",
  },
  benchmark: {
    label: "KOSPI 200",
    color: "var(--muted-foreground)",
  },
} satisfies ChartConfig

export default function ReturnsPage() {
  const latestReturn = returns[returns.length - 1]
  const previousReturn = returns[returns.length - 2]

  const monthlyReturns = returns

  const avgDailyReturn = returns.reduce((sum, r) => sum + r.dailyReturn, 0) / returns.length
  const maxDailyReturn = Math.max(...returns.map(r => r.dailyReturn))
  const minDailyReturn = Math.min(...returns.map(r => r.dailyReturn))

  const positiveDays = returns.filter(r => r.dailyReturn > 0).length
  const negativeDays = returns.filter(r => r.dailyReturn <= 0).length
  const winRate = (positiveDays / returns.length) * 100

  // 종목별 수익률 계산
  const totalAbsProfit = portfolio.reduce((sum, item) => sum + Math.abs(item.profit), 0)
  const stockReturns = portfolio.map(item => ({
    symbol: item.symbol,
    name: item.name,
    profit: item.profit,
    profitPercent: item.profitPercent,
    contribution: (item.profit / summary.totalProfit) * 100,
    // 바 표시용 정규화된 값 (0-100 범위)
    barWidth: (Math.abs(item.profit) / totalAbsProfit) * 100,
  })).sort((a, b) => b.profitPercent - a.profitPercent)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">수익률 분석</h2>
        <p className="text-muted-foreground">
          포트폴리오 성과 및 수익률 추이
        </p>
      </div>

      {/* 요약 카드 */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">누적 수익률</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${latestReturn.cumulativeReturn >= 0 ? "text-profit-positive" : "text-profit-negative"}`}>
              {latestReturn.cumulativeReturn >= 0 ? "+" : ""}{latestReturn.cumulativeReturn.toFixed(2)}%
            </div>
            <p className="text-xs text-muted-foreground">
              30일 기준
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">일평균 수익률</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${avgDailyReturn >= 0 ? "text-profit-positive" : "text-profit-negative"}`}>
              {avgDailyReturn >= 0 ? "+" : ""}{avgDailyReturn.toFixed(2)}%
            </div>
            <p className="text-xs text-muted-foreground">
              최대 +{maxDailyReturn.toFixed(2)}% / 최소 {minDailyReturn.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">승률</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{winRate.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              {positiveDays}일 수익 / {negativeDays}일 손실
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">오늘의 수익률</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold flex items-center gap-1 ${latestReturn.dailyReturn >= 0 ? "text-profit-positive" : "text-profit-negative"}`}>
              {latestReturn.dailyReturn >= 0 ? (
                <TrendingUp className="h-5 w-5" />
              ) : (
                <TrendingDown className="h-5 w-5" />
              )}
              {latestReturn.dailyReturn >= 0 ? "+" : ""}{latestReturn.dailyReturn.toFixed(2)}%
            </div>
            <p className="text-xs text-muted-foreground">
              전일 {previousReturn.dailyReturn >= 0 ? "+" : ""}{previousReturn.dailyReturn.toFixed(2)}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* 차트 영역 */}
      <Tabs defaultValue="portfolio" className="space-y-4">
        <TabsList>
          <TabsTrigger value="portfolio">포트폴리오 가치</TabsTrigger>
          <TabsTrigger value="daily">일일 수익률</TabsTrigger>
          <TabsTrigger value="cumulative">누적 수익률</TabsTrigger>
        </TabsList>

        <TabsContent value="portfolio">
          <Card>
            <CardHeader>
              <CardTitle>포트폴리오 가치 추이</CardTitle>
              <CardDescription>최근 30일 포트폴리오 가치 변동</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[350px] w-full">
                <AreaChart data={monthlyReturns}>
                  <defs>
                    <linearGradient id="fillPortfolio" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--color-portfolioValue)" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="var(--color-portfolioValue)" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => value.slice(5)}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                    tickLine={false}
                    axisLine={false}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Area
                    type="monotone"
                    dataKey="portfolioValue"
                    stroke="var(--color-portfolioValue)"
                    fill="url(#fillPortfolio)"
                  />
                </AreaChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="daily">
          <Card>
            <CardHeader>
              <CardTitle>일일 수익률</CardTitle>
              <CardDescription>최근 30일 일일 수익률 분포</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[350px] w-full">
                <BarChart data={monthlyReturns}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => value.slice(8)}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    tickFormatter={(value) => `${value}%`}
                    tickLine={false}
                    axisLine={false}
                  />
                  <ReferenceLine y={0} stroke="#666" />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar
                    dataKey="dailyReturn"
                    fill="var(--color-dailyReturn)"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="cumulative">
          <Card>
            <CardHeader>
              <CardTitle>누적 수익률 비교</CardTitle>
              <CardDescription>포트폴리오 vs 시장 벤치마크 (KOSPI 200)</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[350px] w-full">
                <LineChart data={monthlyReturns}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => value.slice(5)}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    tickFormatter={(value) => `${value}%`}
                    tickLine={false}
                    axisLine={false}
                  />
                  <ReferenceLine y={0} stroke="#666" />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Line
                    type="monotone"
                    dataKey="cumulativeReturn"
                    stroke="var(--color-cumulativeReturn)"
                    name="포트폴리오"
                    strokeWidth={3}
                    dot={{ fill: "var(--color-cumulativeReturn)", r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="benchmarkCumulativeReturn"
                    stroke="var(--color-benchmark)"
                    name="KOSPI 200"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 종목별 수익 기여도 */}
      <Card>
        <CardHeader>
          <CardTitle>종목별 수익 기여도</CardTitle>
          <CardDescription>포트폴리오 수익에 대한 각 종목의 기여도</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>종목</TableHead>
                <TableHead className="text-right">수익률</TableHead>
                <TableHead className="text-right">수익금</TableHead>
                <TableHead className="text-right">기여도</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {stockReturns.map((item) => (
                <TableRow key={item.symbol}>
                  <TableCell>
                    <div>
                      <div className="font-medium">{item.symbol}</div>
                      <div className="text-xs text-muted-foreground">{item.name}</div>
                    </div>
                  </TableCell>
                  <TableCell className="text-right">
                    <div className={`flex items-center justify-end gap-1 ${item.profitPercent >= 0 ? "text-profit-positive" : "text-profit-negative"
                      }`}>
                      {item.profitPercent >= 0 ? (
                        <ArrowUp className="h-3 w-3" />
                      ) : (
                        <ArrowDown className="h-3 w-3" />
                      )}
                      {item.profitPercent >= 0 ? "+" : ""}{item.profitPercent.toFixed(2)}%
                    </div>
                  </TableCell>
                  <TableCell className={`text-right ${item.profit >= 0 ? "text-profit-positive" : "text-profit-negative"}`}>
                    {item.profit >= 0 ? "+" : ""}${item.profit.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${item.profit >= 0 ? "bg-profit-positive" : "bg-profit-negative"}`}
                          style={{ width: `${Math.max(item.barWidth, 5)}%` }}
                        />
                      </div>
                      <span className="text-sm w-14 text-right">
                        {item.contribution >= 0 ? "+" : ""}{item.contribution.toFixed(1)}%
                      </span>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
