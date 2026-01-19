"use client"

import { ArrowDown, ArrowUp, Plus, TrendingDown, TrendingUp } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { portfolio, predictions } from "@/lib/data"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Pie, PieChart as RechartsPie, Cell } from "recharts"

const COLORS = [
  "var(--chart-pie-1)",
  "var(--chart-pie-2)",
  "var(--chart-pie-3)",
  "var(--chart-pie-4)",
  "var(--chart-pie-5)",
]

const pieData = portfolio.map((item, index) => ({
  name: item.symbol,
  value: item.totalValue,
  fill: COLORS[index % COLORS.length],
}))

const chartConfig = {
  value: {
    label: "가치",
  },
  ...Object.fromEntries(
    portfolio.map((item, index) => [
      item.symbol,
      { label: item.symbol, color: COLORS[index % COLORS.length] },
    ])
  ),
} satisfies ChartConfig

export default function PortfolioPage() {
  const totalValue = portfolio.reduce((sum, item) => sum + item.totalValue, 0)
  const totalProfit = portfolio.reduce((sum, item) => sum + item.profit, 0)
  const totalCost = portfolio.reduce((sum, item) => sum + item.avgPrice * item.quantity, 0)
  const totalReturn = ((totalValue - totalCost) / totalCost) * 100

  const profitableCount = portfolio.filter(p => p.profit > 0).length
  const lossCount = portfolio.filter(p => p.profit <= 0).length

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">포트폴리오</h2>
          <p className="text-muted-foreground">
            보유 종목 현황 및 자산 배분
          </p>
        </div>
        <Button size="sm">
          <Plus className="h-4 w-4 mr-2" />
          종목 추가
        </Button>
      </div>

      {/* 요약 카드 */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">총 자산</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${totalValue.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              {portfolio.length}개 종목 보유
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">총 수익</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${totalProfit >= 0 ? "text-profit-positive" : "text-profit-negative"}`}>
              {totalProfit >= 0 ? "+" : ""}${totalProfit.toLocaleString()}
            </div>
            <p className={`text-xs ${totalReturn >= 0 ? "text-profit-positive" : "text-profit-negative"}`}>
              {totalReturn >= 0 ? "+" : ""}{totalReturn.toFixed(2)}% 수익률
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-profit-positive" />
              수익 종목
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-profit-positive">{profitableCount}</div>
            <p className="text-xs text-muted-foreground">
              전체의 {((profitableCount / portfolio.length) * 100).toFixed(0)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-profit-negative" />
              손실 종목
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-profit-negative">{lossCount}</div>
            <p className="text-xs text-muted-foreground">
              전체의 {((lossCount / portfolio.length) * 100).toFixed(0)}%
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {/* 자산 배분 차트 */}
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>자산 배분</CardTitle>
            <CardDescription>종목별 비중</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-[250px]">
              <RechartsPie>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                  nameKey="name"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <ChartTooltip
                  content={<ChartTooltipContent />}
                  formatter={(value: number) => `$${value.toLocaleString()}`}
                />
              </RechartsPie>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* 보유 종목 테이블 */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>보유 종목 상세</CardTitle>
            <CardDescription>종목별 수익 현황</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>종목</TableHead>
                  <TableHead className="text-right">수량</TableHead>
                  <TableHead className="text-right">평균단가</TableHead>
                  <TableHead className="text-right">현재가</TableHead>
                  <TableHead className="text-right">평가금액</TableHead>
                  <TableHead className="text-right">수익</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {portfolio.map((item) => {
                  const prediction = predictions.find(p => p.symbol === item.symbol)
                  return (
                    <TableRow key={item.symbol}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div>
                            <div className="font-medium">{item.symbol}</div>
                            <div className="text-xs text-muted-foreground">{item.name}</div>
                          </div>
                          {prediction && (
                            <Badge
                              variant={
                                prediction.signal === "BUY" ? "default" :
                                  prediction.signal === "SELL" ? "destructive" : "secondary"
                              }
                              className={`text-xs ${prediction.signal === "BUY" ? "bg-signal-buy" : ""}`}
                            >
                              {prediction.signal === "BUY" ? "매수" :
                                prediction.signal === "SELL" ? "매도" : "관망"}
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">{item.quantity}</TableCell>
                      <TableCell className="text-right">${item.avgPrice.toFixed(2)}</TableCell>
                      <TableCell className="text-right">${item.currentPrice.toFixed(2)}</TableCell>
                      <TableCell className="text-right font-medium">
                        ${item.totalValue.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className={`flex items-center justify-end gap-1 ${item.profit >= 0 ? "text-profit-positive" : "text-profit-negative"
                          }`}>
                          {item.profit >= 0 ? (
                            <ArrowUp className="h-3 w-3" />
                          ) : (
                            <ArrowDown className="h-3 w-3" />
                          )}
                          <span className="font-medium">
                            {item.profit >= 0 ? "+" : ""}{item.profitPercent.toFixed(2)}%
                          </span>
                        </div>
                        <div className={`text-xs ${item.profit >= 0 ? "text-profit-positive" : "text-profit-negative"}`}>
                          {item.profit >= 0 ? "+" : ""}${item.profit.toLocaleString()}
                        </div>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
