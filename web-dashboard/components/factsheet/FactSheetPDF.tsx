/* eslint-disable jsx-a11y/alt-text */
"use client"

import React from "react"
import {
  Document,
  Page,
  Text,
  View,
  StyleSheet,
  Font,
} from "@react-pdf/renderer"
import { MonthlyFactsheet } from "@/lib/api"
import { SNOWBALLING_ETF } from "@/lib/types/snowballing-etf"

// 한글 폰트 등록 (네이버 나눔바른고딕 - Moonspam Repo)
Font.register({
  family: "NanumBarunGothic",
  src: "https://raw.githubusercontent.com/moonspam/NanumBarunGothic/master/NanumBarunGothicSubset.ttf",
})

Font.register({
  family: "NanumBarunGothicBold",
  src: "https://raw.githubusercontent.com/moonspam/NanumBarunGothic/master/NanumBarunGothicBoldSubset.ttf",
})

const styles = StyleSheet.create({
  page: {
    flexDirection: "column",
    backgroundColor: "#FFFFFF",
    padding: 30,
    fontFamily: "NanumBarunGothic",
  },
  header: {
    marginBottom: 20,
    borderBottomWidth: 2,
    borderBottomColor: "#003264", // Orange accent
    borderStyle: "solid",
    paddingBottom: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-end",
  },
  headerTitle: {
    fontSize: 20,
    fontFamily: "NanumBarunGothicBold",
    color: "#000000",
  },
  headerSubtitle: {
    fontSize: 10,
    color: "#666666",
    marginTop: 4,
  },
  headerLogo: {
    fontSize: 14,
    fontFamily: "NanumBarunGothicBold",
    color: "#003264",
  },
  dateTag: {
    fontSize: 9,
    textAlign: "right",
    color: "#666666",
    marginTop: 5,
    marginBottom: 10,
  },
  section: {
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 12,
    fontFamily: "NanumBarunGothicBold",
    color: "#003264",
    marginBottom: 6,
    borderBottomWidth: 0.5,
    borderBottomColor: "#CCCCCC",
    borderStyle: "solid",
    paddingBottom: 2,
  },
  row: {
    flexDirection: "row",
    gap: 15,
  },
  col: {
    flex: 1,
  },
  // Table Styles
  table: {
    width: "100%",
    marginBottom: 10,
  },
  tableRow: {
    flexDirection: "row",
    borderBottomWidth: 0.5,
    borderBottomColor: "#E0E0E0",
    borderStyle: "solid",
    alignItems: "center",
    minHeight: 18,
  },
  tableHeader: {
    backgroundColor: "#e6ecffff",
  },
  tableCellLabel: {
    flex: 2,
    fontSize: 8,
    color: "#333333",
    padding: 3,
  },
  tableCellValue: {
    flex: 3,
    fontSize: 8,
    padding: 3,
    textAlign: "right",
    fontFamily: "NanumBarunGothicBold",
  },
  // Box Style
  infoBox: {
    backgroundColor: "#e6ecffff",
    padding: 10,
    borderRadius: 4,
    marginBottom: 10,
  },
  infoBoxTitle: {
    fontSize: 10,
    fontFamily: "NanumBarunGothicBold",
    color: "#003264",
    marginBottom: 4,
  },
  infoBoxText: {
    fontSize: 8,
    marginBottom: 2,
    lineHeight: 1.4,
  },
  // Performance Table
  perfTable: {
    marginTop: 5,
    borderTopWidth: 1,
    borderTopColor: "#333333",
    borderStyle: "solid",
  },
  perfHeaderCell: {
    flex: 1,
    fontSize: 8,
    padding: 4,
    textAlign: "center",
    backgroundColor: "#F5F5F5",
    fontFamily: "NanumBarunGothicBold",
  },
  perfCell: {
    flex: 1,
    fontSize: 8,
    padding: 4,
    textAlign: "center",
  },
  textRed: {
    color: "#FF0000",
  },
  textBlue: {
    color: "#0000FF",
  },
  // Top 10 Table
  top10Row: {
    flexDirection: "row",
    borderBottomWidth: 0.5,
    borderBottomColor: "#E0E0E0",
    borderStyle: "solid",
    paddingVertical: 3,
  },
  top10Rank: { width: 30, fontSize: 8, textAlign: "center" },
  top10Name: { flex: 1, fontSize: 8 },
  top10Ticker: { width: 50, fontSize: 8, color: "#666666" },
  top10Weight: { width: 40, fontSize: 8, textAlign: "right" },

  footer: {
    position: "absolute",
    bottom: 20,
    left: 30,
    right: 30,
    fontSize: 8,
    color: "#999999",
    textAlign: "center",
    borderTopWidth: 0.5,
    borderTopColor: "#E0E0E0",
    borderStyle: "solid",
    paddingTop: 10,
  },
})

interface FactSheetPDFProps {
  data: MonthlyFactsheet
}

export const FactSheetPDF = ({ data }: FactSheetPDFProps) => {
  const isPositive = (val: number | null | undefined) => (val ?? 0) >= 0

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.headerTitle}>{SNOWBALLING_ETF.name}</Text>
            <Text style={styles.headerSubtitle}>
              ({SNOWBALLING_ETF.symbol}) | {SNOWBALLING_ETF.benchmark}
            </Text>
          </View>
          <View>
            <Text style={styles.headerLogo}>Snowballing AI ETF</Text>
          </View>
        </View>
        <Text style={styles.dateTag}>
          기준일: {new Date(data.snapshot_date).toLocaleDateString("ko-KR")}
        </Text>

        <View style={{ flexDirection: "row", gap: 20 }}>
          {/* Left Column */}
          <View style={{ flex: 1 }}>
            {/* Basic Info */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>기본정보</Text>
              <View style={styles.table}>
                <View style={[styles.tableRow, styles.tableHeader]}>
                  <Text style={styles.tableCellLabel}>최초설정일</Text>
                  <Text style={styles.tableCellValue}>{SNOWBALLING_ETF.inceptionDate}</Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>기초지수</Text>
                  <Text style={styles.tableCellValue}>{SNOWBALLING_ETF.benchmark}</Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>순자산총액</Text>
                  <Text style={styles.tableCellValue}>
                    ${data.nav ? (data.nav * 1000000).toLocaleString() : "-"} (추정)
                  </Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>1주당순자산(NAV)</Text>
                  <Text style={styles.tableCellValue}>
                    ${data.nav?.toFixed(2) ?? "-"}
                  </Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>총보수</Text>
                  <Text style={styles.tableCellValue}>{SNOWBALLING_ETF.expenseRatio}</Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>리밸런싱 주기</Text>
                  <Text style={styles.tableCellValue}>{SNOWBALLING_ETF.rebalanceFrequency}</Text>
                </View>
              </View>
            </View>

            {/* Trading Info */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>거래정보</Text>
              <View style={styles.table}>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>종목코드</Text>
                  <Text style={styles.tableCellValue}>{SNOWBALLING_ETF.symbol}</Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>설정단위</Text>
                  <Text style={styles.tableCellValue}>1주</Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>거래단위</Text>
                  <Text style={styles.tableCellValue}>1주</Text>
                </View>
              </View>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>위험지표</Text>
              <View style={styles.table}>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>변동성 (Volatility)</Text>
                  <Text style={styles.tableCellValue}>{data.volatility?.toFixed(2)}%</Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>샤프지수 (Sharpe)</Text>
                  <Text style={styles.tableCellValue}>{data.sharpe_ratio?.toFixed(2)}</Text>
                </View>
                <View style={styles.tableRow}>
                  <Text style={styles.tableCellLabel}>최대낙폭 (MDD)</Text>
                  <Text style={[styles.tableCellValue, styles.textBlue]}>{data.max_drawdown?.toFixed(2)}%</Text>
                </View>
              </View>
            </View>
          </View>

          {/* Right Column */}
          <View style={{ flex: 1.5 }}>
            {/* Investment Points */}
            <View style={styles.infoBox}>
              <Text style={styles.infoBoxTitle}>✅ 투자 포인트</Text>
              <Text style={styles.infoBoxText}>1. AI 기반 운용 전략으로 성장주에 분산 투자</Text>
              <Text style={styles.infoBoxText}>2. TabPFN 모델을 활용한 3개월 예상 수익률 Top 10 선정</Text>
              <Text style={styles.infoBoxText}>3. 기계적 리밸런싱을 통한 감정 배제 및 위험 관리</Text>
            </View>

            {/* Performance */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>운용성과 (%)</Text>
              <View style={styles.perfTable}>
                {/* Header */}
                <View style={[styles.tableRow, { backgroundColor: "#F9F9F9" }]}>
                  <Text style={styles.perfHeaderCell}>구분</Text>
                  <Text style={styles.perfHeaderCell}>1M</Text>
                  <Text style={styles.perfHeaderCell}>YTD</Text>
                </View>
                {/* Row */}
                <View style={styles.tableRow}>
                  <Text style={[styles.perfCell, { fontFamily: 'NanumBarunGothicBold' }]}>{SNOWBALLING_ETF.symbol}</Text>
                  <Text style={[styles.perfCell, isPositive(data.monthly_return) ? styles.textRed : styles.textBlue]}>
                    {data.monthly_return?.toFixed(2)}
                  </Text>
                  <Text style={[styles.perfCell, isPositive(data.ytd_return) ? styles.textRed : styles.textBlue]}>
                    {data.ytd_return?.toFixed(2)}
                  </Text>
                </View>
              </View>
              <Text style={{ fontSize: 8, color: '#999', marginTop: 4 }}>
                * 과거의 운용실적이 미래의 수익을 보장하는 것은 아닙니다.
              </Text>
            </View>

            {/* Top 10 Holdings */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>상위 10종목 (Top 10)</Text>
              <View style={{ borderTopWidth: 1, borderTopColor: '#333', borderStyle: 'solid' }}>
                <View style={[styles.top10Row, { backgroundColor: '#F9F9F9' }]}>
                  <Text style={[styles.top10Rank, { fontFamily: 'NanumBarunGothicBold' }]}>No.</Text>
                  <Text style={[styles.top10Name, { fontFamily: 'NanumBarunGothicBold' }]}>종목명</Text>
                  <Text style={[styles.top10Ticker, { fontFamily: 'NanumBarunGothicBold' }]}>Ticker</Text>
                  <Text style={[styles.top10Weight, { fontFamily: 'NanumBarunGothicBold' }]}>비중</Text>
                </View>
                {data.compositions.map((comp) => (
                  <View key={comp.rank} style={styles.top10Row}>
                    <Text style={styles.top10Rank}>{comp.rank}</Text>
                    <Text style={styles.top10Name}>{comp.stock_name || comp.ticker}</Text>
                    <Text style={styles.top10Ticker}>{comp.ticker}</Text>
                    <Text style={styles.top10Weight}>{comp.weight.toFixed(1)}%</Text>
                  </View>
                ))}
              </View>
            </View>
          </View>
        </View>

        {/* Footer */}
        <Text style={styles.footer}>
          본 자료는 펀드의 단순 정보제공을 위해 작성된 것으로써, 본 자료를 투자권유의 목적으로 제시하거나 제공할 수 없습니다.
          {"\n"}
          Snowballing Asset Management | http://ahnbi2.suwon.ac.kr/
        </Text>
      </Page>
    </Document>
  )
}

export default FactSheetPDF
