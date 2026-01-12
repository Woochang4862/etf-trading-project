"use client"

import { useState, useEffect } from "react"
import { Download, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { MonthlyFactsheet } from "@/lib/api"

interface PDFDownloadButtonProps {
  data: MonthlyFactsheet
}

export function PDFDownloadButton({ data }: PDFDownloadButtonProps) {
  const [isClient, setIsClient] = useState(false)
  const [PDFComponents, setPDFComponents] = useState<{
    PDFDownloadLink: React.ComponentType<{
      document: React.ReactElement
      fileName: string
      children: (props: { loading: boolean; error: Error | null }) => React.ReactNode
    }>
    FactSheetPDF: React.ComponentType<{ data: MonthlyFactsheet }>
  } | null>(null)

  useEffect(() => {
    setIsClient(true)
    // Dynamically import react-pdf components only on client
    Promise.all([
      import("@react-pdf/renderer"),
      import("./FactSheetPDF"),
    ]).then(([reactPdf, factSheetModule]) => {
      setPDFComponents({
        PDFDownloadLink: reactPdf.PDFDownloadLink as unknown as typeof PDFComponents extends null ? never : NonNullable<typeof PDFComponents>["PDFDownloadLink"],
        FactSheetPDF: factSheetModule.FactSheetPDF,
      })
    })
  }, [])

  const fileName = `SNOWBALL_ETF_FactSheet_${data.year}_${data.month}.pdf`

  // Loading state while client-side hydration or module loading
  if (!isClient || !PDFComponents) {
    return (
      <Button disabled variant="outline">
        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
        PDF 준비중...
      </Button>
    )
  }

  const { PDFDownloadLink, FactSheetPDF } = PDFComponents

  return (
    <PDFDownloadLink
      document={<FactSheetPDF data={data} />}
      fileName={fileName}
    >
      {({ loading, error }) => (
        <Button disabled={loading} variant="outline">
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              생성 중...
            </>
          ) : error ? (
            <>
              <Download className="h-4 w-4 mr-2" />
              오류 - 재시도
            </>
          ) : (
            <>
              <Download className="h-4 w-4 mr-2" />
              PDF 다운로드
            </>
          )}
        </Button>
      )}
    </PDFDownloadLink>
  )
}

export default PDFDownloadButton
