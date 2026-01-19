"use client"

import { useEffect, useRef, useState } from "react"
import { createChart, ColorType, IChartApi, CandlestickData, Time, CandlestickSeries, HistogramSeries, LineStyle } from "lightweight-charts"

export interface ReferenceLine {
  price: number
  color: string
  label: string
}

interface CandlestickChartProps {
  data: CandlestickData<Time>[]
  height?: number
  className?: string
  referenceLines?: ReferenceLine[]
}

export function CandlestickChart({ data, height = 400, className = "", referenceLines = [] }: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const [isDarkMode, setIsDarkMode] = useState(false)

  useEffect(() => {
    // Check for dark mode
    const checkDarkMode = () => {
      setIsDarkMode(document.documentElement.classList.contains("dark"))
    }
    checkDarkMode()

    // Observer for theme changes
    const observer = new MutationObserver(checkDarkMode)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })

    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!chartContainerRef.current) return

    const colors = isDarkMode
      ? {
        backgroundColor: "#1a1a2e",
        textColor: "#9ca3af",
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderUpColor: "#22c55e",
        borderDownColor: "#ef4444",
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
        gridColor: "#374151",
      }
      : {
        backgroundColor: "#ffffff",
        textColor: "#6b7280",
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderUpColor: "#16a34a",
        borderDownColor: "#dc2626",
        wickUpColor: "#16a34a",
        wickDownColor: "#dc2626",
        gridColor: "#e5e7eb",
      }

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: colors.backgroundColor },
        textColor: colors.textColor,
      },
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      rightPriceScale: {
        borderColor: colors.gridColor,
      },
      timeScale: {
        borderColor: colors.gridColor,
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: 1,
      },
    })

    chartRef.current = chart

    // Add candlestick series (v5 API)
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: colors.upColor,
      downColor: colors.downColor,
      borderUpColor: colors.borderUpColor,
      borderDownColor: colors.borderDownColor,
      wickUpColor: colors.wickUpColor,
      wickDownColor: colors.wickDownColor,
    })

    candlestickSeries.setData(data)

    // Add volume series (optional, v5 API)
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: "#00E5FF",
      priceFormat: {
        type: "volume",
      },
      priceScaleId: "",
    })

    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    })

    const volumeData = data.map((d) => ({
      time: d.time,
      value: (d as { volume?: number }).volume || Math.random() * 10000000,
      color:
        (d.close as number) >= (d.open as number)
          ? "rgba(34, 197, 94, 0.5)"
          : "rgba(239, 68, 68, 0.5)",
    }))

    volumeSeries.setData(volumeData)

    // Add reference lines
    if (referenceLines.length > 0) {
      referenceLines.forEach((line) => {
        candlestickSeries.createPriceLine({
          price: line.price,
          color: line.color,
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: line.label,
        })
      })
    }

    // Fit content
    chart.timeScale().fitContent()

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      chart.remove()
    }
  }, [data, height, isDarkMode, referenceLines])

  return (
    <div ref={chartContainerRef} className={`w-full rounded-lg overflow-hidden ${className}`} />
  )
}
