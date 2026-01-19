"use client"

import { useEffect, useRef, useState } from "react"
import { Check, CircleDot } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface TimelineItem {
  date: string
  title: string
  description: string
  status: "completed" | "current" | "upcoming"
}

const timelineItems: TimelineItem[] = [
  {
    date: "2024.10",
    title: "핵심 모델 개발 완료",
    description: "AI 기반 ETF 운용을 위한 미래 수익률 예측 엔진의 기초 설계를 완료하였습니다.",
    status: "completed",
  },
  {
    date: "2024.12 ~ 2025.01",
    title: "기술 실용성 검증",
    description: "자체 ETF 운용 경진대회를 통해 모델 성능을 고도화하고 객관적인 수익 지표를 확보했습니다.",
    status: "completed",
  },
  {
    date: "2025.01",
    title: "운용 아키텍처 확정",
    description: "선입선출(FIFO) 기반의 '롤링 코호트(Rolling Cohort)' 구조를 설계하여 운용의 안정성을 확보했습니다.",
    status: "completed",
  },
  {
    date: "2025.Q1",
    title: "특허 및 제도적 준비",
    description: "핵심 운용 기술의 특허 출원을 준비 중이며, 국내 금융 규제(상관계수 0.7)에 대응하는 자동 제어 시스템 설계를 마쳤습니다.",
    status: "current",
  },
]

export function TimelineSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const [visibleItems, setVisibleItems] = useState<Set<number>>(new Set())

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const index = parseInt(entry.target.getAttribute("data-index") || "0")
          if (entry.isIntersecting) {
            setVisibleItems((prev) => new Set([...prev, index]))
          }
        })
      },
      { threshold: 0.3 }
    )

    const items = sectionRef.current?.querySelectorAll("[data-index]")
    items?.forEach((item) => observer.observe(item))

    return () => observer.disconnect()
  }, [])

  return (
    <section
      ref={sectionRef}
      className="py-24 px-6"
    >
      <div className="max-w-4xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">
            우리는 이미 <span className="text-brand-primary">실행 중</span>입니다
          </h2>
          <p className="text-muted-foreground text-lg">
            Development Status - 현재까지의 성과
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-16">
          <div className="flex justify-between text-sm text-muted-foreground mb-2">
            <span>프로젝트 진행도</span>
            <span>75%</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-brand-primary to-brand-primary-dark transition-all duration-1000"
              style={{ width: "75%" }}
            />
          </div>
        </div>

        {/* Timeline */}
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-border" />

          <div className="space-y-8">
            {timelineItems.map((item, index) => (
              <div
                key={index}
                data-index={index}
                className={`relative pl-20 transition-all duration-700 ${visibleItems.has(index)
                  ? "opacity-100 translate-x-0"
                  : "opacity-0 -translate-x-8"
                  }`}
                style={{ transitionDelay: `${index * 150}ms` }}
              >
                {/* Timeline dot */}
                <div
                  className={`absolute left-6 w-5 h-5 rounded-full border-2 flex items-center justify-center ${item.status === "completed"
                    ? "bg-brand-primary border-brand-primary"
                    : item.status === "current"
                      ? "bg-background border-brand-primary"
                      : "bg-muted border-muted-foreground"
                    }`}
                >
                  {item.status === "completed" ? (
                    <Check className="w-3 h-3 text-brand-navy" />
                  ) : item.status === "current" ? (
                    <CircleDot className="w-3 h-3 text-brand-primary" />
                  ) : null}
                </div>

                <Card
                  className={`${item.status === "current"
                    ? "border-brand-primary/50 bg-brand-primary/5"
                    : ""
                    }`}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{item.title}</CardTitle>
                      <CardDescription className="text-sm font-medium">
                        {item.date}
                      </CardDescription>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{item.description}</p>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section >
  )
}
