"use client"

import { useEffect, useRef, useState } from "react"
import { Brain, Globe, TrendingUp } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface FeatureItem {
  icon: React.ReactNode
  title: string
  description: string
  items: string[]
}

const features: FeatureItem[] = [
  {
    icon: <Brain className="w-8 h-8" />,
    title: "멀티모달 데이터 통합",
    description: "다양한 데이터 소스를 활용한 AI 모델",
    items: [
      "가격 데이터 기반 기술적 분석",
      "뉴스 및 경제지표 통합",
      "SNS 센티먼트 분석",
      "LLM 모델 도입 예정",
    ],
  },
  {
    icon: <TrendingUp className="w-8 h-8" />,
    title: "상품 라인업 확장",
    description: "다양한 시장 상황에 대응하는 상품군",
    items: [
      "나스닥 100 기반 모델",
      "KOSPI/KOSDAQ 특화 상품",
      "인버스(Inverse) 종목 포함",
      "하락장 방어 상품 개발",
    ],
  },
  {
    icon: <Globe className="w-8 h-8" />,
    title: "글로벌 진출",
    description: "K-Fintech 모델의 해외 진출",
    items: [
      "국내 시장 안착 후 확장",
      "미국 시장 진출 타진",
      "아시아(동남아) 시장 검토",
      "글로벌 파트너십 구축",
    ],
  },
]

export function FeaturesSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
        }
      },
      { threshold: 0.2 }
    )

    if (sectionRef.current) {
      observer.observe(sectionRef.current)
    }

    return () => observer.disconnect()
  }, [])

  return (
    <section
      ref={sectionRef}
      className="py-24 px-6 bg-transparent"
    >
      <div className="max-w-6xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            성장 및 다변화 전략
          </h2>
          <p className="text-muted-foreground text-lg">
            Scalability - 확장 가능한 AI ETF 생태계
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card
              key={index}
              className={`hover:border-brand-primary/50 transition-all duration-700 ${isVisible
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-8"
                }`}
              style={{ transitionDelay: `${index * 200}ms` }}
            >
              <CardHeader>
                <div className="w-14 h-14 rounded-lg bg-brand-primary/20 flex items-center justify-center text-brand-primary mb-4">
                  {feature.icon}
                </div>
                <CardTitle className="text-foreground text-xl">{feature.title}</CardTitle>
                <CardDescription className="text-muted-foreground">
                  {feature.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {feature.items.map((item, itemIndex) => (
                    <li
                      key={itemIndex}
                      className="flex items-start gap-2 text-muted-foreground"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-brand-primary mt-2 flex-shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
