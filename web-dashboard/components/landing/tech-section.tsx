"use client"

import { useEffect, useRef, useState } from "react"
import Link from "next/link"
import { ArrowRight, BarChart3, Code2, Lock, Zap } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface TechItem {
  icon: React.ReactNode
  title: string
  description: string
  badge?: string
}

const techItems: TechItem[] = [
  {
    icon: <BarChart3 className="w-6 h-6" />,
    title: "실시간 백테스팅 결과 시각화",
    description: "모델의 성과를 투명하게 공개하여 투자자 신뢰 확보",
    badge: "라이브",
  },
  {
    icon: <Lock className="w-6 h-6" />,
    title: "규제 대응 자동 제어",
    description: "국내 금융 규제(상관계수 0.7)에 맞춘 자동 포트폴리오 조정",
  },
  {
    icon: <Code2 className="w-6 h-6" />,
    title: "오픈 알고리즘 플랫폼",
    description: "외부 개발자가 알고리즘을 업로드하고 검증받을 수 있는 생태계",
    badge: "예정",
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "실시간 예측 시스템",
    description: "미국 정규장 마감 후 자동 예측 및 리밸런싱 신호 생성",
  },
]

export function TechSection() {
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
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">
            기술 인프라 & <span className="text-brand-primary">투명성</span>
          </h2>
          <p className="text-muted-foreground text-lg">
            Technical Infrastructure - 신뢰와 투명성을 위한 기술 기반
          </p>
        </div>

        {/* Tech Grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-16">
          {techItems.map((item, index) => (
            <Card
              key={index}
              className={`hover:border-brand-primary/50 transition-all duration-500 ${isVisible
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-8"
                }`}
              style={{ transitionDelay: `${index * 100}ms` }}
            >
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="w-10 h-10 rounded-lg bg-brand-primary/10 flex items-center justify-center text-brand-primary">
                    {item.icon}
                  </div>
                  {item.badge && (
                    <Badge
                      variant="outline"
                      className={
                        item.badge === "라이브"
                          ? "border-green-500 text-green-500"
                          : "border-muted-foreground text-muted-foreground"
                      }
                    >
                      {item.badge}
                    </Badge>
                  )}
                </div>
                <CardTitle className="text-lg mt-4">{item.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>{item.description}</CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* CTA Section */}
        <div
          className={`text-center transition-all duration-700 ${isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
            }`}
          style={{ transitionDelay: "400ms" }}
        >
          <Card className="bg-gradient-to-r from-brand-navy to-brand-navy-dark border-0">
            <CardContent className="py-12">
              <h3 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
                AI가 운용하는 ETF의 미래를 경험하세요
              </h3>
              <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
                실시간 예측 결과와 포트폴리오 성과를 직접 확인하고,
                데이터 기반의 투자 인사이트를 얻어보세요.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link href="/dashboard">
                  <Button
                    size="lg"
                    className="bg-brand-primary hover:bg-brand-primary-dark text-brand-navy font-semibold"
                  >
                    대시보드 시작하기
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </Link>
                <Link href="/predictions">
                  <Button
                    size="lg"
                    variant="outline"
                    className="border-white/30 text-foreground hover:bg-white/10"
                  >
                    예측 결과 보기
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
