"use client"

import Link from "next/link"
import { ArrowRight, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-brand-primary/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-brand-primary/5 rounded-full blur-3xl animate-pulse delay-1000" />
        {/* Grid pattern */}

      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-6 text-center py-16">
        {/* Status Badge */}
        <Badge
          variant="outline"
          className="mb-8 border-brand-primary/50 text-brand-primary bg-brand-primary/10 px-4 py-2"
        >
          <Sparkles className="w-4 h-4 mr-2" />
          현재 단계: 파트너사 협업 및 시스템 고도화 중
        </Badge>

        {/* Main Copy */}
        <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-foreground mb-6 leading-tight">
          데이터가 증명하는{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-primary to-brand-primary-dark">
            AI 운용의 실체
          </span>
        </h1>

        {/* Sub Copy */}
        <p className="text-lg md:text-xl text-muted-foreground mb-4 max-w-3xl mx-auto">
          Snowballing AI ETF 개발 프로젝트
        </p>
        <p className="text-base md:text-lg text-muted-foreground mb-12 max-w-2xl mx-auto">
          단순한 아이디어를 넘어, 알고리즘 검증과 규제 대응 설계를 마친
          <br className="hidden md:block" />
          차세대 Active ETF 솔루션입니다.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link href="/dashboard">
            <Button
              size="lg"
              className="bg-brand-primary hover:bg-brand-primary-dark text-brand-navy font-semibold px-8 py-6 text-lg"
            >
              대시보드 시작하기
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
          <Link href="/factsheet">
            <Button
              size="lg"
              variant="outline"
              className="border-white/30 text-white hover:bg-white/10 px-8 py-6 text-lg"
            >
              팩트시트 보기
            </Button>
          </Link>
        </div>

        {/* Tech Badges */}
        <div className="mt-16 flex flex-wrap justify-center gap-3">
          {["AI/ML 기반 예측", "Rolling Cohort 운용", "규제 대응 설계", "실시간 모니터링"].map((tech) => (
            <Badge
              key={tech}
              variant="secondary"
              className="bg-white/10 text-foreground/80 border-white/20 px-4 py-1.5"
            >
              {tech}
            </Badge>
          ))}
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-foreground/30 rounded-full flex items-start justify-center p-2">
          <div className="w-1 h-2 bg-foreground/50 rounded-full animate-scroll" />
        </div>
      </div>
    </section>
  )
}
