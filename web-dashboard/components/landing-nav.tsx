"use client"

import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"

export function LandingNav() {
    return (
        <nav className="fixed top-0 left-0 right-0 z-50 bg-landing-nav-bg/85 backdrop-blur-md border-b border-landing-nav-border">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
                {/* 왼쪽: 로고 */}
                <div className="flex items-center gap-4">
                    <Link href="/" className="flex items-center gap-2">
                        <Image src="/icon.png" alt="Logo" width={32} height={32} />
                        <span className="hidden sm:inline font-semibold text-white">
                            Snowballing AI ETF
                        </span>
                    </Link>
                </div>

                {/* 오른쪽: CTA 버튼 */}
                <Link href="/dashboard">
                    <Button
                        size="sm"
                        className="bg-brand-primary hover:bg-brand-primary-dark text-black font-semibold"
                    >
                        <span className="hidden sm:inline">대시보드</span>
                        <span className="sm:hidden">대시보드</span>
                        <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                </Link>
            </div>
        </nav>
    )
}
