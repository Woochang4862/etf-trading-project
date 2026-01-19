"use client"

import { Home, Clock, Layers, Cpu } from "lucide-react"
import { useEffect, useState } from "react"

const tocItems = [
    { title: "소개", id: "hero", icon: Home },
    { title: "타임라인", id: "timeline", icon: Clock },
    { title: "핵심 기능", id: "features", icon: Layers },
    { title: "기술 인프라", id: "tech", icon: Cpu },
]

export function LandingSidebar() {
    const [activeId, setActiveId] = useState("hero")

    const scrollToSection = (id: string) => {
        const element = document.getElementById(id)
        if (element) {
            element.scrollIntoView({ behavior: "smooth" })
            setActiveId(id)
        }
    }

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setActiveId(entry.target.id);
                    }
                });
            },
            {
                rootMargin: "-20% 0px -60% 0px",
                threshold: 0
            }
        )

        tocItems.forEach((item) => {
            const element = document.getElementById(item.id)
            if (element) observer.observe(element)
        })

        return () => observer.disconnect()
    }, [])

    return (
        <div className="fixed left-8 top-1/2 -translate-y-1/2 z-50 flex flex-col items-start gap-6 bg-transparent">
            {tocItems.map((item) => {
                const isActive = activeId === item.id;
                return (
                    <button
                        key={item.title}
                        onClick={() => scrollToSection(item.id)}
                        className={`
              flex flex-row items-center gap-4 transition-all duration-300 ease-out cursor-pointer group
              ${isActive ? "scale-110 opacity-100 translate-x-2" : "scale-100 opacity-40 hover:opacity-70"}
            `}
                    >
                        <div className={`
              flex items-center justify-center rounded-full transition-all duration-300
              ${isActive ? "w-12 h-12 bg-brand-primary text-black shadow-lg shadow-brand-primary/20" : "w-10 h-10 bg-white/10 text-white"}
            `}>
                            <item.icon className={isActive ? "w-6 h-6" : "w-5 h-5"} />
                        </div>
                        <span className={`
              text-sm font-medium transition-colors duration-300 whitespace-nowrap
              ${isActive ? "text-brand-primary font-bold" : "text-white/60"}
            `}>
                            {item.title}
                        </span>
                    </button>
                );
            })}
        </div>
    )
}
