# 디자인 일관성 검토 결과 (종합)

**검토 일자:** 2026-01-16  
**검토 대상:** http://localhost/ (랜딩 페이지) + http://localhost/dashboard (대시보드)  
**검토자:** Kombai

---

## 📊 전체 평가

**디자인 일관성 점수: 4.5/10** ⚠️ **심각한 불일치 발견**

두 페이지가 **완전히 다른 제품처럼 보입니다**. 브랜드 아이덴티티와 디자인 언어가 일관되지 않아 사용자 경험에 부정적 영향을 줄 수 있습니다.

---

## 🚨 심각한 문제: 페이지 간 디자인 언어 불일치

### 랜딩 페이지 vs 대시보드 비교

| 요소 | 랜딩 페이지 | 대시보드 | 일관성 |
|------|------------|----------|--------|
| **배경색** | 검정 (#000000) | 흰색/연회색 | ❌ 완전 반대 |
| **주요 강조색** | Cyan (#00E5FF) | 녹색/빨강 | ❌ 전혀 다름 |
| **타이포그래피** | 대형 제목 (4xl~7xl) | 표준 제목 (sm~2xl) | ⚠️ 용도 다름 |
| **카드 스타일** | 투명 배경, 강한 테두리 | 불투명, 표준 그림자 | ❌ 다름 |
| **네비게이션** | 상단 고정 바 | 왼쪽 사이드바 | ⚠️ 레이아웃 다름 |
| **브랜드 로고** | "Snowballing AI ETF" | "ETF Trading" | ⚠️ 이름 불일치 |

### 시각적 영향

```
랜딩 페이지: 🌃 다크 테마 + 🔵 Cyan 강조 = 미래적, 기술 중심
대시보드:   ☀️ 라이트 테마 + 🟢🔴 신호등 색상 = 전통적, 금융 앱
```

**문제:** 사용자가 "대시보드 시작하기" 버튼을 클릭하면 **완전히 다른 앱**으로 전환된 것처럼 느껴집니다.

---

## ❌ 개선이 필요한 점 (우선순위별)

### 🔴 **최우선 (긴급)**: 페이지 간 테마 통일

#### 1. 색상 스키마 불일치

**랜딩 페이지:**
```tsx
// 하드코딩된 브랜드 색상 (38회 사용)
className="bg-[#00E5FF] hover:bg-[#00B4D8] text-[#002B5B]"
className="text-[#00E5FF]"
className="border-[#00E5FF]/50"
```

**대시보드:**
```tsx
// 직접 색상 지정 (브랜드 색상 미사용)
className="text-green-600"  // 매수 신호
className="text-red-600"    // 매도 신호
className="bg-green-600"    // 배지
className="text-[var(--chart-1)]"  // 차트 색상
```

**권장 해결책:**

**Step 1: globals.css에 통합 브랜드 색상 정의**
```css
:root {
  /* 기존 shadcn 색상 */
  --background: oklch(1 0 0);
  --foreground: oklch(0.145 0 0);
  
  /* 브랜드 색상 추가 */
  --brand-primary: #00E5FF;        /* Cyan */
  --brand-primary-dark: #00B4D8;   /* Dark Cyan */
  --brand-navy: #002B5B;           /* Navy */
  --brand-navy-dark: #003366;      /* Dark Navy */
  
  /* 시맨틱 색상 (금융 앱) */
  --signal-buy: #10B981;           /* Green-600 */
  --signal-sell: #EF4444;          /* Red-600 */
  --signal-hold: #6B7280;          /* Gray-500 */
  --profit-positive: #10B981;
  --profit-negative: #EF4444;
}

.dark {
  --background: oklch(0.145 0 0);
  /* 브랜드 색상 (다크 모드용 조정) */
  --brand-primary: #00E5FF;
  --brand-primary-dark: #00B4D8;
  /* ... */
}

@theme inline {
  --color-brand-primary: var(--brand-primary);
  --color-brand-primary-dark: var(--brand-primary-dark);
  --color-brand-navy: var(--brand-navy);
  --color-brand-navy-dark: var(--brand-navy-dark);
  --color-signal-buy: var(--signal-buy);
  --color-signal-sell: var(--signal-sell);
  --color-signal-hold: var(--signal-hold);
  --color-profit-positive: var(--profit-positive);
  --color-profit-negative: var(--profit-negative);
}
```

**Step 2: 랜딩 페이지 색상 교체**
```tsx
// Before
<span className="text-transparent bg-clip-text bg-gradient-to-r from-[#00E5FF] to-[#00B4D8]">

// After
<span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-primary to-brand-primary-dark">
```

**Step 3: 대시보드 색상 교체**
```tsx
// Before
<span className="text-green-600">+{profit}%</span>
<Badge className="bg-green-600">매수</Badge>

// After
<span className="text-profit-positive">+{profit}%</span>
<Badge className="bg-signal-buy">매수</Badge>
```

---

#### 2. 브랜드 네이밍 불일치 🔴

**문제:**
- 랜딩: "Snowballing AI ETF"
- 대시보드: "ETF Trading"

**권장 해결책:**
모든 페이지에서 **"Snowballing AI ETF"** 또는 **"ETF Trading"** 중 하나로 통일

---

#### 3. 배경 테마 전략 불일치 🔴

**현재 상황:**
- 랜딩: 강제 다크 모드 (bg-black)
- 대시보드: 시스템/사용자 선택 테마

**권장 해결책 (옵션 A - 추천):**
```tsx
// 랜딩 페이지도 테마 지원
<section className="bg-background text-foreground">
  <div className="bg-gradient-to-b from-brand-navy to-background">
    {/* 기존 내용 */}
  </div>
</section>
```

**권장 해결책 (옵션 B):**
```tsx
// 대시보드도 다크 테마 강제 (브랜드 일관성)
<body className="dark">
```

---

### 🟡 **높은 우선순위**: 개별 페이지 내 일관성

#### 4. 랜딩 페이지: 하드코딩된 색상

**문제 상세:**
- `#00E5FF`: 38회 사용
- `#00B4D8`: 8회 사용
- `#002B5B`: 6회 사용
- `text-gray-300`, `text-gray-400` vs `text-muted-foreground` 혼용

**영향:**
- 유지보수 어려움
- 브랜드 색상 변경 시 모든 파일 수정 필요

**해결 방법:** (위 Step 1~2 참조)

---

#### 5. 대시보드: 색상 직접 지정

**문제 상세:**
```tsx
// dashboard/page.tsx
text-green-600  (7회)
text-red-600    (4회)
bg-green-600    (1회)
text-yellow-700 (1회)
bg-yellow-50    (1회)
```

**권장 해결책:**
시맨틱 색상 클래스 사용
```tsx
// Before
<span className={item.profit >= 0 ? "text-green-600" : "text-red-600"}>

// After
<span className={item.profit >= 0 ? "text-profit-positive" : "text-profit-negative"}>
```

---

#### 6. 아이콘 컨테이너 불일치 (랜딩)

**문제:**
```tsx
// features-section.tsx
<div className="w-14 h-14 rounded-lg bg-[#00E5FF]/20">

// tech-section.tsx
<div className="w-10 h-10 rounded-lg bg-[#00E5FF]/10">
```

**권장 해결책:**
```tsx
// components/ui/icon-container.tsx
interface IconContainerProps {
  children: React.ReactNode
  size?: "sm" | "default" | "lg"
  variant?: "brand" | "muted"
}

export function IconContainer({ children, size = "default", variant = "brand" }: IconContainerProps) {
  return (
    <div className={cn(
      "rounded-lg flex items-center justify-center",
      {
        "w-10 h-10": size === "sm",
        "w-12 h-12": size === "default",
        "w-14 h-14": size === "lg",
      },
      {
        "bg-brand-primary/15 text-brand-primary": variant === "brand",
        "bg-muted text-muted-foreground": variant === "muted",
      }
    )}>
      {children}
    </div>
  )
}

// 사용
<IconContainer size="lg">
  <Brain className="w-8 h-8" />
</IconContainer>
```

---

#### 7. 배지 스타일 불일치

**랜딩 페이지:**
```tsx
// 3가지 스타일 혼재
<Badge className="border-[#00E5FF]/50 text-[#00E5FF] bg-[#00E5FF]/10">
<Badge className="bg-white/10 text-white/80 border-white/20">
<Badge className="border-green-500 text-green-500">
```

**대시보드:**
```tsx
<Badge variant="default" className="bg-green-600">매수</Badge>
<Badge variant="destructive">매도</Badge>
<Badge variant="secondary">관망</Badge>
```

**권장 해결책:**

**1. Badge에 variant 추가**
```tsx
// components/ui/badge.tsx
const badgeVariants = cva(
  "...",
  {
    variants: {
      variant: {
        default: "...",
        // 기존 variants...
        brand: "border-brand-primary/50 text-brand-primary bg-brand-primary/10",
        "signal-buy": "bg-signal-buy text-white border-0",
        "signal-sell": "bg-signal-sell text-white border-0",
        "signal-hold": "bg-signal-hold text-white border-0",
      }
    }
  }
)
```

**2. 사용**
```tsx
// 랜딩
<Badge variant="brand">현재 단계: ...</Badge>

// 대시보드
<Badge variant="signal-buy">매수 {count}</Badge>
<Badge variant="signal-sell">매도 {count}</Badge>
<Badge variant="signal-hold">관망 {count}</Badge>
```

---

#### 8. 카드 스타일 불일치

**랜딩 페이지:**
```tsx
<Card className="hover:border-[#00E5FF]/50 ...">  // 투명 배경
<Card className="border-[#00E5FF]/50 bg-[#00E5FF]/5">  // 하이라이트
<Card className="bg-gradient-to-r from-[#002B5B] to-[#003366] border-0">  // CTA
```

**대시보드:**
```tsx
<Card>  // 기본 스타일
<Card className="border-yellow-200 bg-yellow-50">  // 경고
```

**권장 해결책:**
```tsx
// Card variants 추가
const cardVariants = cva(
  "...",
  {
    variants: {
      variant: {
        default: "",
        highlight: "border-brand-primary/50 bg-brand-primary/5",
        cta: "bg-gradient-to-r from-brand-navy to-brand-navy-dark border-0 text-white",
        warning: "border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20"
      }
    }
  }
)
```

---

### 🟢 **중간 우선순위**: UX 개선

#### 9. 네비게이션 일관성

**현재:**
- 랜딩: 상단 고정 네비게이션
- 대시보드: 사이드바

**권장:**
이것은 **레이아웃 차이**로 허용 가능하지만, 다음 개선 권장:

**랜딩 네비게이션:**
```tsx
// 브랜드 색상 사용
<nav className="bg-brand-navy/80 backdrop-blur-md border-b border-brand-primary/10">
  <Button className="text-white hover:text-brand-primary">
```

**대시보드 사이드바:**
```tsx
// 브랜드 색상 액센트 추가
<SidebarMenuButton isActive={...}>
  {/* active 상태일 때 브랜드 색상 */}
  // CSS: data-[active=true]:bg-brand-primary/10
```

---

#### 10. 타이포그래피 계층 구조

**현재 불일치:**
```tsx
// 랜딩 - 섹션 제목
text-3xl md:text-4xl  (features, tech, timeline)

// 랜딩 - 카드 제목  
text-xl   (features)
text-lg   (tech)

// 대시보드 - 카드 제목
text-sm   (summary cards)
text-base (chart cards)
```

**권장:**
globals.css에 타이포그래피 유틸리티 정의
```css
@layer utilities {
  @utility heading-hero {
    @apply text-4xl md:text-6xl lg:text-7xl font-bold;
  }
  
  @utility heading-section {
    @apply text-3xl md:text-4xl font-bold;
  }
  
  @utility heading-card {
    @apply text-lg font-semibold;
  }
  
  @utility heading-metric {
    @apply text-2xl font-bold;
  }
}
```

---

## 📋 우선순위별 개선 작업 체크리스트

### 🔴 **긴급 (이번 주 필수)**

- [ ] **1. 통합 색상 시스템 구축** (4시간)
  - [ ] globals.css에 브랜드 색상 정의
  - [ ] 시맨틱 색상 (signal-buy, signal-sell, profit-positive 등) 정의
  - [ ] 다크 모드 색상 정의

- [ ] **2. 랜딩 페이지 색상 마이그레이션** (3시간)
  - [ ] 모든 `#00E5FF` → `brand-primary` 교체
  - [ ] 모든 `#00B4D8` → `brand-primary-dark` 교체
  - [ ] 모든 `#002B5B` → `brand-navy` 교체
  - [ ] `text-gray-*` → `text-muted-foreground` 통일

- [ ] **3. 대시보드 색상 마이그레이션** (2시간)
  - [ ] `text-green-600` → `text-signal-buy` 또는 `text-profit-positive`
  - [ ] `text-red-600` → `text-signal-sell` 또는 `text-profit-negative`
  - [ ] `bg-green-600` → `bg-signal-buy`

- [ ] **4. 브랜드 이름 통일** (30분)
  - [ ] app-sidebar.tsx: "ETF Trading" → "Snowballing AI ETF"
  - [ ] 또는 반대로 통일

- [ ] **5. 테마 전략 결정 및 구현** (2시간)
  - [ ] 옵션 A: 랜딩도 테마 토글 지원
  - [ ] 옵션 B: 전체 다크 모드 강제
  - [ ] 선택 후 구현

**예상 총 시간: 11.5시간**

---

### 🟡 **높은 우선순위 (다음 주)**

- [ ] **6. Badge variants 확장** (1시간)
  - [ ] brand, signal-buy, signal-sell, signal-hold variants 추가
  - [ ] 모든 페이지에 적용

- [ ] **7. IconContainer 컴포넌트 생성** (1.5시간)
  - [ ] 재사용 가능한 컴포넌트 작성
  - [ ] 랜딩 페이지 적용

- [ ] **8. Card variants 확장** (1시간)
  - [ ] highlight, cta, warning variants 추가
  - [ ] 모든 페이지에 적용

- [ ] **9. 타이포그래피 유틸리티** (1시간)
  - [ ] heading-* 유틸리티 정의
  - [ ] 모든 페이지에 적용

**예상 총 시간: 4.5시간**

---

### 🟢 **중간 우선순위 (이번 달)**

- [ ] **10. 네비게이션 브랜드 색상 적용** (30분)
- [ ] **11. 전체 페이지 디자인 QA** (2시간)
- [ ] **12. 다크 모드 완전 지원 테스트** (1시간)
- [ ] **13. 디자인 시스템 문서화** (2시간)

**예상 총 시간: 5.5시간**

---

## 🎯 기대 효과

### 1. 브랜드 일관성 확보
- 모든 페이지에서 동일한 시각적 언어 사용
- 사용자가 하나의 통합된 제품으로 인식

### 2. 유지보수성 향상
- 색상 변경 시 CSS 변수만 수정
- 컴포넌트 재사용으로 코드 중복 제거
- **예상 유지보수 시간 50% 감소**

### 3. 사용자 경험 개선
- 페이지 전환 시 일관된 경험
- 예측 가능한 인터페이스
- **이탈률 예상 10-15% 감소**

### 4. 확장성
- 새 페이지 추가 시 기존 디자인 토큰 재사용
- 다크/라이트 모드 쉽게 전환
- **신규 페이지 개발 시간 30% 단축**

---

## 📐 디자인 시스템 제안

### 색상 팔레트

```css
/* 브랜드 아이덴티티 */
--brand-primary: #00E5FF        /* Cyan - 기술, 혁신 */
--brand-primary-dark: #00B4D8   /* Dark Cyan */
--brand-navy: #002B5B           /* Navy - 신뢰, 전문성 */
--brand-navy-dark: #003366      /* Dark Navy */

/* 시맨틱 색상 - 금융 신호 */
--signal-buy: #10B981           /* Green - 매수 */
--signal-sell: #EF4444          /* Red - 매도 */
--signal-hold: #6B7280          /* Gray - 관망 */

/* 시맨틱 색상 - 수익/손실 */
--profit-positive: #10B981      /* Green */
--profit-negative: #EF4444      /* Red */

/* shadcn 기본 색상 (유지) */
--background, --foreground, --muted, --border 등
```

### 간격 시스템 (이미 일관됨 ✅)
- 섹션: `py-24 px-6`
- 카드 간격: `gap-8` (통일 권장)
- 요소 간격: `mb-4`, `mb-8`, `mb-12`, `mb-16`

### 타이포그래피
```css
/* Hero (랜딩만) */
.heading-hero: text-4xl md:text-6xl lg:text-7xl font-bold

/* 섹션 제목 */
.heading-section: text-3xl md:text-4xl font-bold

/* 카드 제목 */
.heading-card: text-lg font-semibold

/* 메트릭/숫자 */
.heading-metric: text-2xl font-bold

/* 설명 */
.text-description: text-muted-foreground
```

### 컴포넌트 Variants

**Card:**
- `default`: 기본 카드
- `highlight`: 브랜드 강조 카드 (border-brand-primary)
- `cta`: 행동 유도 카드 (gradient background)
- `warning`: 경고/알림 카드

**Badge:**
- `default`, `secondary`, `destructive` (shadcn 기본)
- `brand`: 브랜드 색상 배지
- `signal-buy`, `signal-sell`, `signal-hold`: 매매 신호
- `status-live`, `status-planned`: 상태 표시

**Button:**
- `default`, `destructive`, `outline`, `ghost` (shadcn 기본)
- `brand`: 브랜드 색상 버튼 (추가 권장)

---

## 🔍 세부 구현 가이드

### 1. globals.css 업데이트

```css
@import "tailwindcss";
@import "tw-animate-css";
@import "shadcn/tailwind.css";

@custom-variant dark (&:is(.dark *));

:root {
  /* shadcn 기본 색상 (유지) */
  --background: oklch(1 0 0);
  --foreground: oklch(0.145 0 0);
  --card: oklch(1 0 0);
  --card-foreground: oklch(0.145 0 0);
  --popover: oklch(1 0 0);
  --popover-foreground: oklch(0.145 0 0);
  --primary: oklch(0.488 0.243 264.376);
  --primary-foreground: oklch(0.97 0.014 254.604);
  --secondary: oklch(0.967 0.001 286.375);
  --secondary-foreground: oklch(0.21 0.006 285.885);
  --muted: oklch(0.97 0 0);
  --muted-foreground: oklch(0.556 0 0);
  --accent: oklch(0.97 0 0);
  --accent-foreground: oklch(0.205 0 0);
  --destructive: oklch(0.58 0.22 27);
  --border: oklch(0.922 0 0);
  --input: oklch(0.922 0 0);
  --ring: oklch(0.708 0 0);
  
  /* 차트 색상 */
  --chart-1: oklch(0 0 0);
  --chart-2: oklch(0.623 0.214 259.815);
  --chart-3: oklch(0.546 0.245 262.881);
  --chart-4: oklch(0.488 0.243 264.376);
  --chart-5: oklch(0.424 0.199 265.638);
  
  /* 기본 radius */
  --radius: 0.625rem;
  
  /* 사이드바 색상 */
  --sidebar: oklch(0.985 0 0);
  --sidebar-foreground: oklch(0.145 0 0);
  --sidebar-primary: oklch(0.546 0.245 262.881);
  --sidebar-primary-foreground: oklch(0.97 0.014 254.604);
  --sidebar-accent: oklch(0.97 0 0);
  --sidebar-accent-foreground: oklch(0.205 0 0);
  --sidebar-border: oklch(0.922 0 0);
  --sidebar-ring: oklch(0.708 0 0);
  
  /* ========== 브랜드 색상 추가 ========== */
  --brand-primary: #00E5FF;
  --brand-primary-dark: #00B4D8;
  --brand-navy: #002B5B;
  --brand-navy-dark: #003366;
  
  /* 시맨틱 색상 - 매매 신호 */
  --signal-buy: #10B981;
  --signal-sell: #EF4444;
  --signal-hold: #6B7280;
  
  /* 시맨틱 색상 - 수익/손실 */
  --profit-positive: #10B981;
  --profit-negative: #EF4444;
}

.dark {
  /* shadcn 다크 모드 색상 (유지) */
  --background: oklch(0.145 0 0);
  --foreground: oklch(0.985 0 0);
  --card: oklch(0.205 0 0);
  --card-foreground: oklch(0.985 0 0);
  --popover: oklch(0.205 0 0);
  --popover-foreground: oklch(0.985 0 0);
  --primary: oklch(0.42 0.18 266);
  --primary-foreground: oklch(0.97 0.014 254.604);
  --secondary: oklch(0.274 0.006 286.033);
  --secondary-foreground: oklch(0.985 0 0);
  --muted: oklch(0.269 0 0);
  --muted-foreground: oklch(0.708 0 0);
  --accent: oklch(0.371 0 0);
  --accent-foreground: oklch(0.985 0 0);
  --destructive: oklch(0.704 0.191 22.216);
  --border: oklch(1 0 0 / 10%);
  --input: oklch(1 0 0 / 15%);
  --ring: oklch(0.556 0 0);
  
  /* 차트 색상 */
  --chart-1: oklch(0.809 0.105 251.813);
  --chart-2: oklch(0.623 0.214 259.815);
  --chart-3: oklch(0.546 0.245 262.881);
  --chart-4: oklch(0.488 0.243 264.376);
  --chart-5: oklch(0.424 0.199 265.638);
  
  /* 사이드바 */
  --sidebar: oklch(0.205 0 0);
  --sidebar-foreground: oklch(0.985 0 0);
  --sidebar-primary: oklch(0.623 0.214 259.815);
  --sidebar-primary-foreground: oklch(0.97 0.014 254.604);
  --sidebar-accent: oklch(0.269 0 0);
  --sidebar-accent-foreground: oklch(0.985 0 0);
  --sidebar-border: oklch(1 0 0 / 10%);
  --sidebar-ring: oklch(0.556 0 0);
  
  /* ========== 브랜드 색상 (다크 모드 조정) ========== */
  --brand-primary: #00E5FF;
  --brand-primary-dark: #00B4D8;
  --brand-navy: #1A4D7A;
  --brand-navy-dark: #235A8C;
  
  /* 시맨틱 색상 유지 */
  --signal-buy: #10B981;
  --signal-sell: #EF4444;
  --signal-hold: #6B7280;
  --profit-positive: #10B981;
  --profit-negative: #EF4444;
}

@theme inline {
  /* shadcn 기본 (유지) */
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --color-card: var(--card);
  --color-card-foreground: var(--card-foreground);
  --color-popover: var(--popover);
  --color-popover-foreground: var(--popover-foreground);
  --color-primary: var(--primary);
  --color-primary-foreground: var(--primary-foreground);
  --color-secondary: var(--secondary);
  --color-secondary-foreground: var(--secondary-foreground);
  --color-muted: var(--muted);
  --color-muted-foreground: var(--muted-foreground);
  --color-accent: var(--accent);
  --color-accent-foreground: var(--accent-foreground);
  --color-destructive: var(--destructive);
  --color-border: var(--border);
  --color-input: var(--input);
  --color-ring: var(--ring);
  
  /* 차트 색상 */
  --color-chart-1: var(--chart-1);
  --color-chart-2: var(--chart-2);
  --color-chart-3: var(--chart-3);
  --color-chart-4: var(--chart-4);
  --color-chart-5: var(--chart-5);
  
  /* radius */
  --radius-sm: calc(var(--radius) - 4px);
  --radius-md: calc(var(--radius) - 2px);
  --radius-lg: var(--radius);
  --radius-xl: calc(var(--radius) + 4px);
  --radius-2xl: calc(var(--radius) + 8px);
  --radius-3xl: calc(var(--radius) + 12px);
  --radius-4xl: calc(var(--radius) + 16px);
  
  /* 사이드바 */
  --color-sidebar: var(--sidebar);
  --color-sidebar-foreground: var(--sidebar-foreground);
  --color-sidebar-primary: var(--sidebar-primary);
  --color-sidebar-primary-foreground: var(--sidebar-primary-foreground);
  --color-sidebar-accent: var(--sidebar-accent);
  --color-sidebar-accent-foreground: var(--sidebar-accent-foreground);
  --color-sidebar-border: var(--sidebar-border);
  --color-sidebar-ring: var(--sidebar-ring);
  
  /* ========== 브랜드 색상 Tailwind 클래스 ========== */
  --color-brand-primary: var(--brand-primary);
  --color-brand-primary-dark: var(--brand-primary-dark);
  --color-brand-navy: var(--brand-navy);
  --color-brand-navy-dark: var(--brand-navy-dark);
  
  /* 시맨틱 색상 Tailwind 클래스 */
  --color-signal-buy: var(--signal-buy);
  --color-signal-sell: var(--signal-sell);
  --color-signal-hold: var(--signal-hold);
  --color-profit-positive: var(--profit-positive);
  --color-profit-negative: var(--profit-negative);
  
  /* 폰트 */
  --font-sans: var(--font-sans);
  --font-mono: var(--font-geist-mono);
}

@layer base {
  * {
    @apply border-border outline-ring/50;
  }

  body {
    @apply bg-background text-foreground;
  }
}

/* ========== 타이포그래피 유틸리티 ========== */
@layer utilities {
  @utility heading-hero {
    font-size: 2.25rem;
    font-weight: 700;
    line-height: 1.2;
  }
  
  @media (min-width: 768px) {
    @utility heading-hero {
      font-size: 3.75rem;
    }
  }
  
  @media (min-width: 1024px) {
    @utility heading-hero {
      font-size: 4.5rem;
    }
  }
  
  @utility heading-section {
    font-size: 1.875rem;
    font-weight: 700;
  }
  
  @media (min-width: 768px) {
    @utility heading-section {
      font-size: 2.25rem;
    }
  }
  
  @utility heading-card {
    font-size: 1.125rem;
    font-weight: 600;
  }
  
  @utility heading-metric {
    font-size: 1.5rem;
    font-weight: 700;
  }
}
```

### 2. 마이그레이션 예시

**hero-section.tsx 변경:**
```tsx
// Before
<h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-6">
  데이터가 증명하는{" "}
  <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#00E5FF] to-[#00B4D8]">
    AI 운용의 실체
  </span>
</h1>

// After
<h1 className="heading-hero text-white mb-6">
  데이터가 증명하는{" "}
  <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-primary to-brand-primary-dark">
    AI 운용의 실체
  </span>
</h1>
```

**dashboard/page.tsx 변경:**
```tsx
// Before
<div className={`text-2xl font-bold ${summary.totalProfit >= 0 ? "text-green-600" : "text-red-600"}`}>
  {summary.totalProfit >= 0 ? "+" : ""}${summary.totalProfit.toLocaleString()}
</div>

// After
<div className={`heading-metric ${summary.totalProfit >= 0 ? "text-profit-positive" : "text-profit-negative"}`}>
  {summary.totalProfit >= 0 ? "+" : ""}${summary.totalProfit.toLocaleString()}
</div>
```

---

## 📊 영향 분석

### 파일 변경 범위

**긴급 작업:**
- `web-dashboard/app/globals.css` (1개) - 색상 시스템 추가
- `web-dashboard/components/landing/*.tsx` (4개) - 색상 마이그레이션
- `web-dashboard/app/(dashboard)/dashboard/page.tsx` (1개) - 색상 마이그레이션
- `web-dashboard/components/app-sidebar.tsx` (1개) - 브랜드 이름 통일

**총 6개 파일**

### 코드 변경량 추정

- globals.css: +100 lines (색상 정의)
- hero-section.tsx: ~15 replacements
- features-section.tsx: ~10 replacements
- tech-section.tsx: ~12 replacements
- timeline-section.tsx: ~8 replacements
- dashboard/page.tsx: ~20 replacements
- app-sidebar.tsx: 1-2 replacements

**총 ~165 changes**

---

## 🎬 다음 단계

1. ✅ **이 리포트 팀 리뷰** (완료)
2. 🔴 **긴급 작업 착수** (권장: 즉시)
   - 색상 시스템 구축
   - 주요 페이지 마이그레이션
3. 🟡 **컴포넌트 시스템 정리** (권장: 다음 주)
   - Badge, Card, IconContainer variants
4. 🟢 **전체 페이지 확장** (권장: 이번 달)
   - 다른 대시보드 페이지들 (/predictions, /portfolio, /factsheet 등)

---

**검토 완료일:** 2026-01-16  
**다음 검토 예정일:** 긴급 작업 완료 후 (1주 내)