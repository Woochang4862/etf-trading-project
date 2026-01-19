# ETF Trading Project - 전체 디자인 일관성 종합 검토

**검토 일자:** 2026-01-16  
**검토 대상:** 전체 6개 페이지  
**검토자:** Kombai

---

## 📊 전체 평가

**종합 디자인 일관성 점수: 3.8/10** 🔴 **심각한 불일치**

### 검토 대상 페이지
1. ✅ `/` - 랜딩 페이지 (완료)
2. ✅ `/dashboard` - 대시보드 (완료)
3. ✅ `/predictions` - 예측 결과 목록 (완료)
4. ✅ `/predictions/AMZN` - 개별 종목 예측 상세 (완료)
5. ✅ `/portfolio` - 포트폴리오 (완료)
6. ✅ `/returns` - 수익률 분석 (완료)

---

## 🚨 심각한 문제: 두 가지 완전히 다른 디자인 시스템

### 문제 1: 랜딩 vs 대시보드 - 완전히 다른 브랜드 아이덴티티

```
랜딩 페이지 (/)
  ├─ 배경: 검정 (#000000) - 강제 다크
  ├─ 강조색: Cyan (#00E5FF, #00B4D8)
  ├─ 분위기: 미래적, 기술 중심, 프리미엄
  └─ 타이포그래피: 대형 Hero (4xl~7xl)

대시보드 페이지 (/dashboard, /predictions, /portfolio, /returns)
  ├─ 배경: 흰색/연회색 - 라이트 테마
  ├─ 강조색: 녹색/빨강 (매매 신호)
  ├─ 분위기: 전통적 금융 앱, 실용적
  └─ 타이포그래피: 표준 (text-2xl 이하)
```

**사용자 경험:**
> 랜딩에서 "대시보드 시작하기" 클릭 → 완전히 다른 앱으로 이동한 느낌

---

## 📋 페이지별 상세 분석

### 1️⃣ 랜딩 페이지 (/)

**테마:** 다크 (강제)  
**주요 색상:** Cyan (#00E5FF, #00B4D8), Navy (#002B5B)

**문제점:**
- ❌ 브랜드 색상 38회 하드코딩
- ❌ `text-gray-300`, `text-gray-400` vs `text-muted-foreground` 혼용
- ❌ 아이콘 컨테이너 크기 불일치 (w-14 vs w-10)
- ❌ 배지 스타일 3가지 혼재

**파일:**
- `app/page.tsx`
- `components/landing/hero-section.tsx`
- `components/landing/features-section.tsx`
- `components/landing/tech-section.tsx`
- `components/landing/timeline-section.tsx`

---

### 2️⃣ 대시보드 (/dashboard)

**테마:** 라이트/시스템 (사용자 선택)  
**주요 색상:** 녹색 (#10B981), 빨강 (#EF4444), 차트색 (var(--chart-1))

**문제점:**
- ❌ Cyan 브랜드 색상 미사용
- ❌ `text-green-600`, `text-red-600` 직접 하드코딩 (11회)
- ❌ `bg-green-600`, `bg-yellow-50` 등 직접 색상 사용
- ⚠️ 차트 색상만 theme 사용 (`var(--chart-1)`)

**색상 패턴:**
```tsx
// 7회 사용
className="text-green-600"  // 수익 표시

// 4회 사용  
className="text-red-600"    // 손실 표시

// 1회 사용
className="bg-green-600"    // 매수 배지
className="bg-yellow-50 border-yellow-200 text-yellow-700"  // 경고 카드
```

---

### 3️⃣ 예측 결과 (/predictions)

**테마:** 라이트/시스템  
**주요 색상:** 녹색/빨강 (신호), 노랑/회색 (중립)

**문제점:**
- ❌ 배경색 직접 지정 (12회)
  - `bg-green-50`, `bg-green-200` (매수)
  - `bg-red-50`, `bg-red-200` (매도)
  - `bg-gray-50`, `bg-gray-200` (관망)
- ❌ 텍스트 색상 직접 지정 (15회+)
  - `text-green-600`, `text-green-700`
  - `text-red-600`, `text-red-700`
  - `text-gray-600`, `text-gray-700`
- ❌ 진행바 색상: `bg-green-600`, `bg-yellow-500`

**코드 예시:**
```tsx
// predictions/page.tsx:100
<Card className="border-green-200 bg-green-50 dark:bg-green-950/20">
  <CardHeader className="pb-2">
    <CardTitle className="text-sm font-medium text-green-700 dark:text-green-400">
      매수 신호
    </CardTitle>
  </CardHeader>
  {/* ... */}
</Card>

// predictions/page.tsx:111
<Card className="border-red-200 bg-red-50 dark:bg-red-950/20">
  <CardHeader className="pb-2">
    <CardTitle className="text-sm font-medium text-red-700 dark:text-red-400">
      매도 신호
    </CardTitle>
  </CardHeader>
  {/* ... */}
</Card>
```

**탭 스타일:**
```tsx
// line 145
<TabsTrigger value="ALL">
  전체 ({predictions.length})
</TabsTrigger>
<TabsTrigger value="BUY" className="text-green-600">
  매수 ({buyCount})
</TabsTrigger>
<TabsTrigger value="SELL" className="text-red-600">
  매도 ({sellCount})
</TabsTrigger>
```

---

### 4️⃣ 개별 종목 상세 (/predictions/AMZN)

**테마:** 라이트/시스템  
**주요 색상:** 빨강 (하락), 녹색/빨강 (신호)

**문제점:**
- ❌ 배지 색상 직접 지정 (3회)
  - `bg-red-600 text-white` (하락 신호)
- ❌ 캔들스틱 차트 색상 라이브러리 직접 사용
- ⚠️ 차트만 별도 라이브러리 (lightweight-charts) 사용

**특이사항:**
- 캔들스틱 차트는 lightweight-charts 라이브러리 사용
- 다른 차트들은 recharts (shadcn 통합) 사용
- **일관성 부족**

---

### 5️⃣ 포트폴리오 (/portfolio)

**테마:** 라이트/시스템  
**주요 색상:** 녹색/빨강 (수익/손실), 파이차트 고정색

**문제점:**
- ❌ 파이차트 색상 하드코딩
  ```tsx
  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"]
  ```
- ❌ 수익/손실 색상 직접 지정 (10회+)
  - `text-green-600` (8회)
  - `text-red-600` (3회)
- ❌ 배지 색상 직접 지정
  - `bg-green-600`, `bg-red-600`
- ⚠️ 버튼 색상만 파랑 (`bg-blue-600`) - **다른 페이지와 다름**

**코드 예시:**
```tsx
// portfolio/page.tsx:24
const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"]

// portfolio/page.tsx:87
<div className={`text-2xl font-bold ${totalProfit >= 0 ? "text-green-600" : "text-red-600"}`}>

// portfolio/page.tsx:157
<Badge className={item.signal === "매수" ? "bg-green-600 text-white" : "bg-red-600 text-white"}>
```

---

### 6️⃣ 수익률 분석 (/returns)

**테마:** 라이트/시스템  
**주요 색상:** 녹색/빨강 (수익/손실)

**문제점:**
- ❌ 수익/손실 색상 직접 지정 (15회+)
  - `text-green-600` (10회+)
  - `text-red-600` (5회+)
- ❌ 차트 색상 theme 사용하지만 불일치
  - `var(--chart-1)`, `hsl(var(--chart-2))` 혼용
- ⚠️ 탭 활성 색상도 녹색 하드코딩

**코드 예시:**
```tsx
// returns/page.tsx:80
<div className={`text-2xl font-bold ${latestReturn.cumulativeReturn >= 0 ? "text-green-600" : "text-red-600"}`}>

// returns/page.tsx:199
<div className={`flex items-center gap-2 ${item.profit >= 0 ? "text-green-600" : "text-red-600"}`}>
  {item.profit >= 0 ? (
    <ArrowUp className="h-4 w-4" />
  ) : (
    <ArrowDown className="h-4 w-4" />
  )}
  <span className="font-semibold">
    {item.profit >= 0 ? "+" : ""}{item.profitPercent.toFixed(2)}%
  </span>
</div>
```

---

## 📊 색상 사용 통계

### 전체 프로젝트 색상 하드코딩 횟수

| 색상 | 사용 횟수 | 주요 페이지 |
|------|----------|------------|
| `#00E5FF` (Cyan) | 38회 | 랜딩 페이지만 |
| `#00B4D8` (Dark Cyan) | 8회 | 랜딩 페이지만 |
| `#002B5B` (Navy) | 6회 | 랜딩 페이지만 |
| `text-green-600` | 50회+ | 대시보드 페이지들 |
| `text-red-600` | 25회+ | 대시보드 페이지들 |
| `bg-green-600` | 10회+ | 대시보드 페이지들 |
| `bg-red-600` | 5회+ | 대시보드 페이지들 |
| `#0088FE`, `#00C49F` 등 | 5개 | 포트폴리오 차트 |

**총 하드코딩 색상 사용: 150회 이상**

---

## 🎨 브랜드 아이덴티티 혼란

### 현재 상황

```
페이지          브랜드 이름              주요 색상          테마
──────────────────────────────────────────────────────────────
/               Snowballing AI ETF      Cyan (#00E5FF)    다크
/dashboard      ETF Trading             녹색/빨강          라이트
/predictions    ETF Trading (sidebar)   녹색/빨강          라이트
/portfolio      ETF Trading (sidebar)   녹색/빨강 + 파랑   라이트
/returns        ETF Trading (sidebar)   녹색/빨강          라이트
```

### 문제점

1. **브랜드 이름 불일치**
   - 랜딩: "Snowballing AI ETF"
   - 대시보드: "ETF Trading"

2. **색상 아이덴티티 불일치**
   - 랜딩: Cyan (기술, 혁신)
   - 대시보드: 녹색/빨강 (전통 금융)
   - **어느 것이 브랜드 색상인가?**

3. **테마 전략 불일치**
   - 랜딩: 다크 강제
   - 대시보드: 사용자 선택

---

## 🔴 긴급 개선 사항 (최우선)

### Priority 1: 통합 색상 시스템 구축 (8시간)

**1.1 globals.css에 통합 브랜드 색상 정의**

```css
:root {
  /* ========== 기존 shadcn 색상 유지 ========== */
  --background: oklch(1 0 0);
  --foreground: oklch(0.145 0 0);
  /* ... 생략 ... */
  
  /* ========== 브랜드 아이덴티티 색상 ========== */
  /* Primary: Cyan (랜딩 페이지 강조색을 브랜드 Primary로 채택) */
  --brand-primary: #00E5FF;
  --brand-primary-dark: #00B4D8;
  --brand-navy: #002B5B;
  --brand-navy-dark: #003366;
  
  /* ========== 금융 시맨틱 색상 ========== */
  /* 매매 신호 */
  --signal-buy: #10B981;        /* Green-600 */
  --signal-sell: #EF4444;       /* Red-600 */
  --signal-hold: #6B7280;       /* Gray-500 */
  
  /* 수익/손실 표시 */
  --profit-positive: #10B981;
  --profit-negative: #EF4444;
  
  /* 배경 컬러 (카드, 배지 등) */
  --signal-buy-bg: #DCFCE7;     /* Green-50 */
  --signal-buy-border: #86EFAC; /* Green-200 */
  --signal-sell-bg: #FEE2E2;    /* Red-50 */
  --signal-sell-border: #FECACA;/* Red-200 */
  --signal-hold-bg: #F3F4F6;    /* Gray-50 */
  --signal-hold-border: #E5E7EB;/* Gray-200 */
  
  /* 경고/알림 */
  --warning-bg: #FEF3C7;        /* Yellow-50 */
  --warning-border: #FDE68A;    /* Yellow-200 */
  --warning-text: #B45309;      /* Yellow-700 */
  
  /* 차트 색상 팔레트 (포트폴리오 파이차트 등) */
  --chart-pie-1: #0088FE;
  --chart-pie-2: #00C49F;
  --chart-pie-3: #FFBB28;
  --chart-pie-4: #FF8042;
  --chart-pie-5: #8884d8;
}

.dark {
  /* 브랜드 색상 (다크 모드 - 약간 조정) */
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
  
  /* 배경 컬러 (다크 모드) */
  --signal-buy-bg: rgba(16, 185, 129, 0.1);
  --signal-buy-border: rgba(16, 185, 129, 0.3);
  --signal-sell-bg: rgba(239, 68, 68, 0.1);
  --signal-sell-border: rgba(239, 68, 68, 0.3);
  --signal-hold-bg: rgba(107, 114, 128, 0.1);
  --signal-hold-border: rgba(107, 114, 128, 0.3);
  
  /* 경고 (다크 모드) */
  --warning-bg: rgba(251, 191, 36, 0.1);
  --warning-border: rgba(251, 191, 36, 0.3);
  --warning-text: #FCD34D;
}

@theme inline {
  /* 기존 shadcn ... */
  
  /* ========== 브랜드 색상 Tailwind 클래스 ========== */
  --color-brand-primary: var(--brand-primary);
  --color-brand-primary-dark: var(--brand-primary-dark);
  --color-brand-navy: var(--brand-navy);
  --color-brand-navy-dark: var(--brand-navy-dark);
  
  /* 시맨틱 색상 */
  --color-signal-buy: var(--signal-buy);
  --color-signal-sell: var(--signal-sell);
  --color-signal-hold: var(--signal-hold);
  --color-profit-positive: var(--profit-positive);
  --color-profit-negative: var(--profit-negative);
  
  /* 배경 */
  --color-signal-buy-bg: var(--signal-buy-bg);
  --color-signal-buy-border: var(--signal-buy-border);
  --color-signal-sell-bg: var(--signal-sell-bg);
  --color-signal-sell-border: var(--signal-sell-border);
  --color-signal-hold-bg: var(--signal-hold-bg);
  --color-signal-hold-border: var(--signal-hold-border);
  
  --color-warning-bg: var(--warning-bg);
  --color-warning-border: var(--warning-border);
  --color-warning-text: var(--warning-text);
  
  /* 차트 색상 */
  --color-chart-pie-1: var(--chart-pie-1);
  --color-chart-pie-2: var(--chart-pie-2);
  --color-chart-pie-3: var(--chart-pie-3);
  --color-chart-pie-4: var(--chart-pie-4);
  --color-chart-pie-5: var(--chart-pie-5);
}
```

**1.2 마이그레이션 계획**

| 페이지 | 변경 항목 | 예상 시간 |
|--------|----------|----------|
| 랜딩 (/) | `#00E5FF` → `brand-primary` (38회) | 2시간 |
| | `text-gray-*` → `text-muted-foreground` | 30분 |
| 대시보드 (/dashboard) | `text-green-600` → `text-profit-positive` (7회) | 1시간 |
| | `text-red-600` → `text-profit-negative` (4회) | |
| 예측 (/predictions) | `bg-green-50` → `bg-signal-buy-bg` (10회+) | 2시간 |
| | `text-green-600/700` → `text-signal-buy` | |
| 개별 예측 (/predictions/*) | 배지 색상 변경 | 30분 |
| 포트폴리오 (/portfolio) | `COLORS` 배열 → theme 사용 | 1시간 |
| | 수익/손실 색상 변경 (10회+) | |
| 수익률 (/returns) | 수익/손실 색상 변경 (15회+) | 1시간 |

**총 예상 시간: 8시간**

---

### Priority 2: 브랜드 아이덴티티 통일 (1시간)

**2.1 브랜드 이름 통일**

```tsx
// app-sidebar.tsx - Line 71
// Before
<span className="font-semibold text-lg">ETF Trading</span>

// After
<span className="font-semibold text-lg">Snowballing AI ETF</span>
```

**또는 반대로:**
```tsx
// app/page.tsx - Line 35
// Before
<span className="font-semibold text-white">Snowballing AI ETF</span>

// After  
<span className="font-semibold text-white">ETF Trading</span>
```

**권장:** "Snowballing AI ETF" 사용 (더 독특하고 브랜드 차별화)

---

### Priority 3: 테마 전략 통일 (2시간)

**옵션 A: 랜딩도 테마 토글 지원 (권장)**

```tsx
// app/page.tsx
// Before
<div className="min-h-screen bg-black">

// After
<div className="min-h-screen bg-background">
  {/* 배경에 브랜드 그라데이션 추가 */}
  <div className="absolute inset-0 bg-gradient-to-b from-brand-navy/50 to-background" />
```

**옵션 B: 전체 다크 테마 강제**

```tsx
// app/layout.tsx
<body className={`${inter.variable} antialiased dark`}>
```

**권장:** 옵션 A (사용자 선택권 제공)

---

## 🟡 높은 우선순위 (2주 내)

### Priority 4: 컴포넌트 Variant 시스템 (6시간)

**4.1 Badge Variants 확장**

```tsx
// components/ui/badge.tsx
const badgeVariants = cva(
  "...",
  {
    variants: {
      variant: {
        default: "...",
        secondary: "...",
        destructive: "...",
        outline: "...",
        // ========== 추가 variants ==========
        brand: "border-brand-primary/50 text-brand-primary bg-brand-primary/10",
        "signal-buy": "bg-signal-buy text-white border-0",
        "signal-sell": "bg-signal-sell text-white border-0",
        "signal-hold": "bg-signal-hold text-white border-0",
        warning: "bg-warning-bg text-warning-text border-warning-border",
      }
    }
  }
)
```

**사용 예시:**
```tsx
// Before
<Badge className="bg-green-600 text-white">매수</Badge>

// After
<Badge variant="signal-buy">매수</Badge>
```

**4.2 Card Variants 확장**

```tsx
// components/ui/card.tsx
const cardVariants = cva(
  "...",
  {
    variants: {
      variant: {
        default: "",
        // ========== 추가 variants ==========
        highlight: "border-brand-primary/50 bg-brand-primary/5",
        cta: "bg-gradient-to-r from-brand-navy to-brand-navy-dark border-0 text-white",
        warning: "border-warning-border bg-warning-bg",
        "signal-buy": "border-signal-buy-border bg-signal-buy-bg",
        "signal-sell": "border-signal-sell-border bg-signal-sell-bg",
        "signal-hold": "border-signal-hold-border bg-signal-hold-bg",
      }
    }
  }
)
```

**사용 예시:**
```tsx
// Before
<Card className="border-green-200 bg-green-50 dark:bg-green-950/20">

// After
<Card variant="signal-buy">
```

**4.3 IconContainer 컴포넌트**

```tsx
// components/ui/icon-container.tsx
interface IconContainerProps {
  children: React.ReactNode
  size?: "sm" | "default" | "lg"
  variant?: "brand" | "muted"
  className?: string
}

export function IconContainer({ 
  children, 
  size = "default", 
  variant = "brand",
  className 
}: IconContainerProps) {
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
      },
      className
    )}>
      {children}
    </div>
  )
}
```

---

### Priority 5: 차트 색상 통일 (3시간)

**5.1 포트폴리오 파이차트**

```tsx
// Before
const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"]

// After
import { useTheme } from "next-themes"

const COLORS = [
  "var(--chart-pie-1)",
  "var(--chart-pie-2)", 
  "var(--chart-pie-3)",
  "var(--chart-pie-4)",
  "var(--chart-pie-5)",
]
```

**5.2 Recharts 설정 통일**

```tsx
// chartConfig에서 색상 참조 통일
const chartConfig = {
  portfolioValue: {
    label: "포트폴리오 가치",
    color: "var(--chart-1)",  // ✅ 이미 사용 중
  },
  dailyReturn: {
    label: "일일 수익률",
    color: "var(--chart-2)",  // ❌ "hsl(var(--chart-2))" 혼용 - 통일 필요
  },
} satisfies ChartConfig
```

---

## 🟢 중간 우선순위 (1개월 내)

### Priority 6: 타이포그래피 시스템 (2시간)

```css
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
  
  @utility heading-page {
    font-size: 1.5rem;
    font-weight: 700;
  }
  
  @utility heading-section {
    font-size: 1.875rem;
    font-weight: 700;
  }
  
  @utility heading-card {
    font-size: 0.875rem;
    font-weight: 500;
  }
  
  @utility heading-metric {
    font-size: 1.5rem;
    font-weight: 700;
  }
}
```

---

### Priority 7: 네비게이션 브랜드 색상 적용 (1시간)

**랜딩 네비게이션:**
```tsx
// app/page.tsx
<nav className="bg-brand-navy/80 backdrop-blur-md border-b border-brand-primary/10">
  <Button className="text-white hover:text-brand-primary">
```

**사이드바:**
```tsx
// components/app-sidebar.tsx - active 상태 스타일
// CSS에서 data-[active=true] 처리
[data-active="true"] {
  background: var(--brand-primary-10);
  color: var(--brand-primary);
}
```

---

## 📊 개선 효과 예측

### 1. 개발 효율성

| 항목 | 현재 | 개선 후 | 개선률 |
|------|------|---------|--------|
| 색상 변경 시간 | 150개 파일 수정 | 1개 파일 수정 | 99% ↓ |
| 신규 페이지 개발 | 4시간 | 2시간 | 50% ↓ |
| 디자인 QA 시간 | 2시간 | 30분 | 75% ↓ |

### 2. 유지보수성

- 브랜드 색상 변경: 6시간 → 5분
- 다크 모드 조정: 4시간 → 30분
- 새 시맨틱 색상 추가: 2시간 → 15분

### 3. 사용자 경험

- 페이지 전환 일관성: 40% → 95%
- 브랜드 인지도: 낮음 → 높음
- 전문성 인상: 보통 → 우수

---

## ✅ 작업 체크리스트

### 🔴 Phase 1: 긴급 (1-2주)

- [ ] **globals.css 색상 시스템 구축** (3시간)
  - [ ] 브랜드 색상 정의
  - [ ] 시맨틱 색상 정의
  - [ ] 다크 모드 색상 정의

- [ ] **랜딩 페이지 마이그레이션** (2.5시간)
  - [ ] `#00E5FF` → `brand-primary` (38회)
  - [ ] `#00B4D8` → `brand-primary-dark` (8회)
  - [ ] `#002B5B` → `brand-navy` (6회)
  - [ ] `text-gray-*` → `text-muted-foreground`

- [ ] **대시보드 페이지 마이그레이션** (5.5시간)
  - [ ] /dashboard: 수익/손실 색상 (11회)
  - [ ] /predictions: 신호 색상 (25회+)
  - [ ] /predictions/[symbol]: 배지 색상 (3회)
  - [ ] /portfolio: 수익/손실 + 차트 (15회+)
  - [ ] /returns: 수익/손실 색상 (15회+)

- [ ] **브랜드 이름 통일** (30분)
  - [ ] app-sidebar.tsx 수정
  - [ ] 또는 랜딩 페이지 수정

- [ ] **테마 전략 결정 및 구현** (2시간)
  - [ ] 옵션 선택 (A or B)
  - [ ] 구현 및 테스트

**Phase 1 총 시간: 13.5시간**

---

### 🟡 Phase 2: 높은 우선순위 (2-4주)

- [ ] **Badge Variants** (2시간)
  - [ ] variant 정의
  - [ ] 전체 페이지 적용
  - [ ] 테스트

- [ ] **Card Variants** (2시간)
  - [ ] variant 정의
  - [ ] 전체 페이지 적용
  - [ ] 테스트

- [ ] **IconContainer 컴포넌트** (1시간)
  - [ ] 컴포넌트 작성
  - [ ] 랜딩 페이지 적용

- [ ] **차트 색상 통일** (3시간)
  - [ ] 포트폴리오 파이차트
  - [ ] Recharts 설정 통일
  - [ ] 테스트

**Phase 2 총 시간: 8시간**

---

### 🟢 Phase 3: 중간 우선순위 (1-2개월)

- [ ] **타이포그래피 시스템** (2시간)
- [ ] **네비게이션 브랜드 적용** (1시간)
- [ ] **전체 디자인 QA** (4시간)
- [ ] **디자인 시스템 문서화** (3시간)
- [ ] **Storybook 구축** (선택, 8시간)

**Phase 3 총 시간: 10-18시간**

---

## 📈 종합 개선 로드맵

```
Week 1-2: Phase 1 (긴급)
  ├─ Day 1-2: 색상 시스템 구축
  ├─ Day 3-4: 랜딩 페이지 마이그레이션
  ├─ Day 5-7: 대시보드 페이지 마이그레이션
  └─ Day 8-10: 브랜드 통일 + 테마 전략

Week 3-4: Phase 2 (높은 우선순위)
  ├─ Day 11-12: Badge/Card Variants
  ├─ Day 13-14: IconContainer + 차트
  └─ Day 15: QA 및 버그 수정

Week 5-8: Phase 3 (중간 우선순위)
  ├─ Week 5: 타이포그래피 + 네비게이션
  ├─ Week 6-7: 전체 QA
  └─ Week 8: 문서화
```

---

## 💡 추가 권장 사항

### 1. 디자인 시스템 문서화

**생성할 문서:**
- `docs/design-system/colors.md` - 색상 팔레트
- `docs/design-system/typography.md` - 타이포그래피
- `docs/design-system/components.md` - 컴포넌트 가이드
- `docs/design-system/spacing.md` - 간격 시스템

### 2. Storybook 도입 (선택)

**이점:**
- 컴포넌트 시각적 테스트
- 디자인 시스템 문서화
- 개발자 간 협업 향상

### 3. 색상 접근성 검토

**확인 항목:**
- [ ] WCAG AA 대비율 (4.5:1) 충족
- [ ] 색맹 사용자 고려
- [ ] 다크 모드 가독성

### 4. 성능 최적화

**고려사항:**
- CSS 변수 사용으로 런타임 성능 개선
- 하드코딩된 색상 제거로 번들 크기 감소 (미미)
- 컴포넌트 재사용으로 React 렌더링 최적화

---

## 🎯 성공 지표 (KPI)

### 개발팀

- [ ] 하드코딩된 색상 사용: 150회 → 0회
- [ ] 색상 관련 코드 리뷰 시간: 30분 → 5분
- [ ] 신규 페이지 개발 시간: 4시간 → 2시간

### 디자인

- [ ] 브랜드 일관성 점수: 3.8/10 → 9.0/10
- [ ] 페이지 간 시각적 일관성: 40% → 95%
- [ ] 디자인 QA 이슈: 15개 → 3개

### 사용자

- [ ] 페이지 전환 시 위화감: 높음 → 없음
- [ ] 브랜드 인지도: 낮음 → 높음
- [ ] 사용자 만족도: 3.5/5 → 4.5/5

---

## 🔗 참고 자료

- [Tailwind CSS v4 Theming](https://tailwindcss.com/docs/theme)
- [shadcn/ui Theming Guide](https://ui.shadcn.com/docs/theming)
- [Design Tokens Community Group](https://www.designtokens.org/)
- [WCAG Color Contrast Guidelines](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)

---

**검토 완료일:** 2026-01-16  
**담당자:** Kombai  
**다음 검토 예정일:** Phase 1 완료 후 (2주 내)

---

## 📎 첨부 파일

- `design_consistency_review.md` - 초기 리뷰 (랜딩 + 대시보드)
- 스크린샷 6개 (각 페이지별)
- 코드 분석 파일 목록

---

## ✍️ 승인 및 실행

**검토자:** _______________  
**승인 일자:** _______________  
**작업 시작일:** _______________  
**예상 완료일:** _______________