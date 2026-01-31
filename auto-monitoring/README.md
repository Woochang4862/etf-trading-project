# ETF Trading Pipeline Monitor

A real-time monitoring dashboard for the ETF trading data pipeline, built with Next.js 16, TypeScript, and shadcn/ui.

## 🎯 Overview

This dashboard provides live monitoring of:
- **Data Scraping**: Real-time status of TradingView data collection across 101 stocks
- **Model Training**: Monthly ML model retraining pipeline
- **Predictions**: Daily RSI/MACD signal generation

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

The dashboard will be available at `http://localhost:3000`

## 📁 Project Structure

```
auto-monitoring/
├── app/
│   ├── api/
│   │   ├── scraping/status/route.ts    # Scraping status endpoint
│   │   ├── training/status/route.ts    # Training status endpoint
│   │   └── prediction/status/route.ts  # Prediction status endpoint
│   ├── layout.tsx                      # Root layout with fonts
│   └── page.tsx                        # Main dashboard page
├── components/
│   ├── dashboard/
│   │   ├── dashboard-header.tsx        # Dashboard header with gradient design
│   │   ├── stats-overview.tsx          # Statistics cards with number tickers
│   │   ├── scraping-status.tsx         # Main scraping status component
│   │   ├── training-status.tsx         # Training pipeline status
│   │   ├── prediction-status.tsx       # Prediction signals display
│   │   └── symbol-grid.tsx             # 101-stock status grid
│   └── ui/                             # shadcn/ui components
├── hooks/
│   ├── use-scraping-status.ts          # SWR hook for scraping data
│   ├── use-training-status.ts          # SWR hook for training data
│   └── use-prediction-status.ts        # SWR hook for prediction data
├── lib/
│   ├── types.ts                        # TypeScript type definitions
│   ├── constants.ts                    # Constants (symbols, timeframes, etc.)
│   ├── log-parser.ts                   # Log file parser utility
│   └── dummy-data.ts                   # Dummy data generators
└── public/                             # Static assets
```

## 🎨 Design System

### Visual Aesthetic
The dashboard features a **bold, technical aesthetic** with:

- **Typography**: Black headings with gradient text effects, monospace for data
- **Color Palette**:
  - Blue (#3b82f6) for data operations
  - Emerald (#10b981) for success states
  - Red (#ef4444) for errors
  - Violet (#8b5cf6) for predictions
  - Amber (#f59e0b) for training
- **Motion**: Animated pulse indicators for live states, smooth transitions
- **Spatial Design**: Layered gradients, subtle blur effects, border accents

### Component Highlights

#### 1. Scraping Status (Hero Component)
- **Large statistics grid** with color-coded cards
- **Live pulse indicator** when scraping is running
- **Multi-color gradient progress bar** with blur glow
- **Scrollable error list** with type badges and timestamps
- **Session metadata** with emoji icons

#### 2. Stats Overview
- **Number ticker animations** on hover
- **Gradient backgrounds** with corner accents
- **Icon badges** that scale on hover
- **Bottom edge highlights**

#### 3. Symbol Grid
- **Filterable 101-stock grid** (all/in_progress/completed/failed)
- **Timeframe indicators** (4 colored badges per symbol)
- **Status summary bar** with percentage visualization
- **Color-coded status** (pending/running/done/partial/failed)

#### 4. Training & Prediction Status
- **Left border accent** (amber for training, violet for predictions)
- **Schedule cards** (last run vs. next scheduled)
- **Model cards** with status badges and accuracy metrics
- **Signal summary** (buy/hold/sell counts)

## 🔧 API Endpoints

### GET `/api/scraping/status`
Returns real-time scraping status by parsing log files.

**Response:**
```typescript
{
  status: 'idle' | 'running' | 'completed' | 'error' | 'partial',
  lastRun: string | null,
  currentSession: {
    startTime: string,
    headlessMode: boolean,
    dbUploadEnabled: boolean,
    sshTunnelActive: boolean
  } | null,
  progress: {
    totalSymbols: number,
    completedSymbols: number,
    currentSymbol: string | null,
    currentTimeframe: string | null,
    percentage: number
  },
  statistics: {
    totalDownloads: number,
    successfulUploads: number,
    failedDownloads: number,
    totalRowsUploaded: number
  },
  symbols: SymbolScrapingStatus[],
  errors: ScrapingError[]
}
```

### GET `/api/training/status`
Returns model training status (currently dummy data).

### GET `/api/prediction/status`
Returns prediction signals (currently dummy data).

## 📊 Data Flow

1. **Log Parsing**: API routes read log files from the scraping pipeline
2. **SWR Hooks**: Frontend hooks fetch data with auto-refresh (5s for scraping, 30s for others)
3. **Real-time Updates**: Components re-render automatically when data changes
4. **Error Handling**: Loading states, error states, and fallback UI

## 🎯 Log File Integration

The scraping status endpoint parses:
- **Main log**: `/home/ahnbi2/etf-trading-project/data-scraping/tradingview_scraper_upload.log`
- **Downloads**: `/home/ahnbi2/etf-trading-project/data-scraping/downloads/*.csv`

Log format:
```
2026-01-30 21:30:00,123 - INFO - [SESSION_START] Headless: True, DB Upload: True
2026-01-30 21:30:05,456 - INFO - [SYMBOL_START] AAPL
2026-01-30 21:30:10,789 - INFO - [DOWNLOAD] AAPL_12달.csv (250 rows)
2026-01-30 21:30:15,012 - INFO - [UPLOAD] Table: AAPL_D, Rows: 250
2026-01-30 21:30:20,345 - ERROR - [TIMEOUT] NVDA_1주 - Download timeout
```

## 🚀 Future Enhancements

1. **WebSocket Integration**: Real-time push updates instead of polling
2. **Historical Charts**: Recharts integration for trend visualization
3. **Alert System**: Browser notifications for errors
4. **Export Functionality**: Download reports as CSV/PDF
5. **Dark/Light Theme Toggle**: User preference persistence
6. **Real Training/Prediction APIs**: Replace dummy data with actual ML service

## 🛠️ Tech Stack

- **Framework**: Next.js 16 (App Router, Turbopack)
- **Language**: TypeScript
- **UI Components**: shadcn/ui (Vega style)
- **Styling**: Tailwind CSS
- **Icons**: @hugeicons/react
- **Data Fetching**: SWR
- **Animations**: CSS animations, number-ticker

## 📝 Development Notes

- Auto-refresh intervals: 5s (scraping), 30s (training/prediction)
- Log parsing happens server-side for security
- Dummy data for training/prediction (real integration pending)
- Responsive design with mobile breakpoints
- Accessibility: Semantic HTML, ARIA labels

## 🎨 Design Philosophy

This dashboard was built with a **designer-turned-developer** mindset:
- **Bold over timid**: Strong colors, clear hierarchy, confident typography
- **Intentional motion**: Animations serve purpose (live indicators, loading states)
- **Technical aesthetic**: Monospace fonts, tabular numbers, gradient accents
- **Information density**: Maximum data visibility without overwhelming
- **Visual coherence**: Consistent color language across components

---

**Built for**: ETF Trading Pipeline Project
**Status**: Ready for production deployment
**Last Updated**: 2026-01-30
