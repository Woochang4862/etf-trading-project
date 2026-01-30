---
name: data-scraping-pipeline
description: |
  TradingView 데이터 스크래핑 자동화 파이프라인 가이드.
  사용 시점: (1) 스크래퍼 실행 방법 문의 시, (2) 데이터 파이프라인 아키텍처 이해 필요 시,
  (3) DB 업로드 설정/디버깅 시, (4) 스크래핑 자동화 확장 작업 시.
  Triggers: TradingView scraper, data scraping, 주가 데이터, CSV 다운로드, DB 업로드,
  SSH tunnel, 자동화 파이프라인, Playwright.
---

# Data Scraping Pipeline

TradingView에서 주가 데이터를 자동으로 수집하여 MySQL 데이터베이스에 업로드하는 파이프라인.

## 아키텍처 개요

```
[TradingView] ──Playwright──▶ [CSV 다운로드] ──db_service──▶ [MySQL DB]
                                   │                            │
                                   ▼                            ▼
                            downloads/                    etf2_db (원격)
                            - NVDA_D.csv                  - NVDA_D 테이블
                            - AAPL_1h.csv                 - AAPL_1h 테이블
```

## 핵심 파일 구조

```
data-scraping/
├── tradingview_playwright_scraper_upload.py  # 메인 스크래퍼 (최종 버전)
├── db_service.py                              # DB 연결 및 업로드 서비스
├── cookies.json                               # TradingView 로그인 쿠키
├── pyproject.toml                             # Poetry 의존성
├── poetry.lock                                # 의존성 잠금 파일
├── downloads/                                 # CSV 다운로드 폴더
└── tradingview_data/                          # 데이터 저장 폴더
```

## 실행 가이드

### 1. 환경 설정

```bash
cd /Users/jeong-uchang/etf-trading-project/data-scraping

# 의존성 설치
poetry install
playwright install chromium

# 환경변수 설정 (.env 파일)
TRADINGVIEW_USERNAME=your_username
TRADINGVIEW_PASSWORD=your_password
UPLOAD_TO_DB=true
USE_EXISTING_TUNNEL=true
```

### 2. SSH 터널 (DB 업로드 시 필수)

```bash
# 터널 시작 (한 번만 실행)
ssh -f -N -L 3306:127.0.0.1:5100 ahnbi2@ahnbi2.suwon.ac.kr

# 터널 확인
pgrep -f "ssh.*3306"
```

### 3. 스크래퍼 실행

```bash
poetry run python tradingview_playwright_scraper_upload.py
```

## 주요 설정

### TIME_PERIODS (시간대 설정)

```python
TIME_PERIODS = [
    {"name": "12달", "button_text": "1Y", "interval": "1 날"},   # Daily 데이터
    {"name": "1달", "button_text": "1M", "interval": "30 분"},   # 30분 데이터
    {"name": "1주", "button_text": "5D", "interval": "5 분"},    # 5분 데이터
    {"name": "1일", "button_text": "1D", "interval": "1 분"},    # 1분 데이터
]
```

### STOCK_LIST (종목 리스트)

```python
STOCK_LIST = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", ...
]
```

## 데이터 플로우

1. **로그인**: `cookies.json` 사용 (없으면 자동 로그인)
2. **심볼 검색**: `#header-toolbar-symbol-search` 클릭 → 종목 입력
3. **시간대 변경**: 버튼 클릭 (1Y, 1M, 5D, 1D)
4. **CSV 다운로드**: `차트 데이터 다운로드` 메뉴
5. **DB 업로드**: `db_service.upload_csv()` 호출

## DB 테이블 구조

```sql
CREATE TABLE `{symbol}_{timeframe}` (
    `time` DATETIME NOT NULL PRIMARY KEY,
    `symbol` VARCHAR(32) NOT NULL,
    `timeframe` VARCHAR(16) NOT NULL,
    `open` DOUBLE,
    `high` DOUBLE,
    `low` DOUBLE,
    `close` DOUBLE,
    `volume` BIGINT,
    `rsi` DOUBLE,
    `macd` DOUBLE
)
```

## 트러블슈팅

### 로그인 실패
- `cookies.json` 삭제 후 재실행
- CAPTCHA 발생 시 수동 해결 필요 (headless=False로 실행)

### DB 연결 실패
- SSH 터널 확인: `pgrep -f "ssh.*3306"`
- 터널 재시작: `ssh -f -N -L 3306:127.0.0.1:5100 ahnbi2@ahnbi2.suwon.ac.kr`

### 요소 찾기 실패
- TradingView UI 변경 가능성
- `headless=False`로 실행하여 UI 확인

## 자동화 확장 계획

### Cron Job 설정

```bash
# 매일 오전 7시 데이터 수집
0 7 * * * cd /path/to/data-scraping && poetry run python tradingview_playwright_scraper_upload.py >> /var/log/scraper.log 2>&1
```

### 모니터링

```bash
# 로그 확인
tail -f tradingview_scraper_upload.log

# DB 데이터 확인
python -c "from db_service import DatabaseService; db=DatabaseService(); db.connect(); print(db.table_exists('NVDA_D'))"
```

## 관련 문서

- **DB 연결**: `.claude/skills/db-ssh-tunneling/skill.md`
- **프로젝트 개요**: `.claude/skills/ai-etf-project/skill.md`
