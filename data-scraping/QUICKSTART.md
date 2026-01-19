# Quick Start Guide - TradingView Scraper

## 1️⃣ 설치 (5분)

```bash
cd /Users/jeong-uchang/etf-trading-project/data-scraping

# 의존성 설치
poetry install

# Playwright 브라우저 설치
playwright install chromium
```

## 2️⃣ 자격증명 설정 (1분)

### 옵션 A: 환경 변수 (권장)

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집 (vim, nano, VS Code 등)
nano .env
```

`.env` 내용:
```
TRADINGVIEW_USERNAME=your_actual_username
TRADINGVIEW_PASSWORD=your_actual_password
```

### 옵션 B: 코드 내 직접 수정

`tradingview_scraper.py` 또는 `test_scraper.py`의 `main()` 함수 수정:

```python
USERNAME = "your_username"
PASSWORD = "your_password"
```

## 3️⃣ 테스트 (2분)

### 로그인 테스트

```bash
# Poetry 환경 활성화
poetry shell

# 로그인 테스트 (브라우저 visible)
python test_scraper.py login
```

성공 시:
```
✓ Login successful!
✓ Test completed successfully!
```

### 단일 종목 테스트 (AAPL)

```bash
# AAPL 1개 종목에 대한 전체 크롤링 테스트
python test_scraper.py stock
```

이 테스트는 AAPL의 6가지 시간프레임 데이터를 다운로드합니다.

## 4️⃣ 전체 실행

### 전체 종목 크롤링

```bash
# headless=False로 디버깅 가능
python tradingview_scraper.py

# 프로덕션: tradingview_scraper.py의 headless=True로 변경 후 실행
```

## 5️⃣ 결과 확인

다운로드된 CSV 파일은 다운로드 폴더 (보통 `~/Downloads/`)에 저장됩니다.

파일명 예시:
- `AAPL_12M.csv`
- `AAPL_1M.csv`
- `AAPL_1W.csv`
- 등...

## 🔧 문제 해결

### "playwright not found" 에러

```bash
playwright install chromium
```

### "ImportError: No module named 'playwright'" 에러

```bash
poetry install
```

### 로그인 실패

1. 자격증명 확인
2. TradingView 계정이 잠겨 있는지 확인
3. headless=False로 설정하여 브라우저 확인

### 요소를 찾을 수 없음

TradingView UI가 변경되었을 수 있음:
1. `headless=False`로 설정
2. 브라우저를 보며 UI 확인
3. 선택자 수정 필요

## 📊 진행 상황 확인

```bash
# 실시간 로그 보기
tail -f tradingview_scraper.log

# 최근 100줄
tail -100 tradingview_scraper.log
```

## 🔄 다음 실행 시

쿠키가 저장되므로 다음부터는 자동으로 로그인됩니다:

```
✓ Loaded 15 cookies from tradingview_cookies.json
✓ Already logged in (cookies valid)
```

쿠키 만료 시 자동으로 재로그인 시도합니다.

## 📝 팁

- 첫 실행은 항상 `headless=False`로 테스트하세요
- 종목 수가 많으면 시간이 오래 걸립니다 (종목당 약 2-3분)
- 네트워크 상황에 따라 실패할 수 있으므로 로그를 모니터링하세요
- Cloudflare 차단 시 잠시 기다렸다가 다시 시도하세요

## 🎯 다음 단계

1. `self.stock_list`에 원하는 종목 추가
2. 필요한 시간프레임만 `self.time_periods`에 남기고 나머지 주석 처리
3. 배포용으로 `headless=True` 설정
4. 예약 작업 (cron) 등록으로 정기 실행
