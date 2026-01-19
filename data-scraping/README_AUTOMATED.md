# TradingView Stock Data Scraper - Fully Automated

TradingView 주식 데이터 자동 크롤러 - 로그인 자동화 포함

## 주요 기능

✅ **완전 자동화 로그인** - 수동 개입 없이 자격증명으로 자동 로그인
✅ **쿠키 영속성** - 세션 유지, 만료 시 자동 재로그인
✅ **강력한 오류 처리** - 재시도 로직 및 예외 복구
✅ **진행 상황 추적** - 실시간 로깅 및 진행률 표시
✅ **Cloudflare 우회** - Playwright의 스텔스 모드 활용

## 기술 스택

- **Playwright**: Selenium보다 빠르고 안정적
- **Asyncio**: 비동기 처리로 효율적인 작업
- **Type-safe**: 타입 힌트 포함

## 설치

```bash
# Poetry를 사용하는 경우
poetry install

# 또는 직접 설치
pip install playwright
playwright install chromium
```

## 사용법

### 1. 자격증명 설정

환경 변수로 설정하거나 코드 내에서 직접 설정:

```bash
export TRADINGVIEW_USERNAME="your_username"
export TRADINGVIEW_PASSWORD="your_password"
```

또는 `tradingview_scraper.py`의 `main()` 함수에서 직접 수정:

```python
USERNAME = "your_username"
PASSWORD = "your_password"
```

### 2. 스크래핑 실행

```bash
# 테스트 모드 (브라우저 visible)
python tradingview_scraper.py

# 프로덕션 모드 (headless=True로 코드 수정 후)
python tradingview_scraper.py
```

### 3. 종목 리스트 수정

`tradingview_scraper.py`의 `self.stock_list`를 수정:

```python
self.stock_list = [
    "NVDA", "AAPL", "MSFT", "AMZN",  # 원하는 종목 추가
]
```

## 작동 방식

1. **시작**: 브라우저 시작 및 쿠키 로드
2. **로그인 확인**: 저장된 쿠키로 로그인 상태 확인
3. **자동 로그인**: 필요시 자격증명으로 로그인 및 쿠키 저장
4. **종목 검색**: 각 종목 검색 및 선택
5. **데이터 익스포트**: 6가지 시간프레임 데이터 다운로드
   - 12개월, 1개월, 1주, 1일, 1시간, 10분
6. **반복**: 모든 종목에 대해 반복
7. **요약**: 성공/실패 통계 출력

## 파일 구조

```
data-scraping/
├── tradingview_scraper.py      # 메인 스크래퍼
├── tradingview_cookies.json     # 저장된 쿠키 (자동 생성)
├── tradingview_scraper.log     # 로그 파일 (자동 생성)
├── baseline.ipynb               # 기존 Selenium 버전 (참고용)
└── README.md                   # 이 파일
```

## 로그 확인

```bash
# 실시간 로그 보기
tail -f tradingview_scraper.log

# 최근 로그
tail -100 tradingview_scraper.log
```

## 문제 해결

### 로그인 실패
- 자격증명이 올바른지 확인
- TradingView 계정이 잠겨 있지 않은지 확인
- headless=False로 설정하여 브라우저를 보며 디버깅

### 쿠키 만료
- 자동으로 재로그인 시도됨
- `tradingview_cookies.json` 삭제 후 재실행

### 요소를 찾을 수 없음
- TradingView UI가 변경되었을 수 있음
- headless=False로 설정하여 확인 후 선택자 수정

### Cloudflare 차단
- 스텔스 모드 활성화되어 있음
- 재시도하거나 IP 변경 필요

## 베이스라인(Selenium)과의 차이점

| 기능 | 베이스라인 | 자동화 버전 |
|------|-----------|------------|
| 로그인 | 수동 필요 | 완전 자동화 |
| 쿠키 관리 | 수동 저장/로드 | 자동 관리 |
| 기술 스택 | Selenium | Playwright |
| 오류 복구 | 기본적 | 강력한 재시도 |
| UI 선택자 | XPath (취약) | CSS Selector (안정적) |
| 비동기 처리 | 없음 | Asyncio 사용 |

## 진행 예시

```
2024-01-19 11:00:00 - INFO - Starting TradingView scraper...
2024-01-19 11:00:01 - INFO - Loaded 15 cookies from tradingview_cookies.json
2024-01-19 11:00:02 - INFO - Already logged in (cookies valid)
2024-01-19 11:00:03 - INFO - ============================================================
2024-01-19 11:00:03 - INFO - [1/10] Processing NVDA (10%)
2024-01-19 11:00:03 - INFO - ============================================================
2024-01-19 11:00:03 - INFO - [NVDA] Searching...
2024-01-19 11:00:05 - INFO - [NVDA] ✓ Selected
2024-01-19 11:00:05 - INFO -   [NVDA] Processing 1/6: 12개월
2024-01-19 11:00:06 - INFO -   [NVDA - 12개월] Exporting...
2024-01-19 11:00:09 - INFO -   [NVDA - 12개월] ✓ Exported
...
2024-01-19 11:05:00 - INFO - Scraping complete!
2024-01-19 11:05:00 - INFO - Total: 10 stocks
2024-01-19 11:05:00 - INFO - Successful: 10
2024-01-19 11:05:00 - INFO - Failed: 0
```

## 보안 주의사항

⚠️ **자격증명 보안**:
- 코드 내에 비밀번호를 커밋하지 마세요
- 환경 변수 또는 `.env` 파일 사용
- `.env` 파일을 `.gitignore`에 추가

## 라이선스

이 프로젝트는 ETF Trading Project의 일부입니다.
