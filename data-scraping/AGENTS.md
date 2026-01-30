# Data Scraping - TradingView 자동화 파이프라인

## 규칙
1. 크롤링 관련 실행은 사용자가 직접 수행 - 실행 가이드라인만 제공

## 폴더 구조

```
data-scraping/
├── tradingview_playwright_scraper_upload.py  # 메인 스크래퍼 (최종 버전)
├── db_service.py                              # DB 연결/업로드 서비스
├── cookies.json                               # 로그인 쿠키 (자동 생성)
├── pyproject.toml                             # 의존성 정의
├── poetry.lock                                # 의존성 잠금
├── downloads/                                 # CSV 다운로드 폴더
└── tradingview_data/                          # 데이터 저장
```

## 핵심 파일 설명

### tradingview_playwright_scraper_upload.py
- TradingView에서 주가 데이터 CSV 다운로드
- 다운로드된 CSV를 원격 MySQL DB에 자동 업로드
- Playwright 기반 브라우저 자동화
- 쿠키 기반 세션 관리

### db_service.py
- SSH 터널 연결 관리
- CSV 파싱 및 DB 업로드
- 테이블 자동 생성 (UPSERT 지원)

## 실행 방법

```bash
# 1. SSH 터널 시작 (필수)
ssh -f -N -L 3306:127.0.0.1:5100 ahnbi2@ahnbi2.suwon.ac.kr

# 2. 스크래퍼 실행
cd data-scraping
poetry run python tradingview_playwright_scraper_upload.py
```

## 환경 변수 (.env)

```
TRADINGVIEW_USERNAME=your_username
TRADINGVIEW_PASSWORD=your_password
UPLOAD_TO_DB=true
USE_EXISTING_TUNNEL=true
```

## 상세 문서

전체 파이프라인 가이드: `.claude/skills/data-scraping-pipeline/skill.md`
