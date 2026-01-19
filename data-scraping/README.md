# Data Scraping

TradingView 주식 데이터 자동 크롤링

## 📋 옵션 선택

### ✨ 새 버전: 완전 자동화 (권장)

- **파일**: `tradingview_scraper.py`
- **장점**: 로그인 자동화, 쿠키 영속성, Playwright 사용
- **시작**: [QUICKSTART.md](./QUICKSTART.md) 참조
- **문서**: [README_AUTOMATED.md](./README_AUTOMATED.md)

### 📊 기존 버전: Selenium (베이스라인)

- **파일**: `baseline.ipynb`
- **장점**: Jupyter 노트북 인터페이스
- **단점**: 수동 로그인 필요, 쿠키 만료 시 재로그인
- **시작**: 아래 "기존 버전 사용법" 참조

---

## 🚀 새 버전: 완전 자동화 사용법

### 1. 빠른 시작 (5분)

```bash
# 1. 의존성 설치
poetry install
playwright install chromium

# 2. 자격증명 설정
cp .env.example .env
# .env 파일 편집: 사용자명/비밀번호 입력

# 3. 테스트
python test_scraper.py login
python test_scraper.py stock

# 4. 전체 실행
python tradingview_scraper.py
```

### 2. 주요 기능

✅ **완전 자동화 로그인** - 자격증명으로 자동 로그인
✅ **쿠키 영속성** - 세션 유지, 만료 시 자동 재로그인
✅ **Playwright** - Selenium보다 빠르고 안정적
✅ **강력한 오류 처리** - 재시도 로직 및 예외 복구
✅ **실시간 로깅** - 진행 상황 추적

### 3. 문서

- [상세 문서](./README_AUTOMATED.md) - 전체 사용법 및 설정
- [빠른 시작](./QUICKSTART.md) - 5분 퀵스타트 가이드

---

## 📊 기존 버전: Selenium 사용법

### Setup

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate virtual environment:
   ```bash
   poetry shell
   ```

4. Install Jupyter kernel (run inside poetry shell):
   ```bash
   poetry run python -m ipykernel install --user --name=data-scraping
   ```

### Usage

Run the notebook with Jupyter:
```bash
poetry run jupyter notebook baseline.ipynb
```

### Dependencies

- **selenium**: Browser automation for web scraping
- **webdriver-manager**: Automatic ChromeDriver management
- **jupyter**: Jupyter notebook support
- **ipykernel**: Jupyter kernel for Python

### Note

- Make sure Chrome browser is installed on your system
- The notebook uses Selenium to automate TradingView data export
- Cookies are saved to `tradingview_cookies.pkl` after manual login
- **제한사항**: 수동 로그인 필요, 쿠키 만료 시 재로그인 필요
