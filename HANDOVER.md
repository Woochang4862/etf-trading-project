# ETF Trading Pipeline Project - 인수인계 문서

## 1. 프로젝트 개요

**레포지토리**: https://github.com/Ahn-Laboratory/etf-trading-project

### 1.1 프로젝트 목적
ETF 주식 데이터 수집, 분석, 예측을 위한 종합 데이터 파이프라인 시스템입니다. 머신러닝 모델을 활용한 주식 랭킹 예측과 자동화된 데이터 파이프라인을 제공합니다.

### 1.2 핵심 기능
- **데이터 수집**: TradingView에서 실시간 주식 데이터 스크래핑
- **특징 엔지니어링**: 85개 피처 (기술지표 + 거시경제 + 엔지니어링 + Z-score + 랭크)
- **ML 예측**: LightGBM LambdaRank 기반 전체 종목 상대 순위 예측
- **자동화**: Cron 기반 일일 예측, 월간 모델 재학습
- **모니터링**: 실시간 스크래핑 상태 대시보드

### 1.3 기술 스택
| 분야 | 기술 |
|------|------|
| 백엔드 | FastAPI (Python) |
| 프론트엔드 | Next.js 16, TypeScript, shadcn/ui |
| ML/DL | LightGBM, scikit-learn |
| 데이터베이스 | MySQL (원격), SQLite (로컬) |
| 인프라 | Docker, Docker Compose, Nginx |
| 자동화 | Bash, Cron |

---

## 2. 시스템 아키텍처

### 2.1 전체 구조도

``mermaid
graph TD
    %% 스타일 정의
    classDef server fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef proxy fill:#e1bee7,stroke:#4a148c,stroke-width:2px;
    classDef next fill:#000000,stroke:#fff,color:#fff;
    classDef fast fill:#009688,stroke:#004d40,color:#fff;
    classDef db fill:#bbdefb,stroke:#0d47a1,stroke-width:2px;
    classDef cron fill:#fff9c4,stroke:#fbc02d,stroke-dasharray: 5 5;

    %% 외부 사용자
    User([External User])

    %% 프로덕션 서버 서브그래프 (따옴표 추가로 에러 해결)
    subgraph Server ["Production Server (ahnbi2.suwon.ac.kr)"]
        direction TB
        
        %% Nginx
        Nginx["Nginx Reverse Proxy"]:::proxy

        %% 서비스 레이어 (병렬 배치)
        subgraph Apps [Applications]
            direction LR
            WebDash["web-dashboard<br/>Next.js<br/>Port 3000"]:::next
            MLServ["ml-service<br/>FastAPI<br/>Port 8000"]:::fast
            AutoMon["auto-monitoring<br/>Next.js<br/>Port 3001"]:::next
        end

        %% 스크래퍼 (ML 서비스 하위)
        Scraper["scraper-service<br/>FastAPI<br/>Port 8001"]:::fast

        %% 데이터베이스
        MySQL[("MySQL Database (Port 5100)<br/>etf2_db (~500 tables)<br/>etf2_db_processed (features)")]:::db

        %% 자동화
        Cron["Automation (Cron)<br/>- 매일 오전 8시: 전체 종목 예측<br/>- 매월 1일 새벽 3시: 모델 학습"]:::cron
    end

    %% 연결 관계
    User ==> Nginx
    Nginx --> WebDash
    Nginx --> MLServ
    Nginx --> AutoMon
    
    MLServ --> Scraper
    
    %% 레이아웃 정렬을 위한 투명 연결
    Scraper ~~~ MySQL
    MySQL ~~~ Cron
    
    %% 스타일 적용
    class Server server
```

### 2.2 Docker 컨테이너 구성

| 컨테이너 | 포트 | 설명 |
|---------|------|------|
| web-dashboard | 3000 | Next.js 웹 대시보드 |
| ml-service | 8000 | FastAPI ML 예측 서비스 |
| scraper-service | 8001 | FastAPI 스크래핑 서비스 |
| auto-monitoring | 3001 | 스크래핑 모니터링 대시보드 |
| nginx | 80/443 | 리버스 프록시 |

### 2.3 데이터베이스 구조

**원격 MySQL (etf2_db)**
- 호스트: `ahnbi2.suwon.ac.kr:5100` (SSH 터널 통해 접근)
- 약 500개 테이블: `{SYMBOL}_{TIMEFRAME}` (예: AAPL_D, NVDA_1h)
- 컬럼: symbol, timeframe, time, open, high, low, close, volume, rsi, macd

**로컬 SQLite (ml-service/data/predictions.db)**
- 예측 결과 저장

---

## 3. 주요 서비스 상세

### 3.1 ml-service

**위치**: `ml-service/`

**역할**: ML 모델 서빙 및 예측 API 제공

**주요 엔드포인트**:
```
GET  /health                          # 헬스체크
GET  /api/stocks                      # 종목 목록 조회
GET  /api/stocks/{symbol}/history     # 종목 히스토리
POST /api/predictions/ranking         # 전체 종목 랭킹 예측 (주요)
POST /api/predictions/batch           # 일괄 예측
POST /api/predictions/{symbol}        # 단일 종목 예측
GET  /api/predictions                 # 저장된 예측 결과
GET  /api/predictions/ranking/latest  # 최신 랭킹 결과
```

**주요 파일**:
| 파일 | 설명 |
|------|------|
| `app/main.py` | FastAPI 진입점 |
| `app/config.py` | 설정 관리 |
| `app/services/prediction_service.py` | 예측 핵심 로직 |
| `app/services/model_loader.py` | AhnLab 모델 로딩 |
| `app/services/processed_data_service.py` | Processed DB 연동 |

### 3.2 scraper-service

**위치**: `scraper-service/`

**역할**: TradingView 데이터 스크래핑 + 피처 파이프라인

**주요 엔드포인트**:
```
POST /api/jobs/full            # 전체 종목 스크래핑
POST /api/jobs/cancel          # 스크래핑 작업 취소
GET  /api/jobs/status          # 스크래핑 상태
POST /api/jobs/retry           # 실패한 종목 재시도
GET  /api/jobs/logs            # 스크래핑 로그 조회
```

**주요 파일**:
| 파일 | 설명 |
|------|------|
| `app/services/scraper.py` | Docker API 스크래퍼 구현 |
| `app/services/db_service.py` | DB 연동 (Corporate Actions 지원) |
| `app/services/processed_db_service.py` | 피처 처리 DB |

### 3.3 web-dashboard

**위치**: `web-dashboard/`

**역할**: 예측 결과 및 포트폴리오 시각화

**페이지 구성**:
| 경로 | 설명 |
|------|------|
| `/` | 메인 대시보드 |
| `/dashboard` | 대시보드 (중첩 라우트) |
| `/dashboard/predictions` | 예측 결과 페이지 |
| `/dashboard/portfolio` | 포트폴리오 페이지 |
| `/dashboard/returns` | 수익률 분석 |

**기술 스택**: Next.js 16, TypeScript, shadcn/ui (Vega 스타일), Recharts

### 3.4 auto-monitoring

**위치**: `auto-monitoring/`

**역할**: 스크래핑 실시간 모니터링 대시보드

**기능**:
- 스크래핑 상태 (running/partial/completed/idle)
- 진행률 표시
- 심볼별 상태 그리드
- 통계: 다운로드 수, 업로드 수, 총 행 수
- 에러 목록

---

## 4. ML 모델 (AhnLab LightGBM LambdaRank)

**위치**: `etf-model/`

### 4.1 모델 특징
- **알고리즘**: LightGBM LambdaRank
- **피처**: 85개 (기술지표 + 거시경제 + 엔지니어링 + Z-score + 랭크)
- **검증**: 2-fold rolling CV 앙상블
- **데이터 소스**: etf2_db_processed

### 4.2 피처 카테고리
| 카테고리 | 설명 | 파일 |
|---------|------|------|
| Technical | RSI, MACD, 볼린저 밴드 등 | `src/features/technical.py` |
| Momentum | 모멘텀 지표 | `src/features/momentum.py` |
| Volatility | 변동성 지표 | `src/features/volatility.py` |
| Cross-sectional | 크로스 섹션 피처 | `src/features/cross_sectional.py` |
| Returns | 수익률 피처 | `src/features/returns.py` |
| Volume | 거래량 피처 | `src/features/volume.py` |

### 4.3 학습 파이프라인
```
etf-model/src/
├── pipeline.py              # 메인 학습 파이프라인
├── experiment_pipeline.py   # 실험 추적
├── models/
│   ├── factory.py          # 모델 팩토리
│   ├── lightgbm_model.py   # LightGBM 구현
│   └── trainer.py          # 학습 유틸리티
└── features/               # 피처 엔지니어링 모듈
```

---

## 5. 운영 및 배포

### 5.1 서비스 시작/중지

**전체 서비스 시작**:
```bash
./start.sh    # SSH 터널 + Docker 컨테이너 시작
```

**서비스 중지**:
```bash
./stop.sh     # Docker 컨테이너 중지 (SSH 터널 유지)
```

**상태 확인**:
```bash
./status.sh   # 서비스 상태 및 API 헬스체크
```

### 5.2 Docker 명령어

```bash
docker-compose up -d        # 컨테이너 시작
docker-compose down         # 컨테이너 중지
docker-compose logs -f      # 로그 확인
docker-compose build        # 이미지 재빌드
docker-compose ps           # 컨테이너 상태
```

### 5.3 SSH 터널

SSH 터널은 원격 MySQL 데이터베이스에 접근하기 위해 필수입니다.

**수동 시작**:
```bash
ssh -f -N -L 3306:127.0.0.1:5100 ahnbi2@ahnbi2.suwon.ac.kr
```

**상태 확인**:
```bash
pgrep -f "ssh.*3306"  # 프로세스 ID 확인
```

### 5.4 Cron 자동화

**설정된 작업**:
1. **매일 오전 8시**: 전체 종목 예측 (`predict-daily.sh`)
2. **매월 1일 새벽 3시**: 모델 재학습 (`train-monthly.sh`)

**Cron 설정**:
```bash
./scripts/setup-cron.sh
```

**로그 위치**:
- `logs/cron.log` - cron 실행 요약
- `logs/predict-YYYYMMDD.log` - 일일 예측 상세
- `logs/train-YYYYMM.log` - 월간 학습 상세

---

## 6. 접속 정보

### 6.1 서버 접속
- **호스트**: ahnbi2.suwon.ac.kr
- **사용자**: ahnbi2
- **SSH**: `ssh ahnbi2@ahnbi2.suwon.ac.kr`

### 6.2 웹 접속
| 서비스 | URL |
|--------|-----|
| 메인 대시보드 | http://ahnbi2.suwon.ac.kr/ |
| API 문서 | http://ahnbi2.suwon.ac.kr/docs |
| 모니터링 | http://ahnbi2.suwon.ac.kr/monitor |

### 6.3 로컬 개발
| 서비스 | URL |
|--------|-----|
| web-dashboard | http://localhost:3000 |
| ml-service | http://localhost:8000 |
| scraper-service | http://localhost:8001 |
| auto-monitoring | http://localhost:3001 |

---

## 7. 데이터베이스

### 7.1 MySQL 연결

**Docker 컨테이너 내부**:
```
Host: host.docker.internal
Port: 3306
Database: etf2_db
```

**로컬 호스트**:
```
Host: 127.0.0.1
Port: 3306 (SSH 터널 경유)
Database: etf2_db
```

### 7.2 주요 테이블

| 테이블 명명 규칙 | 예시 |
|-----------------|------|
| {SYMBOL}_{TIMEFRAME} | AAPL_D, NVDA_1h, TSLA_4h |
| 시간프레임 | D(일봉), 1h(1시간), 4h(4시간) |

### 7.3 Processed DB

**피처가 저장된 별도 데이터베이스**:
- 기술지표, 거시경제 변수, 엔지니어링 피처 포함
- ml-service에서 예측 시 사용

---

## 8. 트러블슈팅

### 8.1 Docker 관련

**"docker: command not found" (cron 환경)**:
- 스크립트 상단에 PATH 설정 확인:
```bash
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:$PATH"
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

### 8.2 데이터베이스 연결

**MySQL 연결 실패**:
```bash
# SSH 터널 상태 확인
pgrep -f "ssh.*3306"

# 터널 재시작
./start.sh
```

### 8.3 API 관련

**"BATCH_D table not found"**:
- `routers/predictions.py`에서 `/batch` 라우트가 `/{symbol}` 보다 먼저 선언되어 있는지 확인

**CORS 오류**:
- FastAPI CORS 설정에 localhost:3000 허용되어 있는지 확인

### 8.4 로그 확인

**스크��래핑 로그**:
```bash
tail -f scraper-service/tradingview_scraper_upload.log
```

**ML 서비스 로그**:
```bash
docker-compose logs -f ml-service
```

---

## 9. 프로젝트 구조

```
etf-trading-project/
├── docker-compose.yml          # Docker Compose 설정
├── docker-compose.dev.yml      # 개발용 오버라이드
├── start.sh                    # 서비스 시작 스크립트
├── stop.sh                     # 서비스 중지 스크립트
├── status.sh                   # 상태 확인 스크립트
│
├── ml-service/                 # ML 예측 서비스
│   ├── Dockerfile.serving
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── routers/
│   │   └── services/
│   └── data/
│       └── predictions.db
│
├── web-dashboard/              # Next.js 대시보드
│   ├── app/
│   │   ├── page.tsx
│   │   └── (dashboard)/
│   ├── components/
│   │   ├── ui/
│   │   └── app-sidebar.tsx
│   └── lib/
│       ├── api.ts
│       └── data.ts
│
├── scraper-service/            # 스크래핑 서비스
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/
│   │   └── services/
│   └── scripts/
│
├── auto-monitoring/            # 모니터링 대시보드
│   ├── app/
│   │   └── page.tsx
│   ├── components/dashboard/
│   └── lib/
│       ├── scraping-log-parser.ts
│       └── types.ts
│
├── etf-model/                  # ML 모델 개발
│   ├── src/
│   │   ├── features/
│   │   ├── models/
│   │   ├── pipeline.py
│   │   └── experiment_pipeline.py
│   └── requirements.txt
│
├── scripts/                    # 자동화 스크립트
│   ├── predict-daily.sh
│   ├── train-monthly.sh
│   └── setup-cron.sh
│
├── nginx/                      # Nginx 설정
│   └── nginx.conf
│
└── logs/                       # 실행 로그
```

---

## 10. 개발 가이드

### 10.1 로컬 개발 환경 설정

**사전 요구사항**:
- Docker & Docker Compose
- Node.js 18+
- Python 3.10+
- Poetry

**설치 순서**:
1. Git 레포지토리 클론
2. `.env` 파일 설정 (각 서비스별)
3. SSH 터널 시작
4. Docker 컨테이너 시작

### 10.2 웹 대시보드 개발

```bash
cd web-dashboard
npm run dev      # 개발 서버 (http://localhost:3000)
npm run build    # 프로덕션 빌드
npm run start    # 프로덕션 서버
```

### 10.3 ML 모델 개발

```bash
cd etf-model
pip install -r requirements.txt
python src/pipeline.py              # 학습
python scripts/process_features.py  # 피처 처리
```

---

## 11. 참고 문서

| 문서 | 경로 |
|-----|------|
| 프로젝트 메인 문서 | `CLAUDE.md` |
| 아키텍처 개요 | `docs/architecture.md` |
| AhnLab 모델 요약 | `docs/AHNLAB_MODEL_DATA_SUMMARY.md` |
| 학습 파이프라인 | `docs/TRAIN_PIPELINE_INTEGRATION.md` |
| Scraper 서비스 문서 | `scraper-service/CLAUDE.md` |
| Web Dashboard README | `web-dashboard/README.md` |

---

## 12. 연락처 및 지원

**프로젝트 관련 문의사항은 아래 리소스를 참고하세요**:
- GitHub Issues: 프로젝트 레포지토리의 Issues 탭
- CLAUDE.md: 프로젝트 내 개발 가이드라인

---

*문서 작성일: 2026-02-19*
*프로젝트 버전: main 브랜치 기준*
