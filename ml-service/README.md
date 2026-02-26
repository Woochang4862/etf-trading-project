# ML Service - ETF Ranking Prediction API

AhnLab LightGBM LambdaRank 모델 기반 ETF 종목 랭킹 예측 서비스.

## 모델 개요

### AhnLab LightGBM LambdaRank
- **알고리즘**: LightGBM LambdaRank (Learning to Rank)
- **목적**: 전체 종목 간 상대적 순위 예측 (개별 가격 예측이 아님)
- **피처**: 85개 (기술지표 49개 + 엔지니어링 24개 + Z-score 7개 + Rank 5개)
- **학습 방식**: Rolling 2-fold CV 앙상블 (fold별 모델 평균)
- **데이터**: etf2_db_processed (전처리된 피처 DB)

### 매매 로직
순위 기반 비율 매매:
- 1등 (최고 점수) → weight=+1.0 (최대 매수)
- 꼴등 (최저 점수) → weight=-1.0 (최대 매도)
- 중간 순위 → 등분된 비율
- Top-K 컷오프 없이 전체 종목의 순위와 점수를 모두 반환

## Architecture

```
ml-service/
├── app/                    # FastAPI application (serving layer)
│   ├── main.py            # FastAPI entry point
│   ├── config.py          # App settings (processed_db_url, default_model)
│   ├── database.py        # DB engines (remote, processed, local)
│   ├── models.py          # SQLAlchemy ORM (rank, score fields)
│   ├── schemas.py         # Pydantic schemas (RankingResponse)
│   ├── routers/
│   │   └── predictions.py # /ranking endpoint + model management
│   └── services/
│       ├── data_service.py            # MySQL data access (etf2_db)
│       ├── processed_data_service.py  # Feature DB access (etf2_db_processed)
│       ├── prediction_service.py      # AhnLab ranking predictions
│       └── model_loader.py            # AhnLab ensemble loader + version mgmt
│
├── scripts/
│   └── train_ahnlab.py   # Model training script
│
├── ml/                    # ML Core (etf-model/src → ml/)
│   ├── models/
│   │   ├── ahnlab_lgbm.py  # AhnLab LambdaRank model
│   │   └── ...
│   └── features/
│       └── pipeline.py    # FeaturePipeline (85 features)
│
└── data/
    ├── predictions.db     # SQLite predictions
    └── models/
        └── ahnlab_lgbm/   # Trained model files
            ├── current -> v20260219
            ├── v20260219/
            │   ├── ahnlab_lgbm_fold0.txt
            │   ├── ahnlab_lgbm_fold1.txt
            │   └── metadata.json
            └── versions.json
```

## 학습

```bash
# Docker 컨테이너 내에서
python scripts/train_ahnlab.py

# GPU 사용
python scripts/train_ahnlab.py --device gpu

# 특정 연도 예측용 학습
python scripts/train_ahnlab.py --pred-year 2025
```

### 버전 관리
- **학습 시**: `v{YYYYMMDD}` 디렉토리 생성 → `current` symlink 업데이트
- **서빙 시**: `current` symlink가 가리키는 버전 로드
- **롤백**: `current` symlink를 이전 버전으로 변경하면 즉시 롤백

## API Endpoints

Base URL: `http://localhost:8000`

### Prediction Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predictions/ranking` | 전체 종목 순위 예측 (주요 엔드포인트) |
| POST | `/api/predictions/batch` | 일괄 예측 (ranking 위임) |
| POST | `/api/predictions/{symbol}` | 단일 종목 예측 |
| GET | `/api/predictions` | 저장된 예측 결과 조회 |
| GET | `/api/predictions/ranking/latest` | 최신 랭킹 결과 조회 |
| GET | `/api/predictions/latest/{symbol}` | 종목별 최신 예측 |

### Model Management Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/predictions/models` | 사용 가능한 모델 목록 |
| POST | `/api/predictions/models/{name}/load` | 특정 모델 로딩 |
| GET | `/api/predictions/models/current` | 현재 로딩된 모델 정보 |

### 랭킹 예측 응답 형식
```json
{
  "prediction_date": "2026-02-19T10:30:00",
  "timeframe": "D",
  "total_symbols": 101,
  "model_name": "ahnlab_lgbm",
  "model_version": "v20260219",
  "rankings": [
    {"symbol": "NVDA", "rank": 1, "score": 2.345, "direction": "BUY", "weight": 1.0, "current_close": 135.50},
    {"symbol": "AAPL", "rank": 2, "score": 2.100, "direction": "BUY", "weight": 0.98, "current_close": 185.20}
  ]
}
```

## 데이터 흐름

```
etf2_db_processed (MySQL, 85 features per symbol)
  → ProcessedDataService.get_all_latest_features()
    → AhnLabEnsemble.predict() (2-fold average)
      → 점수 정렬 → 순위 부여 → 매매 방향/비율 계산
        → RankingResponse (API)
        → Prediction (SQLite 저장)
```

## 데이터베이스

| DB | 용도 | 테이블 형식 |
|----|------|-------------|
| etf2_db (MySQL) | 원시 OHLCV 데이터 | `{SYMBOL}_{TIMEFRAME}` |
| etf2_db_processed (MySQL) | 전처리된 85개 피처 | `{SYMBOL}_{TIMEFRAME}` |
| predictions.db (SQLite) | 예측 결과 저장 | `predictions` |

## Docker

```bash
# 빌드
docker compose build ml-service

# 실행
docker compose up -d ml-service

# 로그
docker compose logs -f ml-service
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| REMOTE_DB_URL | mysql+pymysql://...etf2_db | 원시 데이터 DB |
| PROCESSED_DB_URL | mysql+pymysql://...etf2_db_processed | 피처 DB |
| LOCAL_DB_PATH | /app/data/predictions.db | SQLite 경로 |
| MODELS_DIR | /app/data/models | 모델 저장 디렉토리 |
| DEFAULT_MODEL | ahnlab_lgbm | 기본 모델 |
| ENABLE_ML_FEATURES | true | ML 피처 활성화 |
| LOG_LEVEL | INFO | 로깅 레벨 |
