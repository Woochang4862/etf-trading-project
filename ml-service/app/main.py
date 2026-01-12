from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.database import init_local_db
from app.routers import health, data, predictions, factsheet

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # 시작 시
    logger.info("Initializing ETF Trading ML Service...")
    init_local_db()
    logger.info("Local database initialized")
    yield
    # 종료 시
    logger.info("Shutting down ETF Trading ML Service...")


app = FastAPI(
    title="ETF Trading ML Service",
    description="""
## ETF 트레이딩 데이터 파이프라인 API

SSH 터널을 통해 원격 MySQL에서 주가 데이터를 조회하고,
RSI/MACD 기반 예측을 수행합니다.

### 주요 기능
- **데이터 조회**: 원격 DB에서 OHLCV + 기술적 지표 조회
- **예측**: 다음 날 주가 방향 예측
- **저장**: 예측 결과를 로컬 SQLite에 저장
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(factsheet.router, prefix="/api/factsheet", tags=["Factsheet"])


@app.get("/")
def root():
    """API 루트"""
    return {
        "name": "ETF Trading ML Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
