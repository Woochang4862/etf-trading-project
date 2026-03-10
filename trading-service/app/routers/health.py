from datetime import datetime

from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """헬스체크"""
    db_status = "ok"
    try:
        from app.database import engine
        with engine.connect() as conn:
            conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="ok",
        trading_mode=settings.trading_mode,
        db=db_status,
        timestamp=datetime.utcnow(),
    )
