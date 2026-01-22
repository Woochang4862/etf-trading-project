from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from app.database import get_remote_db, get_local_db
from app.services.prediction_service import PredictionService
from app.services.dummy_data_service import (
    generate_candlestick_forecast,
    generate_prediction_history,
    get_stock_info
)
from app.schemas import (
    PredictionResponse,
    PredictionListResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionHistoryResponse,
    PredictionWithPerformance,
    CandlestickForecastResponse,
    CandlestickData
)

router = APIRouter()


def get_prediction_service(
    remote_db: Session = Depends(get_remote_db),
    local_db: Session = Depends(get_local_db)
) -> PredictionService:
    return PredictionService(remote_db, local_db)


# NOTE: /batch를 /{symbol}보다 먼저 선언해야 경로 충돌 방지
@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    여러 종목 일괄 예측

    - **symbols**: 예측할 종목 목록 (None이면 전체)
    - **limit**: 최대 종목 수 (기본 10)
    """
    try:
        predictions = service.batch_predict(request.symbols, request.limit)

        return BatchPredictionResponse(
            total=request.limit,
            success=len(predictions),
            failed=request.limit - len(predictions) if request.symbols is None else 0,
            predictions=[PredictionResponse.model_validate(p) for p in predictions]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/{symbol}", response_model=PredictionResponse)
def predict_single(
    symbol: str,
    timeframe: str = Query("D", description="시간프레임"),
    service: PredictionService = Depends(get_prediction_service)
):
    """
    단일 종목 예측 수행

    - **symbol**: 종목 코드 (예: AAPL, NVDA)
    - **timeframe**: 시간프레임 (기본: D=일봉)
    """
    try:
        prediction = service.predict(symbol.upper(), timeframe)
        return PredictionResponse.model_validate(prediction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("", response_model=PredictionListResponse)
def list_predictions(
    symbol: Optional[str] = Query(None, description="종목 필터"),
    limit: int = Query(50, ge=1, le=100, description="조회 수"),
    offset: int = Query(0, ge=0, description="오프셋"),
    service: PredictionService = Depends(get_prediction_service)
):
    """
    저장된 예측 결과 조회

    - **symbol**: 종목 필터 (옵션)
    - **limit**: 조회 수 (기본 50, 최대 100)
    - **offset**: 페이지네이션 오프셋
    """
    predictions = service.get_predictions(symbol, limit, offset)

    return PredictionListResponse(
        count=len(predictions),
        predictions=[PredictionResponse.model_validate(p) for p in predictions]
    )


# NOTE: /history와 /forecast를 /{prediction_id}보다 먼저 선언해야 경로 충돌 방지
@router.get("/history", response_model=PredictionHistoryResponse)
def get_prediction_history(
    symbol: Optional[str] = Query(None, description="종목 필터"),
    days: int = Query(90, ge=1, le=365, description="조회 기간 (일)"),
    service: PredictionService = Depends(get_prediction_service)
):
    """
    예측 히스토리 조회 (더미 데이터)

    - **symbol**: 종목 필터 (옵션, 없으면 전체)
    - **days**: 조회 기간 (기본 90일, 최대 365일)

    3개월 이상 지난 예측은 실제 수익률(actual_return)이 포함됩니다.
    """
    history = generate_prediction_history(
        symbol=symbol.upper() if symbol else None,
        days=days,
        seed=42
    )

    predictions = [
        PredictionWithPerformance(
            id=p["id"],
            symbol=p["symbol"],
            prediction_date=datetime.fromisoformat(p["prediction_date"]),
            target_date=datetime.fromisoformat(p["target_date"]),
            current_close=p["current_close"],
            predicted_close=p["predicted_close"],
            predicted_direction=p["predicted_direction"],
            confidence=p["confidence"],
            rsi_value=p["rsi_value"],
            macd_value=p["macd_value"],
            actual_close=p["actual_close"],
            actual_return=p["actual_return"],
            is_correct=p["is_correct"],
            days_elapsed=p["days_elapsed"],
            has_performance=p["has_performance"]
        )
        for p in history
    ]

    return PredictionHistoryResponse(
        count=len(predictions),
        predictions=predictions
    )


@router.get("/forecast/{symbol}", response_model=CandlestickForecastResponse)
def get_candlestick_forecast(
    symbol: str,
    days: int = Query(90, ge=30, le=365, description="예측 일 수 (MA120 포함시 210일 권장)"),
    current_price: Optional[float] = Query(None, description="현재가 (옵션)")
):
    """
    향후 N일간 캔들스틱 예측 데이터 (더미)

    - **symbol**: 종목 코드
    - **days**: 예측 일 수 (기본 90일, 최대 365일, MA120 포함시 210일 권장)
    - **current_price**: 시작 가격 (옵션, 없으면 샘플 데이터 사용)

    Note: 현재는 더미 데이터입니다. 추후 ML 모델로 교체 예정.
    """
    stock_info = get_stock_info(symbol.upper())
    price = current_price if current_price else stock_info["current_price"]

    candles_data = generate_candlestick_forecast(
        symbol=symbol.upper(),
        current_price=price,
        days=days
    )

    candles = [CandlestickData(**c) for c in candles_data]

    return CandlestickForecastResponse(
        symbol=symbol.upper(),
        current_price=price,
        forecast_days=days,
        data=candles,
        generated_at=datetime.utcnow()
    )


@router.get("/latest/{symbol}", response_model=PredictionResponse)
def get_latest_prediction(
    symbol: str,
    service: PredictionService = Depends(get_prediction_service)
):
    """특정 종목의 최신 예측 조회"""
    prediction = service.get_latest_prediction(symbol.upper())

    if not prediction:
        raise HTTPException(status_code=404, detail=f"No prediction found for {symbol}")

    return PredictionResponse.model_validate(prediction)


@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(
    prediction_id: int,
    service: PredictionService = Depends(get_prediction_service)
):
    """예측 결과 상세 조회"""
    prediction = service.get_prediction_by_id(prediction_id)

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return PredictionResponse.model_validate(prediction)
