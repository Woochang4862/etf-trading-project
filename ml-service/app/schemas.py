from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# Stock Data Schemas
class StockDataPoint(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    rsi: Optional[float] = None
    macd: Optional[float] = None


class StockDataResponse(BaseModel):
    symbol: str
    timeframe: str
    count: int
    data: list[StockDataPoint]


class SymbolListResponse(BaseModel):
    count: int
    symbols: list[str]


# Prediction Schemas
class PredictionCreate(BaseModel):
    symbol: str
    prediction_date: datetime
    target_date: datetime
    current_close: float
    predicted_close: float
    predicted_direction: str
    confidence: float
    rsi_value: Optional[float] = None
    macd_value: Optional[float] = None


class PredictionResponse(BaseModel):
    id: int
    symbol: str
    prediction_date: datetime
    target_date: datetime
    current_close: float
    predicted_close: float
    predicted_direction: str
    confidence: float
    rsi_value: Optional[float] = None
    macd_value: Optional[float] = None
    actual_close: Optional[float] = None
    is_correct: Optional[bool] = None

    class Config:
        from_attributes = True


class PredictionListResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class BatchPredictionRequest(BaseModel):
    symbols: Optional[list[str]] = None  # None이면 전체 종목
    limit: int = 10  # 최대 종목 수


class BatchPredictionResponse(BaseModel):
    total: int
    success: int
    failed: int
    predictions: list[PredictionResponse]


# Health Check
class HealthResponse(BaseModel):
    status: str
    remote_db: str
    local_db: str
    timestamp: datetime


# Factsheet Schemas
class ETFCompositionResponse(BaseModel):
    rank: int
    ticker: str
    weight: float
    stock_name: Optional[str] = None
    sector: Optional[str] = None

    class Config:
        from_attributes = True


class ETFMonthlySnapshotResponse(BaseModel):
    id: int
    year: int
    month: int
    snapshot_date: datetime
    nav: Optional[float] = None
    monthly_return: Optional[float] = None
    ytd_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    compositions: list[ETFCompositionResponse] = []

    class Config:
        from_attributes = True


class FactsheetListItem(BaseModel):
    id: int
    year: int
    month: int
    snapshot_date: datetime


class FactsheetListResponse(BaseModel):
    count: int
    factsheets: list[FactsheetListItem]


class FactsheetGenerateRequest(BaseModel):
    year: int
    month: int


class FactsheetGenerateResponse(BaseModel):
    success: bool
    message: str
    snapshot: Optional[ETFMonthlySnapshotResponse] = None
