from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Prediction(Base):
    """예측 결과 저장 모델 (로컬 SQLite)"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    target_date = Column(DateTime, nullable=False)
    current_close = Column(Float, nullable=False)
    predicted_close = Column(Float, nullable=False)
    predicted_direction = Column(String(10), nullable=False)  # UP or DOWN
    confidence = Column(Float, nullable=False)
    rsi_value = Column(Float, nullable=True)
    macd_value = Column(Float, nullable=True)
    actual_close = Column(Float, nullable=True)  # 나중에 업데이트
    is_correct = Column(Boolean, nullable=True)  # 나중에 업데이트
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.symbol} {self.target_date} {self.predicted_direction}>"


class ETFMonthlySnapshot(Base):
    """월별 ETF 스냅샷 (팩트시트 기준 데이터)"""
    __tablename__ = "etf_monthly_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False, index=True)
    snapshot_date = Column(DateTime, nullable=False)  # 해당 월의 마지막 거래일

    # Performance metrics (equal-weight portfolio)
    nav = Column(Float, nullable=True)  # Net Asset Value (시작 1000 기준)
    monthly_return = Column(Float, nullable=True)  # 월간 수익률 (%)
    ytd_return = Column(Float, nullable=True)  # 연초대비 수익률 (%)

    # Risk metrics
    volatility = Column(Float, nullable=True)  # 변동성 (%)
    sharpe_ratio = Column(Float, nullable=True)  # 샤프 지수
    max_drawdown = Column(Float, nullable=True)  # 최대 낙폭 (%)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    compositions = relationship("ETFComposition", back_populates="snapshot", cascade="all, delete-orphan")

    # Unique constraint on year+month
    __table_args__ = (UniqueConstraint('year', 'month', name='uix_year_month'),)

    def __repr__(self):
        return f"<ETFMonthlySnapshot {self.year}-{self.month:02d}>"


class ETFComposition(Base):
    """ETF 구성 종목 (월별 Top-10)"""
    __tablename__ = "etf_compositions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey('etf_monthly_snapshots.id'), nullable=False)
    rank = Column(Integer, nullable=False)
    ticker = Column(String(20), nullable=False)
    weight = Column(Float, nullable=False, default=10.0)  # Equal weight = 10%

    # Optional: stock info at time of snapshot
    stock_name = Column(String(100), nullable=True)
    sector = Column(String(50), nullable=True)

    # Relationship
    snapshot = relationship("ETFMonthlySnapshot", back_populates="compositions")

    def __repr__(self):
        return f"<ETFComposition {self.ticker} rank={self.rank}>"
