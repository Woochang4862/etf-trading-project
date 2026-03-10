from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import TradingCycle, DailyPurchase
from app.schemas import (
    TradingStatusResponse,
    PortfolioResponse,
    TradeExecutionResponse,
    NewCycleRequest,
    NewCycleResponse,
    CycleStatus,
    PurchaseItem,
)
from app.services.cycle_manager import get_cycle_manager
from app.services.holiday_calendar import is_trading_day, get_next_trading_day
from app.services.trade_executor import execute_daily_trading

router = APIRouter(prefix="/api/trading")


@router.get("/status", response_model=TradingStatusResponse)
def get_status(db: Session = Depends(get_db)):
    """현재 사이클 상태, 거래일, 모드 조회"""
    cycle_mgr = get_cycle_manager()
    cycle = (
        db.query(TradingCycle)
        .filter(TradingCycle.is_active == True)
        .order_by(TradingCycle.id.desc())
        .first()
    )

    today = date.today()
    trading_day = is_trading_day(today)
    next_day = get_next_trading_day(today) if not trading_day else None

    cycle_status = None
    total_holdings = 0
    total_invested = 0.0

    if cycle:
        cycle_status = CycleStatus.model_validate(cycle)
        unsold = cycle_mgr.get_unsold_purchases(db, cycle.id)
        total_holdings = len(unsold)
        total_invested = sum(p.total_amount for p in unsold)

    return TradingStatusResponse(
        cycle=cycle_status,
        trading_mode=settings.trading_mode,
        is_trading_day=trading_day,
        next_trading_day=next_day,
        total_holdings=total_holdings,
        total_invested=total_invested,
    )


@router.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio(db: Session = Depends(get_db)):
    """현재 보유 내역 (미매도 매수건)"""
    cycle = (
        db.query(TradingCycle)
        .filter(TradingCycle.is_active == True)
        .order_by(TradingCycle.id.desc())
        .first()
    )

    if not cycle:
        return PortfolioResponse()

    cycle_mgr = get_cycle_manager()
    holdings = cycle_mgr.get_unsold_purchases(db, cycle.id)
    items = [PurchaseItem.model_validate(h) for h in holdings]

    return PortfolioResponse(
        cycle_id=cycle.id,
        holdings=items,
        total_invested=sum(h.total_amount for h in holdings),
        total_count=len(holdings),
    )


@router.post("/execute", response_model=TradeExecutionResponse)
async def manual_execute(db: Session = Depends(get_db)):
    """수동 매매 실행 (테스트용)"""
    result = await execute_daily_trading(db)
    return TradeExecutionResponse(**result)


@router.post("/cycle/new", response_model=NewCycleResponse)
def create_new_cycle(req: NewCycleRequest, db: Session = Depends(get_db)):
    """새 사이클 강제 시작"""
    cycle_mgr = get_cycle_manager()
    cycle = cycle_mgr.create_new_cycle(db, req.initial_capital)
    return NewCycleResponse(
        success=True,
        message=f"새 사이클 생성: id={cycle.id}",
        cycle=CycleStatus.model_validate(cycle),
    )
