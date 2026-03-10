import asyncio
import logging
from datetime import date

from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import OrderLog
from app.services.kis_client import get_kis_client, OrderResult
from app.services.ranking_client import get_ranking_client
from app.services.capital_manager import get_capital_manager
from app.services.cycle_manager import get_cycle_manager
from app.services.holiday_calendar import is_trading_day

logger = logging.getLogger(__name__)

MAX_RETRY = 3
RETRY_DELAYS = [1, 3, 9]  # 지수 백오프


async def _retry_order(coro_factory, etf_code: str, order_type: str) -> OrderResult:
    """주문 재시도 (최대 3회, 지수 백오프)"""
    last_result = OrderResult(success=False, message="재시도 미실행")
    for attempt in range(MAX_RETRY):
        result = await coro_factory()
        if result.success:
            return result
        last_result = result
        logger.warning(
            f"{order_type} 주문 실패 ({etf_code}, 시도 {attempt + 1}/{MAX_RETRY}): "
            f"{result.message}"
        )
        if attempt < MAX_RETRY - 1:
            await asyncio.sleep(RETRY_DELAYS[attempt])
    return last_result


def _log_order(
    db: Session,
    cycle_id: int,
    order_type: str,
    etf_code: str,
    quantity: int,
    result: OrderResult,
    retry_count: int = 0,
):
    """주문 로그 기록"""
    log = OrderLog(
        cycle_id=cycle_id,
        order_type=order_type,
        etf_code=etf_code,
        quantity=quantity,
        price=result.price if result.success else None,
        order_id=result.order_id,
        status="SUCCESS" if result.success else "FAILED",
        error_message=None if result.success else result.message,
        retry_count=retry_count,
    )
    db.add(log)
    db.commit()


async def execute_daily_trading(db: Session = None) -> dict:
    """
    일일 매매 실행 오케스트레이터.
    Returns: 실행 결과 요약 dict
    """
    own_session = False
    if db is None:
        db = SessionLocal()
        own_session = True

    try:
        today = date.today()

        # 1. 거래일 확인
        if not is_trading_day(today):
            logger.info(f"{today}는 휴장일 — 매매 스킵")
            return {
                "success": True,
                "message": f"{today}는 휴장일입니다.",
                "day_number": 0,
                "sold_count": 0,
                "bought_count": 0,
                "sold_total": 0.0,
                "bought_total": 0.0,
            }

        kis = get_kis_client()
        ranking_client = get_ranking_client()
        capital_mgr = get_capital_manager()
        cycle_mgr = get_cycle_manager()

        # 2. 사이클 조회/생성
        balance = await kis.get_balance()
        total_cash = balance.available_cash
        if total_cash <= 0:
            total_cash = 10_000_000  # 기본값 (모의투자)
            logger.warning(f"잔고 조회 불가, 기본 자금 {total_cash:,.0f}원 사용")

        cycle = cycle_mgr.get_or_create_active_cycle(db, initial_capital=total_cash)

        # 3. 거래일 번호
        day = cycle_mgr.get_current_trading_day(cycle)
        cycle_mgr.update_day_number(db, cycle, day)
        logger.info(f"=== 매매 실행: 사이클 {cycle.id}, 거래일 {day} ===")

        # 4. 랭킹 조회
        rankings = await ranking_client.get_daily_ranking(settings.top_n_etfs)
        if not rankings:
            return {
                "success": False,
                "message": "ml-service 랭킹 조회 실패 — 당일 매매 중단",
                "day_number": day,
                "sold_count": 0,
                "bought_count": 0,
                "sold_total": 0.0,
                "bought_total": 0.0,
            }

        sold_count = 0
        sold_total = 0.0
        bought_count = 0
        bought_total = 0.0
        available = 0.0

        # 5. Day >= 64: FIFO 매도
        if day >= settings.cycle_trading_days + 1:
            old_purchases = cycle_mgr.get_purchases_to_sell(db, cycle.id, day)
            for purchase in old_purchases:
                result = await _retry_order(
                    lambda p=purchase: kis.sell_order(p.etf_code, p.quantity),
                    purchase.etf_code,
                    "SELL",
                )
                _log_order(
                    db, cycle.id, "SELL", purchase.etf_code,
                    purchase.quantity, result,
                )

                if result.success:
                    sell_price = result.price if result.price > 0 else purchase.price
                    cycle_mgr.mark_as_sold(
                        db, [purchase.id], sell_price, today
                    )
                    sold_count += 1
                    sold_total += sell_price * purchase.quantity
                else:
                    logger.error(
                        f"매도 최종 실패: {purchase.etf_code} — {result.message}"
                    )

            available = sold_total
            logger.info(f"매도 완료: {sold_count}건, 총 {sold_total:,.0f}원")
        else:
            # Day 1~63: 잔고 기반 매수
            allocation = capital_mgr.calculate_allocation(total_cash)
            available = allocation.strategy_amount
            logger.info(f"축적 구간 (day {day}): 전략자금 {available:,.0f}원")

        # 6. 매수 실행
        if available <= 0:
            logger.warning("매수 가용 금액 없음")
            return {
                "success": True,
                "message": f"day {day}: 매도 {sold_count}건, 매수 가용 금액 없음",
                "day_number": day,
                "sold_count": sold_count,
                "bought_count": 0,
                "sold_total": sold_total,
                "bought_total": 0.0,
            }

        per_etf = capital_mgr.get_per_etf_amount(available, len(rankings))
        purchase_items = []

        for etf in rankings:
            price = etf.current_close or 0
            if price <= 0:
                logger.warning(f"가격 정보 없음: {etf.symbol}, 스킵")
                continue

            qty = capital_mgr.get_quantity(per_etf, price)
            if qty <= 0:
                continue

            result = await _retry_order(
                lambda c=etf.symbol, q=qty: kis.buy_order(c, q),
                etf.symbol,
                "BUY",
            )
            _log_order(db, cycle.id, "BUY", etf.symbol, qty, result)

            if result.success:
                buy_price = result.price if result.price > 0 else price
                purchase_items.append({
                    "etf_code": etf.symbol,
                    "quantity": qty,
                    "price": buy_price,
                })
                bought_count += 1
                bought_total += buy_price * qty
            else:
                # 잘못된 종목이면 스킵, 잔고 부족이면 수량 감소 1회 재시도
                if "잔고" in result.message and qty > 1:
                    reduced_qty = qty // 2
                    retry_result = await kis.buy_order(etf.symbol, reduced_qty)
                    _log_order(
                        db, cycle.id, "BUY", etf.symbol,
                        reduced_qty, retry_result, retry_count=1,
                    )
                    if retry_result.success:
                        buy_price = retry_result.price if retry_result.price > 0 else price
                        purchase_items.append({
                            "etf_code": etf.symbol,
                            "quantity": reduced_qty,
                            "price": buy_price,
                        })
                        bought_count += 1
                        bought_total += buy_price * reduced_qty

        # 매수 기록
        if purchase_items:
            cycle_mgr.record_purchases(db, cycle.id, day, purchase_items)

        summary = (
            f"day {day}: 매도 {sold_count}건({sold_total:,.0f}원), "
            f"매수 {bought_count}건({bought_total:,.0f}원)"
        )
        logger.info(f"=== 매매 완료: {summary} ===")

        return {
            "success": True,
            "message": summary,
            "day_number": day,
            "sold_count": sold_count,
            "bought_count": bought_count,
            "sold_total": sold_total,
            "bought_total": bought_total,
        }

    except Exception as e:
        logger.exception(f"매매 실행 중 오류: {e}")
        return {
            "success": False,
            "message": f"매매 실행 중 오류: {str(e)}",
            "day_number": 0,
            "sold_count": 0,
            "bought_count": 0,
            "sold_total": 0.0,
            "bought_total": 0.0,
        }
    finally:
        if own_session:
            db.close()
