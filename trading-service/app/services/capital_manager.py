from dataclasses import dataclass

from app.config import settings


@dataclass
class Allocation:
    strategy_amount: float
    fixed_amount: float


class CapitalManager:
    """자금 배분 관리"""

    def calculate_allocation(self, total_balance: float) -> Allocation:
        """전체 자금을 전략(70%) / 고정(30%) 배분"""
        strategy = total_balance * settings.strategy_ratio
        fixed = total_balance * settings.fixed_ratio
        return Allocation(strategy_amount=strategy, fixed_amount=fixed)

    def get_per_etf_amount(self, strategy_amount: float, top_n: int = None) -> float:
        """전략 자금을 종목수로 균등 배분한 단위 금액"""
        if top_n is None:
            top_n = settings.top_n_etfs
        if top_n <= 0:
            return 0.0
        return strategy_amount / top_n

    def get_quantity(self, per_etf_amount: float, price: float) -> int:
        """단위금액으로 매수 가능한 수량 (1주 미만 절사)"""
        if price <= 0:
            return 0
        return int(per_etf_amount // price)


# 싱글턴
_capital_manager = CapitalManager()


def get_capital_manager() -> CapitalManager:
    return _capital_manager
