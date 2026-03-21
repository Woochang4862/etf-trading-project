import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# KIS API tr_id 매핑 (해외주식 - 미국 ETF용)
TR_IDS = {
    "paper": {
        "buy": "VTTT1002U",
        "sell": "VTTT1006U",
        "balance": "VTTS3012R",
        "order_status": "VTTS3018R",
        "current_price": "HHDFS76200200",
    },
    "live": {
        "buy": "TTTT1002U",
        "sell": "TTTT1006U",
        "balance": "TTTS3012R",
        "order_status": "TTTS3018R",
        "current_price": "HHDFS76200200",
    },
}

BASE_URLS = {
    "paper": "https://openapivts.koreainvestment.com:29443",
    "live": "https://openapi.koreainvestment.com:9443",
}

# 미국 거래소 코드 매핑
EXCHANGE_CODE_MAP = {
    "NASD": "NASD",   # NASDAQ
    "NYSE": "NYSE",   # New York Stock Exchange
    "AMEX": "AMEX",   # American Stock Exchange (NYSE American)
}

# 주요 ETF의 거래소 매핑 (필요시 확장)
TICKER_EXCHANGE_MAP = {
    "QQQ": "NASD", "TQQQ": "NASD", "SQQQ": "NASD",
    "SPY": "AMEX", "VOO": "AMEX", "IVV": "AMEX",
    "DIA": "AMEX", "IWM": "AMEX", "VTI": "AMEX",
    "XLF": "AMEX", "XLK": "AMEX", "XLE": "AMEX",
    "XLV": "AMEX", "XLI": "AMEX", "XLU": "AMEX",
    "GLD": "AMEX", "SLV": "AMEX", "TLT": "NASD",
    "HYG": "AMEX", "LQD": "AMEX", "EEM": "AMEX",
    "VEA": "AMEX", "VWO": "AMEX", "ARKK": "AMEX",
    "SOXX": "NASD", "SMH": "AMEX", "KWEB": "AMEX",
}


def get_exchange_code(ticker: str) -> str:
    """티커에 해당하는 거래소 코드 반환 (기본값: NASD)"""
    return TICKER_EXCHANGE_MAP.get(ticker.upper(), settings.default_exchange_code)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    price: float = 0.0
    quantity: float = 0.0
    message: str = ""


@dataclass
class BalanceInfo:
    total_evaluation: float = 0.0
    available_cash: float = 0.0
    currency: str = "USD"
    holdings: list = field(default_factory=list)


class KISClient:
    """한국투자증권 해외주식 API 래퍼 (미국 ETF 전용)"""

    def __init__(self):
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._last_call_time: float = 0.0
        self._mode = settings.trading_mode
        self._base_url = BASE_URLS.get(self._mode, BASE_URLS["paper"])
        self._tr_ids = TR_IDS.get(self._mode, TR_IDS["paper"])

        if self._mode == "live" and not settings.kis_live_confirmation:
            raise ValueError(
                "실투자 모드는 KIS_LIVE_CONFIRMATION=true 설정이 필요합니다."
            )

    def _get_headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._access_token}",
            "appkey": settings.kis_app_key,
            "appsecret": settings.kis_app_secret,
            "tr_id": tr_id,
        }

    async def _rate_limit(self):
        """KIS 초당 20건 제한 준수 (50ms 간격)"""
        elapsed = time.time() - self._last_call_time
        if elapsed < 0.05:
            await asyncio.sleep(0.05 - elapsed)
        self._last_call_time = time.time()

    async def get_access_token(self) -> str:
        """OAuth 토큰 발급/캐싱 (24시간 유효)"""
        now = time.time()
        if self._access_token and now < self._token_expires_at - 60:
            return self._access_token

        url = f"{self._base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": settings.kis_app_key,
            "appsecret": settings.kis_app_secret,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        self._access_token = data["access_token"]
        self._token_expires_at = now + 23 * 3600
        logger.info("KIS access token 발급 완료")
        return self._access_token

    async def _ensure_token(self):
        """토큰 확인 및 자동 갱신"""
        await self.get_access_token()

    async def get_balance(self) -> BalanceInfo:
        """해외주식 잔고 조회"""
        await self._ensure_token()
        await self._rate_limit()

        tr_id = self._tr_ids["balance"]
        account_parts = settings.kis_account_number.split("-")
        if len(account_parts) != 2:
            logger.error(f"잘못된 계좌번호 형식: {settings.kis_account_number}")
            return BalanceInfo()

        url = f"{self._base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
        params = {
            "CANO": account_parts[0],
            "ACNT_PRDT_CD": account_parts[1],
            "OVRS_EXCG_CD": settings.default_exchange_code,
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    url, headers=self._get_headers(tr_id), params=params
                )
                resp.raise_for_status()
                data = resp.json()

            if data.get("rt_cd") != "0":
                logger.error(f"잔고 조회 실패: {data.get('msg1', 'Unknown error')}")
                return BalanceInfo()

            output2 = data.get("output2", {})
            # 해외주식 잔고 응답: output2에 요약, output1에 보유종목
            tot_evlu = float(output2.get("tot_evlu_pfls_amt", 0))
            frcr_pchs_amt = float(output2.get("frcr_pchs_amt1", 0))
            available = float(output2.get("ovrs_ord_psbl_amt", 0))

            return BalanceInfo(
                total_evaluation=tot_evlu + frcr_pchs_amt,
                available_cash=available,
                currency="USD",
                holdings=[
                    {
                        "code": item.get("ovrs_pdno", ""),
                        "name": item.get("ovrs_item_name", ""),
                        "quantity": float(item.get("ovrs_cblc_qty", 0)),
                        "avg_price": float(item.get("pchs_avg_pric", 0)),
                        "current_price": float(item.get("now_pric2", 0)),
                        "pnl_rate": float(item.get("evlu_pfls_rt", 0)),
                        "exchange_code": item.get("ovrs_excg_cd", ""),
                    }
                    for item in data.get("output1", [])
                    if float(item.get("ovrs_cblc_qty", 0)) > 0
                ],
            )
        except httpx.HTTPError as e:
            logger.error(f"잔고 조회 HTTP 오류: {e}")
            return BalanceInfo()

    async def buy_order(
        self, code: str, qty: float, price: Optional[float] = None
    ) -> OrderResult:
        """해외주식 매수 주문"""
        return await self._place_order("buy", code, qty, price)

    async def sell_order(
        self, code: str, qty: float, price: Optional[float] = None
    ) -> OrderResult:
        """해외주식 매도 주문"""
        return await self._place_order("sell", code, qty, price)

    async def _place_order(
        self, side: str, code: str, qty: float, price: Optional[float] = None
    ) -> OrderResult:
        """해외주식 주문 실행 (매수/매도 공통)"""
        await self._ensure_token()
        await self._rate_limit()

        tr_id = self._tr_ids[side]
        account_parts = settings.kis_account_number.split("-")
        if len(account_parts) != 2:
            return OrderResult(success=False, message="잘못된 계좌번호 형식")

        url = f"{self._base_url}/uapi/overseas-stock/v1/trading/order"
        exchange_code = get_exchange_code(code)

        # 시장가 주문: OVRS_ORD_UNPR="0"
        if price is None or settings.order_type == "market":
            ord_unpr = "0"
        else:
            ord_unpr = str(round(price, 2))

        body = {
            "CANO": account_parts[0],
            "ACNT_PRDT_CD": account_parts[1],
            "OVRS_EXCG_CD": exchange_code,
            "PDNO": code.upper(),
            "ORD_QTY": str(round(qty, 4)),
            "OVRS_ORD_UNPR": ord_unpr,
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    url, headers=self._get_headers(tr_id), json=body
                )
                resp.raise_for_status()
                data = resp.json()

            if data.get("rt_cd") == "0":
                output = data.get("output", {})
                return OrderResult(
                    success=True,
                    order_id=output.get("ODNO", ""),
                    price=float(output.get("OVRS_ORD_UNPR", price or 0)),
                    quantity=qty,
                    message="주문 성공",
                )
            else:
                return OrderResult(
                    success=False,
                    message=data.get("msg1", "주문 실패"),
                )
        except httpx.HTTPError as e:
            return OrderResult(success=False, message=f"HTTP 오류: {str(e)}")

    async def get_current_price(self, code: str) -> Optional[float]:
        """해외주식 현재가 조회"""
        await self._ensure_token()
        await self._rate_limit()

        tr_id = self._tr_ids["current_price"]
        exchange_code = get_exchange_code(code)

        url = f"{self._base_url}/uapi/overseas-price/v1/quotations/price"
        params = {
            "AUTH": "",
            "EXCD": exchange_code,
            "SYMB": code.upper(),
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    url, headers=self._get_headers(tr_id), params=params
                )
                resp.raise_for_status()
                data = resp.json()

            if data.get("rt_cd") == "0":
                output = data.get("output", {})
                price = float(output.get("last", 0))
                if price > 0:
                    return price
        except Exception as e:
            logger.warning(f"현재가 조회 실패 ({code}): {e}")

        return None

    async def get_order_status(self, order_id: str) -> dict:
        """해외주식 체결 조회"""
        await self._ensure_token()
        await self._rate_limit()

        tr_id = self._tr_ids["order_status"]
        account_parts = settings.kis_account_number.split("-")
        if len(account_parts) != 2:
            return {"error": "잘못된 계좌번호 형식"}

        url = f"{self._base_url}/uapi/overseas-stock/v1/trading/inquire-ccnl"
        params = {
            "CANO": account_parts[0],
            "ACNT_PRDT_CD": account_parts[1],
            "PDNO": "",
            "ORD_STRT_DT": "",
            "ORD_END_DT": "",
            "SLL_BUY_DVSN": "00",
            "CCLD_NCCS_DVSN": "00",
            "OVRS_EXCG_CD": settings.default_exchange_code,
            "SORT_SQN": "DS",
            "ORD_GNO_BRNO": "",
            "ODNO": order_id,
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    url, headers=self._get_headers(tr_id), params=params
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as e:
            return {"error": f"HTTP 오류: {str(e)}"}


# 싱글턴
_kis_client: Optional[KISClient] = None


def get_kis_client() -> KISClient:
    global _kis_client
    if _kis_client is None:
        _kis_client = KISClient()
    return _kis_client
