import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# KIS API tr_id 매핑
TR_IDS = {
    "paper": {
        "buy": "VTTC0802U",
        "sell": "VTTC0801U",
        "balance": "VTTC8434R",
        "order_status": "VTTC8001R",
    },
    "live": {
        "buy": "TTTC0802U",
        "sell": "TTTC0801U",
        "balance": "TTTC8434R",
        "order_status": "TTTC8001R",
    },
}

BASE_URLS = {
    "paper": "https://openapivts.koreainvestment.com:29443",
    "live": "https://openapi.koreainvestment.com:9443",
}


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    price: float = 0.0
    quantity: int = 0
    message: str = ""


@dataclass
class BalanceInfo:
    total_evaluation: float = 0.0
    available_cash: float = 0.0
    holdings: list = None

    def __post_init__(self):
        if self.holdings is None:
            self.holdings = []


class KISClient:
    """한국투자증권 API 래퍼"""

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
        # 토큰 유효기간: 약 24시간, 안전하게 23시간으로 설정
        self._token_expires_at = now + 23 * 3600
        logger.info("KIS access token 발급 완료")
        return self._access_token

    async def _ensure_token(self):
        """토큰 확인 및 자동 갱신"""
        await self.get_access_token()

    async def get_balance(self) -> BalanceInfo:
        """잔고 조회"""
        await self._ensure_token()
        await self._rate_limit()

        tr_id = self._tr_ids["balance"]
        account_parts = settings.kis_account_number.split("-")
        if len(account_parts) != 2:
            logger.error(f"잘못된 계좌번호 형식: {settings.kis_account_number}")
            return BalanceInfo()

        url = f"{self._base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        params = {
            "CANO": account_parts[0],
            "ACNT_PRDT_CD": account_parts[1],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=self._get_headers(tr_id), params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("rt_cd") != "0":
            logger.error(f"잔고 조회 실패: {data.get('msg1', 'Unknown error')}")
            return BalanceInfo()

        output2 = data.get("output2", [{}])
        summary = output2[0] if output2 else {}

        return BalanceInfo(
            total_evaluation=float(summary.get("tot_evlu_amt", 0)),
            available_cash=float(summary.get("dnca_tot_amt", 0)),
            holdings=[
                {
                    "code": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "quantity": int(item.get("hldg_qty", 0)),
                    "avg_price": float(item.get("pchs_avg_pric", 0)),
                    "current_price": float(item.get("prpr", 0)),
                    "pnl_rate": float(item.get("evlu_pfls_rt", 0)),
                }
                for item in data.get("output1", [])
                if int(item.get("hldg_qty", 0)) > 0
            ],
        )

    async def buy_order(
        self, code: str, qty: int, price: Optional[float] = None
    ) -> OrderResult:
        """매수 주문"""
        return await self._place_order("buy", code, qty, price)

    async def sell_order(
        self, code: str, qty: int, price: Optional[float] = None
    ) -> OrderResult:
        """매도 주문"""
        return await self._place_order("sell", code, qty, price)

    async def _place_order(
        self, side: str, code: str, qty: int, price: Optional[float] = None
    ) -> OrderResult:
        """주문 실행 (매수/매도 공통)"""
        await self._ensure_token()
        await self._rate_limit()

        tr_id = self._tr_ids[side]
        account_parts = settings.kis_account_number.split("-")
        if len(account_parts) != 2:
            return OrderResult(success=False, message="잘못된 계좌번호 형식")

        url = f"{self._base_url}/uapi/domestic-stock/v1/trading/order-cash"

        # 시장가 주문: ORD_DVSN=01, 지정가: ORD_DVSN=00
        if price is None or settings.order_type == "market":
            ord_dvsn = "01"  # 시장가
            ord_price = "0"
        else:
            ord_dvsn = "00"  # 지정가
            ord_price = str(int(price))

        body = {
            "CANO": account_parts[0],
            "ACNT_PRDT_CD": account_parts[1],
            "PDNO": code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(qty),
            "ORD_UNPR": ord_price,
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
                    price=float(output.get("ORD_UNPR", price or 0)),
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

    async def get_order_status(self, order_id: str) -> dict:
        """체결 조회"""
        await self._ensure_token()
        await self._rate_limit()

        tr_id = self._tr_ids["order_status"]
        account_parts = settings.kis_account_number.split("-")
        if len(account_parts) != 2:
            return {"error": "잘못된 계좌번호 형식"}

        url = f"{self._base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        params = {
            "CANO": account_parts[0],
            "ACNT_PRDT_CD": account_parts[1],
            "INQR_STRT_DT": "",
            "INQR_END_DT": "",
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": order_id,
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=self._get_headers(tr_id), params=params)
            resp.raise_for_status()
            return resp.json()


# 싱글턴
_kis_client: Optional[KISClient] = None


def get_kis_client() -> KISClient:
    global _kis_client
    if _kis_client is None:
        _kis_client = KISClient()
    return _kis_client
