"""
TradingView Chart Data Scraper using Playwright with DB Upload
==============================================================
baseline.ipynb의 찐찐 섹션 로직을 Playwright로 재구현하고
CSV 다운로드 후 자동으로 DB에 업로드하는 기능을 추가한 스크래퍼

기능:
- TradingView 로그인 (쿠키 기반)
- 심볼 검색 및 선택
- 시간 단위 변경
- 차트 데이터 CSV 다운로드
- **NEW: 원격 MySQL 데이터베이스로 자동 업로드**

주의사항:
- 첫 로그인 시 CAPTCHA 수동 해결 필요
- 로그인 후 쿠키를 저장하여 재사용 권장

환경변수 (.env):
- TRADINGVIEW_USERNAME: TradingView 사용자명
- TRADINGVIEW_PASSWORD: TradingView 비밀번호
- UPLOAD_TO_DB: DB 업로드 활성화 여부 (기본값: true)
- USE_EXISTING_TUNNEL: 기존 SSH 터널 사용 여부 (기본값: true)
- HEADLESS: Headless 모드 실행 여부 (기본값: false, 쿠키 있으면 자동 true)
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

try:
    from db_service import DatabaseService

    DB_SERVICE_AVAILABLE = True
except ImportError:
    DB_SERVICE_AVAILABLE = False
    print("db_service not available. Install sqlalchemy, pymysql, sshtunnel, pandas.")

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tradingview_scraper_upload.log"),
    ],
)
logger = logging.getLogger(__name__)

# 설정
TIME_PERIODS = [
    {"name": "12달", "button_text": "1Y", "interval": "1 날"},
    {"name": "1달", "button_text": "1M", "interval": "30 분"},
    {"name": "1주", "button_text": "5D", "interval": "5 분"},
    {"name": "1일", "button_text": "1D", "interval": "1 분"},
]

# 종목 리스트 (etf2_db 테이블 기준 - 101개 종목)
# S&P 500 상위 시가총액 종목 + 주요 기술/금융/헬스케어 종목
STOCK_LIST = [
    # Technology (30개)
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "ADBE",
    "CRM", "CSCO", "ORCL", "AMD", "INTC", "QCOM", "TXN", "NOW",
    "INTU", "AMAT", "ADI", "LRCX", "KLAC", "MU", "PANW", "CRWD",
    "ANET", "PLTR", "APP", "IBM", "HOOD", "IBKR",
    # Communication Services (4개)
    "AMZN", "TSLA", "NFLX", "T",
    # Consumer (9개)
    "WMT", "HD", "COST", "MCD", "LOW", "TJX", "BKNG", "PEP", "KO",
    # Financials (15개)
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK",
    "SCHW", "AXP", "C", "SPGI", "COF", "BX",
    # Healthcare (14개)
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "DHR", "AMGN", "ISRG", "GILD", "BSX", "SYK",
    # Industrials (12개)
    "CAT", "GE", "HON", "UNP", "BA", "RTX", "LMT", "DE",
    "ETN", "PLD", "MDT", "MMM",
    # Energy (3개)
    "XOM", "CVX", "COP",
    # Consumer Staples (3개)
    "PG", "PM", "LIN",
    # Utilities & Others (11개)
    "NEE", "CEG", "DIS", "VZ", "TMUS", "UBER", "GEV", "PGR",
    "WELL", "APH", "ACN",
]

DOWNLOAD_DIR = Path("./downloads")
COOKIES_FILE = Path("./cookies.json")


class TradingViewScraper:
    """TradingView 차트 데이터 스크래퍼"""

    def __init__(
        self,
        headless: bool = False,
        upload_to_db: bool = True,
        use_existing_tunnel: bool = True,
    ):
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None

        self.upload_to_db = upload_to_db and DB_SERVICE_AVAILABLE
        self.use_existing_tunnel = use_existing_tunnel
        self.db_service: Optional[DatabaseService] = None

        if self.upload_to_db:
            logger.info("Database upload enabled")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self):
        """브라우저 시작"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        # 다운로드 디렉토리 생성
        DOWNLOAD_DIR.mkdir(exist_ok=True)

        # 브라우저 컨텍스트 생성
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            accept_downloads=True,
        )

        # 쿠키 로드 (있는 경우)
        if COOKIES_FILE.exists():
            with open(COOKIES_FILE, "r") as f:
                cookies = json.load(f)
                await self.context.add_cookies(cookies)
                logger.info(f"쿠키 로드됨: {len(cookies)}개")

        self.page = await self.context.new_page()

        if self.upload_to_db:
            try:
                self.db_service = DatabaseService(
                    use_existing_tunnel=self.use_existing_tunnel
                )
                self.db_service.connect()
                logger.info("Database service connected")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_service = None
                self.upload_to_db = False

    async def close(self):
        """브라우저 종료"""
        if self.db_service:
            self.db_service.close()
            logger.info("Database service closed")

        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def save_cookies(self):
        """현재 쿠키를 파일에 저장"""
        if self.context:
            cookies = await self.context.cookies()
            with open(COOKIES_FILE, "w") as f:
                json.dump(cookies, f)
            logger.info(f"쿠키 저장됨: {len(cookies)}개")

    def check_cookie_expiry(self) -> bool:
        """
        쿠키 만료 여부 확인

        Returns:
            True if cookies are valid, False if expired or missing
        """
        if not COOKIES_FILE.exists():
            return False

        try:
            with open(COOKIES_FILE, "r") as f:
                cookies = json.load(f)

            # 현재 시간 (Unix timestamp)
            current_time = datetime.now().timestamp()

            # 모든 쿠키 중 하나라도 만료되었는지 확인
            for cookie in cookies:
                # 'expires' 필드가 있는 경우만 체크
                if "expires" in cookie:
                    # expires는 Unix timestamp (초 단위)
                    if cookie["expires"] < current_time:
                        logger.warning(
                            f"쿠키 만료됨: {cookie.get('name', 'unknown')}"
                        )
                        return False

            logger.info("쿠키가 유효합니다.")
            return True

        except Exception as e:
            logger.error(f"쿠키 확인 중 오류 발생: {e}")
            return False

    async def login(self, username: str, password: str) -> bool:
        """
        TradingView 로그인

        참고: CAPTCHA가 나타나면 수동으로 해결해야 합니다.
        """
        logger.info("로그인 페이지로 이동...")
        await self.page.goto("https://kr.tradingview.com/accounts/signin/")

        # domcontentloaded만 대기 (networkidle은 TradingView에서 타임아웃 발생)
        await self.page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(2)  # 추가 로딩 대기

        # 이메일 로그인 버튼 클릭
        try:
            await self.page.click('button:has-text("이메일")', timeout=5000)
            await asyncio.sleep(0.5)
        except:
            logger.info("이미 로그인 폼이 표시됨")

        # 아이디/비밀번호 입력
        await self.page.fill(
            'input[name="id_username"], input[placeholder*="유저네임"]', username
        )
        await self.page.fill(
            'input[name="id_password"], input[type="password"]', password
        )

        # 로그인 버튼 클릭
        await self.page.click('button:has-text("로그인")')

        # CAPTCHA 대기 (수동 해결 필요)
        logger.info("CAPTCHA가 나타나면 수동으로 해결해주세요...")
        logger.info("로그인 완료 대기 중... (최대 120초)")

        try:
            # 로그인 성공 시 차트 페이지로 이동하거나 홈으로 리다이렉트됨
            await self.page.wait_for_url(
                lambda url: "chart" in url
                or ("tradingview.com" in url and "signin" not in url),
                timeout=120000,
            )
            logger.info("로그인 성공!")
            await self.save_cookies()
            return True
        except Exception as e:
            logger.error(f"로그인 실패 또는 타임아웃: {e}")
            return False

    async def navigate_to_chart(self):
        """차트 페이지로 이동"""
        logger.info("차트 페이지로 이동 중...")
        await self.page.goto("https://kr.tradingview.com/chart/")

        # domcontentloaded만 대기 (networkidle은 차트 페이지에서 타임아웃 발생 가능)
        await self.page.wait_for_load_state("domcontentloaded")

        # 차트가 로드될 때까지 대기 (차트 영역이 나타날 때까지)
        try:
            await self.page.wait_for_selector(
                'div[class*="chart-container"], canvas', timeout=30000
            )
            logger.info("차트 로드 완료")
        except:
            logger.info("차트 로딩 대기 중...")

        await asyncio.sleep(3)  # 추가 로딩 대기

    async def search_and_select_symbol(self, symbol: str) -> bool:
        """
        심볼 검색 및 선택

        Args:
            symbol: 검색할 심볼 (예: "NVDA", "AAPL")

        Returns:
            성공 여부
        """
        logger.info(f"심볼 검색: {symbol}")

        try:
            # 상단 툴바의 심볼 검색 버튼 클릭 (고정 ID 사용)
            symbol_btn = self.page.locator("#header-toolbar-symbol-search")
            await symbol_btn.click(timeout=5000)

            await asyncio.sleep(1)

            # 검색창에 심볼 입력 - "심볼, ISIN 또는 CUSIP" placeholder 사용
            search_input = (
                self.page.get_by_role("searchbox")
                .or_(self.page.get_by_placeholder("심볼, ISIN 또는 CUSIP"))
                .or_(self.page.locator('input[data-role="search"]'))
                .first
            )

            # 기존 텍스트 지우고 새로 입력
            await search_input.clear()
            await search_input.fill(symbol, timeout=10000)
            await asyncio.sleep(1.5)  # 검색 결과 대기

            # 검색 결과에서 해당 심볼 클릭
            # TradingView 검색 결과는 텍스트 기반으로 검색
            clicked = False

            # 방법 1: NASDAQ 거래소 주식 결과 클릭 (가장 일반적인 미국 주식)
            try:
                # "심볼 찾기" 다이얼로그 내에서 NASDAQ + stock 조합 찾기
                nasdaq_result = (
                    self.page.get_by_text(f"{symbol}")
                    .locator(
                        "xpath=ancestor::*[contains(., 'NASDAQ') and contains(., 'stock')]"
                    )
                    .first
                )
                await nasdaq_result.click(timeout=3000)
                clicked = True
            except:
                pass

            # 방법 2: NYSE 거래소 시도
            if not clicked:
                try:
                    nyse_result = (
                        self.page.get_by_text(f"{symbol}")
                        .locator(
                            "xpath=ancestor::*[contains(., 'NYSE') and contains(., 'stock')]"
                        )
                        .first
                    )
                    await nyse_result.click(timeout=3000)
                    clicked = True
                except:
                    pass

            # 방법 3: 첫 번째 검색 결과 (심볼명과 거래소 텍스트를 포함하는 요소)
            if not clicked:
                try:
                    # dialog 내의 첫 번째 검색 결과 항목
                    first_result = self.page.locator(
                        f'div:has(> div:has-text("{symbol}")):has-text("stock")'
                    ).first
                    await first_result.click(timeout=3000)
                    clicked = True
                except:
                    pass

            # 방법 4: Enter 키로 첫 번째 결과 선택
            if not clicked:
                await self.page.keyboard.press("Enter")
                await asyncio.sleep(0.5)

            await asyncio.sleep(2)  # 차트 로딩 대기
            logger.info(f"심볼 선택 완료: {symbol}")
            return True

        except Exception as e:
            logger.error(f"심볼 검색 실패: {e}")
            # ESC로 다이얼로그 닫기
            await self.page.keyboard.press("Escape")
            await asyncio.sleep(0.5)
            return False

    async def change_time_period(self, button_text: str) -> bool:
        """
        시간 단위 변경

        Args:
            button_text: 버튼 텍스트 (예: "1Y", "1M", "5D", "1D")

        Returns:
            성공 여부
        """
        logger.info(f"시간 단위 변경: {button_text}")

        try:
            # 하단 툴바의 기간 버튼 클릭
            # 버튼 이름이 "1 날 인터벌 의 1 년₩" 형식으로 되어있음
            period_button = self.page.locator(f'button:has-text("{button_text}")').first
            await period_button.click(timeout=5000)

            await asyncio.sleep(2)  # 차트 갱신 대기
            logger.info(f"시간 단위 변경 완료: {button_text}")
            return True

        except Exception as e:
            logger.error(f"시간 단위 변경 실패: {e}")
            # 대체 방법: 하단 툴바에서 텍스트로 찾기
            try:
                alt_button = self.page.get_by_text(button_text, exact=True).first
                await alt_button.click(timeout=5000)
                await asyncio.sleep(2)
                logger.info(f"시간 단위 변경 완료 (대체방법): {button_text}")
                return True
            except:
                logger.error(f"시간 단위 변경 최종 실패: {e}")
                return False

    def _get_timeframe_code(self, period_name: str) -> str:
        """
        Convert Korean period name to database timeframe code.

        Args:
            period_name: Korean period name (e.g., "12개월", "1일")

        Returns:
            Timeframe code for database (e.g., "D", "1h")
        """
        timeframe_map = {
            "12달": "D",  # Daily for 12-month data
            "12개월": "D",  # Daily for 12-month data
            "1달": "D",  # Daily for 1-month data
            "1개월": "D",  # Daily for 1-month data
            "1주": "D",  # Daily for 1-week data
            "1일": "1h",  # Hourly for 1-day data
        }
        return timeframe_map.get(period_name, "D")

    async def export_chart_data(
        self, output_filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        차트 데이터를 CSV로 다운로드

        Args:
            output_filename: 출력 파일명 (없으면 자동 생성)

        Returns:
            다운로드된 파일 경로 또는 None
        """
        logger.info("차트 데이터 다운로드 시작...")

        try:
            arrow_clicked = await self.page.evaluate("""
                () => {
                    const arrows = document.querySelectorAll('div[class*="arrow"]');
                    if (arrows.length > 0) {
                        arrows[0].click();
                        return true;
                    }
                    return false;
                }
            """)

            if not arrow_clicked:
                logger.warning("레이아웃 메뉴를 찾을 수 없습니다")
                return None

            await asyncio.sleep(0.5)

            try:
                download_option = self.page.get_by_role(
                    "row", name="차트 데이터 다운로드"
                )
                await download_option.click(timeout=5000)
            except:
                download_option = self.page.locator("text=차트 데이터 다운로드")
                await download_option.click(timeout=5000)

            await asyncio.sleep(0.5)

            async with self.page.expect_download(timeout=30000) as download_info:
                download_btn = self.page.get_by_role("button", name="다운로드")
                await download_btn.click()

            download = await download_info.value

            # 파일 저장
            if output_filename:
                save_path = DOWNLOAD_DIR / output_filename
            else:
                save_path = DOWNLOAD_DIR / download.suggested_filename

            await download.save_as(save_path)
            logger.info(f"다운로드 완료: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"다운로드 실패: {e}")
            # ESC로 다이얼로그 닫기
            await self.page.keyboard.press("Escape")
            return None

    async def process_single_stock(self, symbol: str, max_retries: int = 3) -> dict:
        """
        단일 종목에 대해 모든 시간대 데이터 수집 (재시도 로직 포함)

        Args:
            symbol: 종목 심볼
            max_retries: 최대 재시도 횟수

        Returns:
            결과 딕셔너리 {period_name: file_path}
        """
        results = {}

        # 심볼 선택 (재시도)
        for attempt in range(max_retries):
            if await self.search_and_select_symbol(symbol):
                break
            else:
                if attempt < max_retries - 1:
                    logger.warning(f"심볼 선택 실패 (시도 {attempt + 1}/{max_retries}). 재시도 중...")
                    await asyncio.sleep(2)
                else:
                    logger.error(f"심볼 선택 최종 실패: {symbol}")
                    return results

        # 각 시간대별로 데이터 수집
        for period in TIME_PERIODS:
            period_name = period["name"]
            button_text = period["button_text"]

            logger.info(f"\n[{symbol}] {period_name} 데이터 수집 중...")

            # 시간 단위 변경 (재시도)
            time_change_success = False
            for attempt in range(max_retries):
                if await self.change_time_period(button_text):
                    time_change_success = True
                    break
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"시간 단위 변경 실패 (시도 {attempt + 1}/{max_retries}). 재시도 중...")
                        await asyncio.sleep(2)

            if not time_change_success:
                logger.error(f"시간 단위 변경 최종 실패: {button_text}")
                continue

            await asyncio.sleep(1)

            # 데이터 다운로드 (재시도)
            timeframe_code = self._get_timeframe_code(period_name)
            filename = (
                f"{symbol}_{period_name}_{datetime.now().strftime('%Y%m%d')}.csv"
            )

            file_path = None
            for attempt in range(max_retries):
                file_path = await self.export_chart_data(filename)
                if file_path:
                    break
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"다운로드 실패 (시도 {attempt + 1}/{max_retries}). 재시도 중...")
                        await asyncio.sleep(2)

            if file_path:
                results[period_name] = str(file_path)

                if self.upload_to_db and self.db_service:
                    try:
                        rows = self.db_service.upload_csv(
                            file_path, symbol, timeframe_code
                        )
                        logger.info(
                            f"  [{symbol} - {period_name}] ✓ Uploaded {rows} rows to DB"
                        )
                    except Exception as e:
                        logger.error(
                            f"  [{symbol} - {period_name}] ✗ DB upload failed: {e}"
                        )
            else:
                logger.error(f"다운로드 최종 실패: {symbol} - {period_name}")

        return results

    async def process_all_stocks(self, stock_list: list = None) -> dict:
        """
        모든 종목에 대해 데이터 수집

        Args:
            stock_list: 종목 리스트 (없으면 기본 STOCK_LIST 사용)

        Returns:
            결과 딕셔너리 {symbol: {period_name: file_path}}
        """
        if stock_list is None:
            stock_list = STOCK_LIST

        all_results = {}

        # 차트 페이지로 이동
        await self.navigate_to_chart()

        for i, symbol in enumerate(stock_list):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"[{i + 1}/{len(stock_list)}] {symbol} 처리 중...")
            logger.info("=" * 50)

            results = await self.process_single_stock(symbol)
            all_results[symbol] = results

            # Rate limiting
            await asyncio.sleep(2)

        return all_results


async def main():
    """메인 실행 함수"""
    # 환경변수에서 로그인 정보 가져오기
    username = os.getenv("TRADINGVIEW_USERNAME")
    password = os.getenv("TRADINGVIEW_PASSWORD")

    upload_to_db = os.getenv("UPLOAD_TO_DB", "true").lower() == "true"
    use_existing_tunnel = os.getenv("USE_EXISTING_TUNNEL", "true").lower() == "true"

    # HEADLESS 환경변수 읽기 (기본값: false)
    headless = os.getenv("HEADLESS", "false").lower() == "true"

    # 쿠키 존재 및 만료 여부 확인
    cookie_valid = False
    if COOKIES_FILE.exists():
        # 임시 스크래퍼 인스턴스로 쿠키 만료 체크
        temp_scraper = TradingViewScraper(
            headless=False, upload_to_db=False, use_existing_tunnel=False
        )
        cookie_valid = temp_scraper.check_cookie_expiry()

    # 쿠키가 없거나 만료된 경우 headless 모드 비활성화
    if headless and not cookie_valid:
        if not COOKIES_FILE.exists():
            logger.warning("쿠키 파일이 없습니다. headless 모드를 비활성화합니다.")
        else:
            logger.warning("쿠키가 만료되었습니다. headless 모드를 비활성화합니다.")
        logger.warning("로그인 시 CAPTCHA 수동 해결이 필요하므로 브라우저 창을 띄웁니다.")
        headless = False

    # 쿠키가 유효하면 자동으로 headless 모드 활성화 (환경변수가 명시적으로 false가 아닌 경우)
    if cookie_valid and os.getenv("HEADLESS") is None:
        logger.info("쿠키 파일이 유효합니다. headless 모드를 자동으로 활성화합니다.")
        headless = True

    if not username or not password:
        logger.warning("환경변수가 설정되지 않았습니다.")
        logger.warning(".env 파일에 다음 내용을 추가하세요:")
        logger.warning("  TRADINGVIEW_USERNAME=your_username")
        logger.warning("  TRADINGVIEW_PASSWORD=your_password")
        return

    # 테스트용: 단일 종목만 처리
    # test_symbols = ["NVDA", "AAPL"]
    # 전체 리스트 사용
    test_symbols = STOCK_LIST

    logger.info(f"Headless 모드: {headless}")
    logger.info(f"DB 업로드: {upload_to_db}")
    logger.info(f"기존 SSH 터널 사용: {use_existing_tunnel}")

    async with TradingViewScraper(
        headless=headless,
        upload_to_db=upload_to_db,
        use_existing_tunnel=use_existing_tunnel,
    ) as scraper:
        # 쿠키가 없으면 로그인
        if not COOKIES_FILE.exists():
            logger.info("로그인이 필요합니다.")
            if not await scraper.login(username, password):
                logger.error("로그인 실패. 프로그램을 종료합니다.")
                return

        # 차트 페이지로 이동
        await scraper.navigate_to_chart()

        # 로그인 상태 확인을 위해 잠시 대기
        await asyncio.sleep(2)

        # 종목 처리
        for symbol in test_symbols:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"{symbol} 데이터 수집 시작")
            logger.info("=" * 50)

            results = await scraper.process_single_stock(symbol)

            logger.info(f"\n{symbol} 결과:")
            for period, path in results.items():
                logger.info(f"  - {period}: {path}")

        logger.info("\n모든 작업 완료!")


if __name__ == "__main__":
    asyncio.run(main())
