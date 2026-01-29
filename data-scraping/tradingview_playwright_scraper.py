"""
TradingView Chart Data Scraper using Playwright
================================================
baseline.ipynb의 찐찐 섹션 로직을 Playwright로 재구현한 스크래퍼

기능:
- TradingView 로그인 (쿠키 기반)
- 심볼 검색 및 선택
- 시간 단위 변경
- 차트 데이터 CSV 다운로드

주의사항:
- 첫 로그인 시 CAPTCHA 수동 해결 필요
- 로그인 후 쿠키를 저장하여 재사용 권장

환경변수 (.env):
- TRADINGVIEW_USERNAME: TradingView 사용자명
- TRADINGVIEW_PASSWORD: TradingView 비밀번호
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

# .env 파일 로드
load_dotenv()

# 설정
TIME_PERIODS = [
    {"name": "12달", "button_text": "1Y", "interval": "1 날"},
    {"name": "1달", "button_text": "1M", "interval": "30 분"},
    {"name": "1주", "button_text": "5D", "interval": "5 분"},
    {"name": "1일", "button_text": "1D", "interval": "1 분"},
]

# 종목 리스트 (baseline.ipynb에서 가져옴)
STOCK_LIST = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "WMT", "JPM", "MA", "PG", "CVX", "HD", "LLY", "ABBV",
    "MRK", "AVGO", "PEP", "KO", "COST", "ADBE", "TMO", "MCD", "CSCO", "CRM",
    # ... 더 많은 종목을 추가할 수 있음
]

DOWNLOAD_DIR = Path("./downloads")
COOKIES_FILE = Path("./cookies.json")


class TradingViewScraper:
    """TradingView 차트 데이터 스크래퍼"""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None

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
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )

        # 다운로드 디렉토리 생성
        DOWNLOAD_DIR.mkdir(exist_ok=True)

        # 브라우저 컨텍스트 생성
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            accept_downloads=True
        )

        # 쿠키 로드 (있는 경우)
        if COOKIES_FILE.exists():
            with open(COOKIES_FILE, 'r') as f:
                cookies = json.load(f)
                await self.context.add_cookies(cookies)
                print(f"쿠키 로드됨: {len(cookies)}개")

        self.page = await self.context.new_page()

    async def close(self):
        """브라우저 종료"""
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
            with open(COOKIES_FILE, 'w') as f:
                json.dump(cookies, f)
            print(f"쿠키 저장됨: {len(cookies)}개")

    async def login(self, username: str, password: str) -> bool:
        """
        TradingView 로그인

        참고: CAPTCHA가 나타나면 수동으로 해결해야 합니다.
        """
        print("로그인 페이지로 이동...")
        await self.page.goto('https://kr.tradingview.com/accounts/signin/')

        # domcontentloaded만 대기 (networkidle은 TradingView에서 타임아웃 발생)
        await self.page.wait_for_load_state('domcontentloaded')
        await asyncio.sleep(2)  # 추가 로딩 대기

        # 이메일 로그인 버튼 클릭
        try:
            await self.page.click('button:has-text("이메일")', timeout=5000)
            await asyncio.sleep(0.5)
        except:
            print("이미 로그인 폼이 표시됨")

        # 아이디/비밀번호 입력
        await self.page.fill('input[name="id_username"], input[placeholder*="유저네임"]', username)
        await self.page.fill('input[name="id_password"], input[type="password"]', password)

        # 로그인 버튼 클릭
        await self.page.click('button:has-text("로그인")')

        # CAPTCHA 대기 (수동 해결 필요)
        print("CAPTCHA가 나타나면 수동으로 해결해주세요...")
        print("로그인 완료 대기 중... (최대 120초)")

        try:
            # 로그인 성공 시 차트 페이지로 이동하거나 홈으로 리다이렉트됨
            await self.page.wait_for_url(
                lambda url: 'chart' in url or ('tradingview.com' in url and 'signin' not in url),
                timeout=120000
            )
            print("로그인 성공!")
            await self.save_cookies()
            return True
        except Exception as e:
            print(f"로그인 실패 또는 타임아웃: {e}")
            return False

    async def navigate_to_chart(self):
        """차트 페이지로 이동"""
        print("차트 페이지로 이동 중...")
        await self.page.goto('https://kr.tradingview.com/chart/')

        # domcontentloaded만 대기 (networkidle은 차트 페이지에서 타임아웃 발생 가능)
        await self.page.wait_for_load_state('domcontentloaded')

        # 차트가 로드될 때까지 대기 (차트 영역이 나타날 때까지)
        try:
            await self.page.wait_for_selector('div[class*="chart-container"], canvas', timeout=30000)
            print("차트 로드 완료")
        except:
            print("차트 로딩 대기 중...")

        await asyncio.sleep(3)  # 추가 로딩 대기

    async def search_and_select_symbol(self, symbol: str) -> bool:
        """
        심볼 검색 및 선택

        Args:
            symbol: 검색할 심볼 (예: "NVDA", "AAPL")

        Returns:
            성공 여부
        """
        print(f"심볼 검색: {symbol}")

        try:
            # 상단 툴바의 심볼 검색 버튼 클릭 (고정 ID 사용)
            symbol_btn = self.page.locator('#header-toolbar-symbol-search')
            await symbol_btn.click(timeout=5000)

            await asyncio.sleep(1)

            # 검색창에 심볼 입력 - "심볼, ISIN 또는 CUSIP" placeholder 사용
            search_input = self.page.get_by_role("searchbox").or_(
                self.page.get_by_placeholder("심볼, ISIN 또는 CUSIP")
            ).or_(
                self.page.locator('input[data-role="search"]')
            ).first

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
                nasdaq_result = self.page.get_by_text(f"{symbol}").locator("xpath=ancestor::*[contains(., 'NASDAQ') and contains(., 'stock')]").first
                await nasdaq_result.click(timeout=3000)
                clicked = True
            except:
                pass

            # 방법 2: NYSE 거래소 시도
            if not clicked:
                try:
                    nyse_result = self.page.get_by_text(f"{symbol}").locator("xpath=ancestor::*[contains(., 'NYSE') and contains(., 'stock')]").first
                    await nyse_result.click(timeout=3000)
                    clicked = True
                except:
                    pass

            # 방법 3: 첫 번째 검색 결과 (심볼명과 거래소 텍스트를 포함하는 요소)
            if not clicked:
                try:
                    # dialog 내의 첫 번째 검색 결과 항목
                    first_result = self.page.locator(f'div:has(> div:has-text("{symbol}")):has-text("stock")').first
                    await first_result.click(timeout=3000)
                    clicked = True
                except:
                    pass

            # 방법 4: Enter 키로 첫 번째 결과 선택
            if not clicked:
                await self.page.keyboard.press('Enter')
                await asyncio.sleep(0.5)

            await asyncio.sleep(2)  # 차트 로딩 대기
            print(f"심볼 선택 완료: {symbol}")
            return True

        except Exception as e:
            print(f"심볼 검색 실패: {e}")
            # ESC로 다이얼로그 닫기
            await self.page.keyboard.press('Escape')
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
        print(f"시간 단위 변경: {button_text}")

        try:
            # 하단 툴바의 기간 버튼 클릭
            # 버튼 이름이 "1 날 인터벌 의 1 년₩" 형식으로 되어있음
            period_button = self.page.locator(f'button:has-text("{button_text}")').first
            await period_button.click(timeout=5000)

            await asyncio.sleep(2)  # 차트 갱신 대기
            print(f"시간 단위 변경 완료: {button_text}")
            return True

        except Exception as e:
            print(f"시간 단위 변경 실패: {e}")
            # 대체 방법: 하단 툴바에서 텍스트로 찾기
            try:
                alt_button = self.page.get_by_text(button_text, exact=True).first
                await alt_button.click(timeout=5000)
                await asyncio.sleep(2)
                print(f"시간 단위 변경 완료 (대체방법): {button_text}")
                return True
            except:
                print(f"시간 단위 변경 최종 실패: {e}")
                return False

    async def export_chart_data(self, output_filename: Optional[str] = None) -> Optional[Path]:
        """
        차트 데이터를 CSV로 다운로드

        Args:
            output_filename: 출력 파일명 (없으면 자동 생성)

        Returns:
            다운로드된 파일 경로 또는 None
        """
        print("차트 데이터 다운로드 시작...")

        try:
            # 레이아웃 메뉴의 화살표 버튼 클릭 (JavaScript 사용 - 클래스명이 동적으로 변경됨)
            arrow_clicked = await self.page.evaluate('''
                () => {
                    // 화살표 아이콘이 있는 div 요소 찾기
                    const arrows = document.querySelectorAll('div[class*="arrow"]');
                    if (arrows.length > 0) {
                        arrows[0].click();
                        return true;
                    }
                    return false;
                }
            ''')

            if not arrow_clicked:
                print("레이아웃 메뉴를 찾을 수 없습니다")
                return None

            await asyncio.sleep(0.5)

            # "차트 데이터 다운로드" 옵션 클릭 (role 기반 선택자)
            try:
                download_option = self.page.get_by_role("row", name="차트 데이터 다운로드")
                await download_option.click(timeout=5000)
            except:
                # 대체 선택자
                download_option = self.page.locator('text=차트 데이터 다운로드')
                await download_option.click(timeout=5000)

            await asyncio.sleep(0.5)

            # 다운로드 버튼 클릭하고 파일 대기
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
            print(f"다운로드 완료: {save_path}")
            return save_path

        except Exception as e:
            print(f"다운로드 실패: {e}")
            # ESC로 다이얼로그 닫기
            await self.page.keyboard.press('Escape')
            return None

    async def process_single_stock(self, symbol: str) -> dict:
        """
        단일 종목에 대해 모든 시간대 데이터 수집

        Args:
            symbol: 종목 심볼

        Returns:
            결과 딕셔너리 {period_name: file_path}
        """
        results = {}

        # 심볼 선택
        if not await self.search_and_select_symbol(symbol):
            return results

        # 각 시간대별로 데이터 수집
        for period in TIME_PERIODS:
            period_name = period["name"]
            button_text = period["button_text"]

            print(f"\n[{symbol}] {period_name} 데이터 수집 중...")

            # 시간 단위 변경
            if await self.change_time_period(button_text):
                await asyncio.sleep(1)

                # 데이터 다운로드
                filename = f"{symbol}_{period_name}_{datetime.now().strftime('%Y%m%d')}.csv"
                file_path = await self.export_chart_data(filename)

                if file_path:
                    results[period_name] = str(file_path)

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
            print(f"\n{'='*50}")
            print(f"[{i+1}/{len(stock_list)}] {symbol} 처리 중...")
            print('='*50)

            results = await self.process_single_stock(symbol)
            all_results[symbol] = results

            # Rate limiting
            await asyncio.sleep(2)

        return all_results


async def main():
    """메인 실행 함수"""
    # 환경변수에서 로그인 정보 가져오기
    username = os.getenv('TRADINGVIEW_USERNAME')
    password = os.getenv('TRADINGVIEW_PASSWORD')

    if not username or not password:
        print("환경변수가 설정되지 않았습니다.")
        print(".env 파일에 다음 내용을 추가하세요:")
        print("  TRADINGVIEW_USERNAME=your_username")
        print("  TRADINGVIEW_PASSWORD=your_password")
        return

    # 테스트용: 단일 종목만 처리
    test_symbols = ["NVDA", "AAPL"]

    async with TradingViewScraper(headless=False) as scraper:
        # 쿠키가 없으면 로그인
        if not COOKIES_FILE.exists():
            print("로그인이 필요합니다.")
            if not await scraper.login(username, password):
                print("로그인 실패. 프로그램을 종료합니다.")
                return

        # 차트 페이지로 이동
        await scraper.navigate_to_chart()

        # 로그인 상태 확인을 위해 잠시 대기
        await asyncio.sleep(2)

        # 테스트 종목 처리
        for symbol in test_symbols:
            print(f"\n{'='*50}")
            print(f"{symbol} 데이터 수집 시작")
            print('='*50)

            results = await scraper.process_single_stock(symbol)

            print(f"\n{symbol} 결과:")
            for period, path in results.items():
                print(f"  - {period}: {path}")

        print("\n모든 작업 완료!")


if __name__ == "__main__":
    asyncio.run(main())
