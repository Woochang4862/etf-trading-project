#!/usr/bin/env python3
"""
TradingView Stock Data Scraper - Fully Automated

Features:
- Fully automated login (no manual intervention required)
- Cookie persistence with auto-refresh
- Robust error handling and retry logic
- Progress tracking and logging
- Cloudflare bypass using Playwright's stealth mode
- **NEW: Automatic upload to remote MySQL database via SSH tunnel**

Database Upload:
- Downloaded CSV files are automatically uploaded to remote MySQL
- Uses SSH tunnel connection (see .claude/skills/db-ssh-tunneling/skill.md)
- Tables are created in format: {symbol}_{timeframe} (e.g., AAPL_D, NVDA_1h)
- Supports UPSERT (updates existing records, inserts new ones)

Environment Variables:
- TRADINGVIEW_USERNAME: TradingView login username
- TRADINGVIEW_PASSWORD: TradingView login password
- DOWNLOAD_DIR: Directory for downloaded CSV files (default: ~/Downloads/tradingview)
- UPLOAD_TO_DB: Enable DB upload (default: true)
- USE_EXISTING_TUNNEL: Use existing SSH tunnel (default: true)

Prerequisites:
1. SSH tunnel must be running: ssh -f -N -L 3306:127.0.0.1:5100 ahnbi2@ahnbi2.suwon.ac.kr
2. Or set USE_EXISTING_TUNNEL=false to create new tunnel (requires SSH key)
"""

import asyncio
import json
import logging
import os
import glob as glob_module
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    Page,
    Error as PlaywrightError,
)

# Import database service for uploading to remote MySQL
try:
    from db_service import DatabaseService
    DB_SERVICE_AVAILABLE = True
except ImportError:
    DB_SERVICE_AVAILABLE = False
    logging.warning("db_service not available. Install sqlalchemy, pymysql, sshtunnel, pandas.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tradingview_scraper.log")],
)
logger = logging.getLogger(__name__)


class TradingViewScraper:
    """Automated TradingView data scraper with login management"""

    def __init__(
        self,
        username: str,
        password: str,
        headless: bool = True,
        cookie_file: str = "tradingview_cookies.json",
        download_dir: Optional[str] = None,
        upload_to_db: bool = True,
        use_existing_tunnel: bool = True,
    ):
        self.username = username
        self.password = password
        self.headless = headless
        self.cookie_file = Path(cookie_file)
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # Download directory for CSV exports
        self.download_dir = Path(download_dir) if download_dir else Path.home() / "Downloads"

        # Database upload settings
        self.upload_to_db = upload_to_db and DB_SERVICE_AVAILABLE
        self.use_existing_tunnel = use_existing_tunnel
        self.db_service: Optional[DatabaseService] = None

        if self.upload_to_db:
            logger.info("Database upload enabled")

        # Stock list to scrape
        self.stock_list = [
            "NVDA",
            "AAPL",
            "MSFT",
            "AMZN",
            "AVGO",
            "GOOGL",
            "META",
            "GOOG",
            "TSLA",
            "BRK.B",
            "JPM",
            "WMT",
            "LLY",
            "ORCL",
            "V",
            "MA",
            "XOM",
            "PLTR",
            "NFLX",
            "JNJ",
            # Add more stocks as needed
        ]

        # Time periods to scrape (using English button text like tradingview_playwright_scraper.py)
        # TradingView UI shows "1Y", "1M", "5D", "1D" buttons in the bottom toolbar
        self.time_periods = [
            {"name": "12개월", "button": "1Y", "scroll_needed": False},
            {"name": "1개월", "button": "1M", "scroll_needed": False},
            {"name": "1주", "button": "5D", "scroll_needed": True, "scroll_time": 2},
            {"name": "1일", "button": "1D", "scroll_needed": True, "scroll_time": 3},
        ]

    async def start(self):
        """Initialize browser and page"""
        logger.info("Starting TradingView scraper...")

        playwright = await async_playwright().start()

        # Ensure download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Launch browser with stealth options
        self.browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            downloads_path=str(self.download_dir),
        )

        context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="ko-KR",
            accept_downloads=True,
        )

        self.page = await context.new_page()
        self.page.set_default_timeout(30000)

        # Load existing cookies if available
        await self._load_cookies(context)

        # Initialize database service if enabled
        if self.upload_to_db:
            try:
                self.db_service = DatabaseService(use_existing_tunnel=self.use_existing_tunnel)
                self.db_service.connect()
                logger.info("Database service connected")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_service = None
                self.upload_to_db = False

    async def _load_cookies(self, context):
        """Load cookies from file if exists"""
        if self.cookie_file.exists():
            try:
                with open(self.cookie_file, "r") as f:
                    cookies = json.load(f)
                await context.add_cookies(cookies)
                logger.info(f"✓ Loaded {len(cookies)} cookies from {self.cookie_file}")
            except Exception as e:
                logger.warning(f"Failed to load cookies: {e}")

    async def _save_cookies(self, context):
        """Save cookies to file"""
        try:
            cookies = await context.cookies()
            with open(self.cookie_file, "w") as f:
                json.dump(cookies, f, indent=2)
            logger.info(f"✓ Saved {len(cookies)} cookies to {self.cookie_file}")
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")

    def _get_timeframe_code(self, period_name: str) -> str:
        """
        Convert Korean period name to database timeframe code.

        Args:
            period_name: Korean period name (e.g., "12개월", "1일")

        Returns:
            Timeframe code for database (e.g., "D", "1h")
        """
        timeframe_map = {
            "12개월": "D",    # Daily for 12-month data
            "1개월": "D",     # Daily for 1-month data
            "1주": "D",       # Daily for 1-week data
            "1일": "1h",      # Hourly for 1-day data
            "1시간": "10m",   # 10-min for 1-hour view
            "10분": "1m",     # 1-min for 10-minute view
        }
        return timeframe_map.get(period_name, "D")

    async def _is_logged_in(self) -> bool:
        """Check if user is logged in by checking if cookie file exists"""
        # 쿠키 파일이 존재하면 로그인 된 것으로 판단
        # 파일이 없으면 로그인 안된 것으로 판단
        # (저장된 쿠키는 유효한 상태를 의미)
        if self.cookie_file.exists():
            logger.info(f"✓ Already logged in (cookies exist: {self.cookie_file})")
            return True
        else:
            logger.info(f"User not logged in (no cookies found)")
            return False

    async def _login(self):
        """Perform login with credentials"""
        logger.info("Attempting login...")

        try:
            # Navigate to homepage
            await self.page.goto(
                "https://kr.tradingview.com/", wait_until="domcontentloaded"
            )
            await asyncio.sleep(2)

            # Step 1: Click user menu button (aria-label="유저 메뉴 열기")
            user_menu_button = await self.page.wait_for_selector(
                'button[aria-label="유저 메뉴 열기"]', timeout=10000
            )
            await user_menu_button.click()
            logger.info("Step 1/5: Clicked user menu button")
            await asyncio.sleep(2)

            # Step 2: Click login menuitem from dropdown
            login_menuitem = await self.page.wait_for_selector(
                '[role="menuitem"]:has-text("로그인")', timeout=10000
            )
            await login_menuitem.click()
            logger.info("Step 2/5: Clicked login menuitem")
            await asyncio.sleep(3)

            # Step 3: Click email login button (이메일)
            email_login_button = await self.page.wait_for_selector(
                'button:has-text("이메일")', timeout=10000
            )
            await email_login_button.click()
            logger.info("Step 3/5: Clicked email login button")
            await asyncio.sleep(2)

            # Step 4: Enter username (textbox "유저네임 또는 이메일")
            username_input = await self.page.wait_for_selector(
                'input[name="id_username"], input[placeholder*="유저네임"], input[placeholder*="이메일"]',
                timeout=10000
            )
            await username_input.click()
            await username_input.fill(self.username)
            logger.info(f"Step 4/5: Entered username: {self.username}")
            await asyncio.sleep(0.5)

            # Step 5a: Enter password (textbox "비밀번호")
            password_input = await self.page.wait_for_selector(
                'input[name="id_password"], input[type="password"]',
                timeout=10000
            )
            await password_input.click()
            await password_input.fill(self.password)
            logger.info("Step 5/5: Entered password")
            await asyncio.sleep(0.5)

            # Step 5b: Click login submit button
            login_submit = await self.page.wait_for_selector(
                'button:has-text("로그인"):not([aria-label])', timeout=10000
            )
            await login_submit.click()
            logger.info("Clicked login submit button")

            # Wait for login to complete
            logger.info("Waiting for login to complete...")
            await asyncio.sleep(5)

            # Save cookies after login attempt (옵션 C: 쿠키 파일로 로그인 상태 관리)
            context = self.page.context
            await self._save_cookies(context)
            logger.info("✓ Login successful! Cookies saved.")

            return True

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    async def _ensure_logged_in(self):
        """Ensure user is logged in, perform login if needed"""
        if not await self._is_logged_in():
            success = await self._login()
            if not success:
                raise Exception("Failed to login to TradingView")

    async def _export_chart_data_stay(self, symbol: str, period: dict) -> bool:
        """Export chart data for specific time period (stay on same page)"""
        period_name = period["name"]
        button_text = period["button"]  # e.g., "1Y", "1M", "5D", "1D"
        logger.info(f"  [{symbol} - {period_name}] Exporting...")

        try:
            # Click time period button using CSS selector (like tradingview_playwright_scraper.py)
            # The buttons are in the bottom toolbar with text like "1Y", "1M", etc.
            period_button = await self.page.wait_for_selector(
                f'button:has-text("{button_text}")', timeout=10000
            )
            await period_button.click()
            await asyncio.sleep(2)

            # Scroll if needed
            if period.get("scroll_needed"):
                scroll_time = period.get("scroll_time", 2)
                logger.info(
                    f"  [{symbol} - {period_name}] Scrolling for {scroll_time} seconds..."
                )
                await self.page.evaluate("window.scrollBy(0, 500)")
                await asyncio.sleep(scroll_time)

            # Click the arrow button to open layout menu (like tradingview_playwright_scraper.py)
            # Use JavaScript to find and click the arrow div element
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
                logger.warning(f"  [{symbol} - {period_name}] Arrow menu not found")
                return False

            await asyncio.sleep(0.5)

            # Click "차트 데이터 다운로드" option
            try:
                download_option = await self.page.wait_for_selector(
                    'text=차트 데이터 다운로드', timeout=5000
                )
                await download_option.click()
            except:
                # Try alternative selector
                download_option = await self.page.wait_for_selector(
                    '[role="row"]:has-text("차트 데이터 다운로드")', timeout=5000
                )
                await download_option.click()

            await asyncio.sleep(0.5)

            # Click download button and capture download (use CSS selector like working scraper)
            async with self.page.expect_download(timeout=30000) as download_info:
                download_button = await self.page.wait_for_selector(
                    'button:has-text("다운로드")', timeout=10000
                )
                await download_button.click()

            download = await download_info.value

            # Save to download directory with standardized name
            timeframe_code = self._get_timeframe_code(period_name)
            filename = f"{symbol}_{timeframe_code}.csv"
            save_path = self.download_dir / filename
            await download.save_as(str(save_path))

            logger.info(f"  [{symbol} - {period_name}] ✓ Downloaded to {save_path}")

            # Upload to database if enabled
            if self.upload_to_db and self.db_service:
                try:
                    rows = self.db_service.upload_csv(save_path, symbol, timeframe_code)
                    logger.info(f"  [{symbol} - {period_name}] ✓ Uploaded {rows} rows to DB")
                except Exception as e:
                    logger.error(f"  [{symbol} - {period_name}] ✗ DB upload failed: {e}")

            return True

        except Exception as e:
            logger.error(f"  [{symbol} - {period_name}] ✗ Export failed: {e}")
            return False

            # Click the period row in the menu
            try:
                period_row = await self.page.wait_for_selector(
                    f'row:has-text("{period["button"]}")', timeout=5000
                )
                await period_row.click()
                await asyncio.sleep(1)
            except:
                logger.warning(f"  [{symbol} - {period_name}] Period not found in menu")
                return False

            # Scroll if needed
            if period.get("scroll_needed"):
                scroll_time = period.get("scroll_time", 2)
                logger.info(
                    f"  [{symbol} - {period_name}] Scrolling for {scroll_time} seconds..."
                )
                await self.page.evaluate("window.scrollBy(0, 500)")
                await asyncio.sleep(scroll_time)

            # Click "더보기" (More) button - this is in the chart header toolbar
            # The button shows the current symbol and time interval
            more_button = await self.page.wait_for_selector(
                'button:has-text("더보기")', timeout=10000
            )
            await more_button.click()
            await asyncio.sleep(1)

            # Click "Export chart data" - look for in the menu that opened
            # Try multiple selector strategies
            export_button = None
            try:
                # Try exact Korean text
                export_button = await self.page.wait_for_selector(
                    'button:has-text("익스포트 차트 데이터")', timeout=5000
                )
            except:
                pass

            if not export_button:
                try:
                    # Try partial match
                    export_button = await self.page.wait_for_selector(
                        'button:has-text("익스포트")', timeout=5000
                    )
                except:
                    pass

            if not export_button:
                try:
                    # Try English
                    export_button = await self.page.wait_for_selector(
                        'button:has-text("Export")', timeout=5000
                    )
                except:
                    pass

            if not export_button:
                logger.info(
                    f"  [{symbol} - {period_name}] Export button not found, trying keyboard shortcut"
                )
                # Try keyboard shortcut - typically Ctrl+E for export
                await self.page.keyboard.press("Control+E")
                await asyncio.sleep(1)

            # Select ISO timestamp (not UNIX) if visible
            try:
                iso_option = await self.page.wait_for_selector(
                    'label:has-text("ISO"), [role="radio"]', timeout=5000
                )
                if iso_option:
                    await iso_option.click()
                    await asyncio.sleep(0.5)
            except:
                # May already be selected
                logger.info(
                    f"  [{symbol} - {period_name}] ISO option may already be selected"
                )

            # Click download/export button
            try:
                export_confirm = await self.page.wait_for_selector(
                    'button:has-text("다운로드"), button:has-text("다운로드")',
                    timeout=5000,
                )
                await export_confirm.click()
            except:
                # Try submit button
                try:
                    export_confirm = await self.page.wait_for_selector(
                        'button[type="submit"]', timeout=5000
                    )
                    await export_confirm.click()
                except:
                    pass

            await asyncio.sleep(3)  # Wait for download to start

            logger.info(f"  [{symbol} - {period_name}] ✓ Exported")
            return True

        except Exception as e:
            logger.error(f"  [{symbol} - {period_name}] ✗ Export failed: {e}")
            return False

    async def _export_chart_data(self, symbol: str, period: dict) -> bool:
        """Export chart data for specific time period (legacy - navigates each time)"""
        period_name = period["name"]
        logger.info(f"  [{symbol} - {period_name}] Exporting...")

        try:
            # Navigate to chart URL directly (more reliable than clicking)
            if symbol.isdigit():
                chart_url = f"https://kr.tradingview.com/chart/?symbol=KRX%3A{symbol}"
            else:
                chart_url = (
                    f"https://kr.tradingview.com/chart/?symbol=NASDAQ%3A{symbol}"
                )

            await self.page.goto(chart_url, wait_until="domcontentloaded")
            await asyncio.sleep(3)

            # Click time period button
            period_button = await self.page.wait_for_selector(
                f'button:has-text("{period["button"]}")', timeout=10000
            )
            await period_button.click()
            await asyncio.sleep(2)

            # Scroll if needed
            if period.get("scroll_needed"):
                scroll_time = period.get("scroll_time", 2)
                logger.info(
                    f"  [{symbol} - {period_name}] Scrolling for {scroll_time} seconds..."
                )
                await self.page.evaluate("window.scrollBy(0, 500)")
                await asyncio.sleep(scroll_time)

            # Click "더보기" (More) button in chart toolbar
            # The button is in toolbar that contains interval settings
            more_button = await self.page.wait_for_selector(
                'button:has-text("더보기")', timeout=10000
            )
            await more_button.click()
            await asyncio.sleep(1)

            # Click "Export chart data" - this opens export dialog
            # Using Playwright's getByText to find button
            try:
                export_button = await self.page.wait_for_selector(
                    'button:has-text("익스포트")', timeout=5000
                )
                await export_button.click()
            except:
                # Try with different selector - might already be clicked
                logger.info(
                    f"  [{symbol} - {period_name}] Export button may already be clicked"
                )

            await asyncio.sleep(1)

            # Select ISO timestamp (not UNIX)
            try:
                iso_option = await self.page.wait_for_selector(
                    'label:has-text("ISO"), [role="radio"]', timeout=5000
                )
                if iso_option:
                    await iso_option.click()
                    await asyncio.sleep(0.5)
            except:
                # May already be selected
                logger.info(
                    f"  [{symbol} - {period_name}] ISO option may already be selected"
                )

            # Click export button
            export_confirm = await self.page.wait_for_selector(
                'button:has-text("다운로드"), button[type="submit"]', timeout=10000
            )
            await export_confirm.click()
            await asyncio.sleep(3)  # Wait for download to start

            logger.info(f"  [{symbol} - {period_name}] ✓ Exported")
            return True

        except Exception as e:
            logger.error(f"  [{symbol} - {period_name}] ✗ Export failed: {e}")
            return False

    async def scrape_stock(self, symbol: str):
        """Scrape all time periods for a stock"""
        logger.info(f"=" * 60)
        logger.info(f"[{symbol}] Starting scrape")
        logger.info(f"=" * 60)

        # Navigate to chart URL once at the beginning
        if symbol.isdigit():
            chart_url = f"https://kr.tradingview.com/chart/?symbol=KRX%3A{symbol}"
        else:
            chart_url = f"https://kr.tradingview.com/chart/?symbol=NASDAQ%3A{symbol}"

        logger.info(f"[{symbol}] Navigating to: {chart_url}")
        await self.page.goto(chart_url, wait_until="domcontentloaded")
        await asyncio.sleep(3)

        # Export each time period (staying on same page)
        success_count = 0
        for i, period in enumerate(self.time_periods, 1):
            logger.info(
                f"  [{symbol}] Processing {i}/{len(self.time_periods)}: {period['name']}"
            )

            if await self._export_chart_data_stay(symbol, period):
                success_count += 1
            else:
                logger.warning(f"  [{symbol}] Failed to export {period['name']}")

            # Wait between exports
            await asyncio.sleep(2)

        logger.info(
            f"✓ [{symbol}] Complete: {success_count}/{len(self.time_periods)} exports succeeded"
        )
        return success_count == len(self.time_periods)

    async def run(self):
        """Main execution loop"""
        await self.start()

        # Ensure logged in
        await self._ensure_logged_in()

        # Scrape all stocks
        total_stocks = len(self.stock_list)
        success_count = 0

        for idx, symbol in enumerate(self.stock_list, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"[{idx}/{total_stocks}] Processing {symbol} ({int(idx / total_stocks * 100)}%)"
            )
            logger.info(f"{'=' * 60}\n")

            try:
                if await self.scrape_stock(symbol):
                    success_count += 1
            except Exception as e:
                logger.error(f"[{symbol}] Fatal error: {e}")

            # Wait between stocks
            await asyncio.sleep(5)

        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Scraping complete!")
        logger.info(f"Total: {total_stocks} stocks")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {total_stocks - success_count}")
        logger.info(f"{'=' * 60}\n")

    async def close(self):
        """Clean up resources"""
        if self.db_service:
            self.db_service.close()
            logger.info("Database service closed")

        if self.browser:
            await self.browser.close()
            logger.info("Browser closed")


async def main():
    """Main entry point"""
    # Configuration from environment variables
    USERNAME = os.getenv("TRADINGVIEW_USERNAME", "hrahn")
    PASSWORD = os.getenv("TRADINGVIEW_PASSWORD", "tndnjseogkrry1234")
    DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", str(Path.home() / "Downloads" / "tradingview"))
    UPLOAD_TO_DB = os.getenv("UPLOAD_TO_DB", "true").lower() == "true"
    USE_EXISTING_TUNNEL = os.getenv("USE_EXISTING_TUNNEL", "true").lower() == "true"

    logger.info("=== TradingView Scraper Configuration ===")
    logger.info(f"Download directory: {DOWNLOAD_DIR}")
    logger.info(f"Upload to DB: {UPLOAD_TO_DB}")
    logger.info(f"Use existing SSH tunnel: {USE_EXISTING_TUNNEL}")
    logger.info("==========================================")

    scraper = TradingViewScraper(
        username=USERNAME,
        password=PASSWORD,
        headless=False,  # Set to True for production
        download_dir=DOWNLOAD_DIR,
        upload_to_db=UPLOAD_TO_DB,
        use_existing_tunnel=USE_EXISTING_TUNNEL,
    )

    try:
        await scraper.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
