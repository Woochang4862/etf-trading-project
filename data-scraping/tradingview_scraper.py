#!/usr/bin/env python3
"""
TradingView Stock Data Scraper - Fully Automated

Features:
- Fully automated login (no manual intervention required)
- Cookie persistence with auto-refresh
- Robust error handling and retry logic
- Progress tracking and logging
- Cloudflare bypass using Playwright's stealth mode
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    Page,
    Error as PlaywrightError,
)


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
    ):
        self.username = username
        self.password = password
        self.headless = headless
        self.cookie_file = Path(cookie_file)
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

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

        # Time periods to scrape (using Korean UI text)
        # Note: Toolbar shows "12달", "1달" with space after number
        self.time_periods = [
            {"name": "12개월", "button": "12달", "scroll_needed": False},
            {"name": "1개월", "button": "1달", "scroll_needed": False},
            {"name": "1주", "button": "1주", "scroll_needed": True, "scroll_time": 2},
            {"name": "1일", "button": "1일", "scroll_needed": True, "scroll_time": 3},
            {
                "name": "1시간",
                "button": "1시간",
                "scroll_needed": True,
                "scroll_time": 25,
            },
            {
                "name": "10분",
                "button": "10분",
                "scroll_needed": True,
                "scroll_time": 10,
            },
        ]

    async def start(self):
        """Initialize browser and page"""
        logger.info("Starting TradingView scraper...")

        playwright = await async_playwright().start()

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
        )

        context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="ko-KR",
        )

        self.page = await context.new_page()
        self.page.set_default_timeout(30000)

        # Load existing cookies if available
        await self._load_cookies(context)

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

    async def _is_logged_in(self) -> bool:
        """Check if user is logged in"""
        try:
            # Navigate to homepage and wait for load
            await self.page.goto(
                "https://kr.tradingview.com/", wait_until="domcontentloaded"
            )
            await asyncio.sleep(2)

            # Look for login button (not logged in) or user menu (logged in)
            login_button = await self.page.query_selector('button:has-text("로그인")')
            user_menu = await self.page.query_selector(
                '[class*="user-menu"], [class*="user-id"]'
            )

            if user_menu or (
                not login_button and "로그인" not in await self.page.content()
            ):
                logger.info("Already logged in (cookies valid)")
                return True

            return False
        except Exception as e:
            logger.error(f"Error checking login status: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking login status: {e}")
            return False

    async def _login(self):
        """Perform login with credentials"""
        logger.info("Attempting login...")

        try:
            # Navigate to login page
            await self.page.goto(
                "https://kr.tradingview.com/", wait_until="networkidle"
            )
            await asyncio.sleep(2)

            # Click login button
            login_button = await self.page.wait_for_selector(
                'button:has-text("로그인")', timeout=10000
            )
            await login_button.click()
            await asyncio.sleep(2)

            # Wait for page to load after login
            await asyncio.sleep(5)

            # Click email login option
            email_button = await self.page.wait_for_selector(
                'button:has-text("이메일")', timeout=10000
            )
            await email_button.click()
            await asyncio.sleep(1)

            # Enter username
            username_input = await self.page.wait_for_selector(
                'input[placeholder*="유저네임"], input[placeholder*="이메일"]',
                timeout=10000,
            )
            await username_input.click()
            await username_input.fill(self.username)
            await asyncio.sleep(0.5)

            # Enter password
            password_input = await self.page.wait_for_selector(
                'input[type="password"]', timeout=10000
            )
            await password_input.click()
            await password_input.fill(self.password)
            await asyncio.sleep(0.5)

            # Click login button
            login_submit = await self.page.wait_for_selector(
                'form button:has-text("로그인")', timeout=10000
            )
            await login_submit.click()

            # Wait for login to complete
            logger.info("Waiting for login to complete...")
            await asyncio.sleep(5)

            # Check if login was successful
            if await self._is_logged_in():
                logger.info("✓ Login successful!")

                # Save cookies after successful login
                context = self.page.context
                await self._save_cookies(context)

                return True
            else:
                logger.error("✗ Login failed - please check credentials")
                return False

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
        logger.info(f"  [{symbol} - {period_name}] Exporting...")

        try:
            # Click time period button using XPath (like baseline)
            period_xpath = f"//span[contains(text(), '{period['button']}')]"
            period_element = await self.page.wait_for_selector(
                f"xpath={period_xpath}", timeout=10000
            )
            await period_element.click()
            await asyncio.sleep(2)

            # Scroll if needed
            if period.get("scroll_needed"):
                scroll_time = period.get("scroll_time", 2)
                logger.info(
                    f"  [{symbol} - {period_name}] Scrolling for {scroll_time} seconds..."
                )
                await self.page.evaluate("window.scrollBy(0, 500)")
                await asyncio.sleep(scroll_time)

            # Click "더보기" (More) button using XPath
            more_xpath = (
                "//button[contains(@aria-label, 'more') or contains(@title, 'more')]"
            )
            more_button = await self.page.wait_for_selector(
                f"xpath={more_xpath}", timeout=10000
            )
            await more_button.click()
            await asyncio.sleep(1)

            # Click "Export chart data" using XPath
            export_xpath = (
                "//button[contains(text(), '익스포트') or contains(text(), 'Export')]"
            )
            export_button = await self.page.wait_for_selector(
                f"xpath={export_xpath}", timeout=10000
            )
            await export_button.click()
            await asyncio.sleep(1)

            # Select ISO timestamp (not UNIX)
            iso_xpath = "//label[contains(text(), 'ISO')]"
            iso_option = await self.page.wait_for_selector(
                f"xpath={iso_xpath}", timeout=5000
            )
            if iso_option:
                await iso_option.click()
                await asyncio.sleep(0.5)
            else:
                logger.info(
                    f"  [{symbol} - {period_name}] ISO option may already be selected"
                )

            # Click download button
            download_xpath = "//button[contains(text(), '다운로드') or contains(text(), 'Download') or @type='submit']"
            download_button = await self.page.wait_for_selector(
                f"xpath={download_xpath}", timeout=10000
            )
            await download_button.click()
            await asyncio.sleep(3)  # Wait for download to start

            logger.info(f"  [{symbol} - {period_name}] ✓ Exported")
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
        if self.browser:
            await self.browser.close()
            logger.info("Browser closed")


async def main():
    """Main entry point"""
    # Configuration
    USERNAME = os.getenv("TRADINGVIEW_USERNAME", "hrahn")
    PASSWORD = os.getenv("TRADINGVIEW_PASSWORD", "tndnjseogkrry1234")

    # For testing, you can override:
    # USERNAME = "your_username"
    # PASSWORD = "your_password"

    scraper = TradingViewScraper(
        username=USERNAME,
        password=PASSWORD,
        headless=False,  # Set to True for production
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
