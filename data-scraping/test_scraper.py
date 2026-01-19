#!/usr/bin/env python3
"""
Quick test script for TradingView scraper
Tests login and basic functionality
"""

import asyncio
import os
from pathlib import Path

# Add parent directory to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from tradingview_scraper import TradingViewScraper


async def test_login():
    """Test only login functionality"""

    # Get credentials
    username = os.getenv("TRADINGVIEW_USERNAME", "hrahn")
    password = os.getenv("TRADINGVIEW_PASSWORD", "tndnjseogkrry1234")

    print("=" * 60)
    print("TradingView Login Test")
    print("=" * 60)
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password)}")
    print("=" * 60)
    print()

    scraper = TradingViewScraper(
        username=username,
        password=password,
        headless=False,  # Show browser for testing
    )

    try:
        print("Starting browser...")
        await scraper.start()

        print("\nChecking login status...")
        is_logged_in = await scraper._is_logged_in()

        if is_logged_in:
            print("✓ Already logged in (valid cookies)")
        else:
            print("Not logged in, attempting login...")
            success = await scraper._login()
            if success:
                print("✓ Login successful!")
            else:
                print("✗ Login failed!")
                return False

        # Wait a bit to see the result
        print("\nWaiting 5 seconds to see result...")
        await asyncio.sleep(5)

        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        print("\nClosing browser...")
        await scraper.close()


async def test_single_stock():
    """Test scraping a single stock"""

    username = os.getenv("TRADINGVIEW_USERNAME", "hrahn")
    password = os.getenv("TRADINGVIEW_PASSWORD", "tndnjseogkrry1234")

    print("=" * 60)
    print("TradingView Single Stock Test (AAPL)")
    print("=" * 60)
    print()

    scraper = TradingViewScraper(
        username=username,
        password=password,
        headless=False,
    )

    # Override stock list to test just one
    scraper.stock_list = ["AAPL"]

    try:
        await scraper.run()
        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await scraper.close()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test TradingView scraper")
    parser.add_argument(
        "test_type",
        choices=["login", "stock"],
        help="Type of test: 'login' (test login only) or 'stock' (test full scraping)",
    )

    args = parser.parse_args()

    if args.test_type == "login":
        success = asyncio.run(test_login())
    else:
        success = asyncio.run(test_single_stock())

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
