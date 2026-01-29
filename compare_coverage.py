import sys
import os
from sqlalchemy import text
from datetime import datetime

# Add data-scraping to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "data-scraping"))

# Import STOCK_LIST
try:
    from tradingview_playwright_scraper_upload import STOCK_LIST
    from db_service import DatabaseService
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


def analyze_coverage():
    print(f"Analyzing Scraper Coverage vs Database...")
    print(f"Scraper Target Count: {len(STOCK_LIST)}")

    db = DatabaseService()
    try:
        db.connect()

        # Get all distinct symbols from DB tables
        with db.engine.connect() as conn:
            tables_query = text("SHOW TABLES")
            tables_result = conn.execute(tables_query).fetchall()

            # Extract symbols (assume format SYMBOL_TIMEFRAME)
            db_symbols = set()
            db_table_status = {}  # {symbol: is_updated}

            for row in tables_result:
                table = row[0]
                if "_" not in table:
                    continue

                parts = table.split("_")
                symbol = parts[0]
                timeframe = parts[1]

                db_symbols.add(symbol)

                # Check update status for Daily tables only to verify recency
                if timeframe == "D":
                    try:
                        query = text(f"SELECT MAX(time) FROM `{table}`")
                        last_date = conn.execute(query).scalar()
                        is_updated = last_date and last_date.year >= 2026
                        db_table_status[symbol] = is_updated
                    except:
                        pass

        # Analysis
        target_set = set(STOCK_LIST)

        # 1. Scraper Targets NOT in DB (Missing tables)
        missing_in_db = target_set - db_symbols

        # 2. DB Symbols NOT in Scraper (Ignored by current run)
        ignored_by_scraper = db_symbols - target_set

        # 3. Targets Status
        updated_targets = []
        pending_targets = []

        for sym in STOCK_LIST:
            if sym in db_table_status:
                if db_table_status[sym]:
                    updated_targets.append(sym)
                else:
                    pending_targets.append(sym)
            elif sym in db_symbols:
                # Has tables but no D table or error
                pending_targets.append(sym)

        print(f"\n{'=' * 60}")
        print(f"COVERAGE ANALYSIS")
        print(f"{'=' * 60}")

        print(
            f"\n1. Current Scraper Progress ({len(updated_targets)}/{len(STOCK_LIST)} Updated)"
        )
        print("-" * 30)
        print(f"✅ Updated (2026): {', '.join(sorted(updated_targets))}")
        print(f"⏳ Pending (Old):  {', '.join(sorted(pending_targets))}")

        if missing_in_db:
            print(f"❌ Missing in DB:  {', '.join(sorted(missing_in_db))}")

        print(f"\n2. Database Coverage")
        print("-" * 30)
        print(f"Total Symbols in DB: {len(db_symbols)}")
        print(f"Targeted by Scraper: {len(STOCK_LIST)}")
        print(f"NOT Targeted (Idle): {len(ignored_by_scraper)}")

        if len(ignored_by_scraper) > 0:
            print(f"\n⚠️ Stocks in DB but NOT in current Scraper List (Top 20):")
            print(f"{', '.join(sorted(list(ignored_by_scraper))[:20])} ...")

    finally:
        db.close()


if __name__ == "__main__":
    analyze_coverage()
