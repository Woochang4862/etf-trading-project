import sys
import os
from sqlalchemy import text
from datetime import datetime

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from db_service import DatabaseService
except ImportError:
    sys.path.append(os.path.join(current_dir, "data-scraping"))
    from db_service import DatabaseService


def check_recent_data():
    db = DatabaseService()

    # Check these symbols and timeframes
    targets = [("NVDA", "D"), ("AAPL", "D"), ("NVDA", "1h"), ("AAPL", "1h")]

    try:
        db.connect()
        print(f"\n{'=' * 60}")
        print(f"Checking Database Status at {datetime.now()}")
        print(f"{'Table':<15} {'Row Count':<12} {'Last Date':<25} {'Status'}")
        print(f"{'-' * 60}")

        with db.engine.connect() as conn:
            for symbol, timeframe in targets:
                table_name = db.get_table_name(symbol, timeframe)

                if not db.table_exists(table_name):
                    print(f"{table_name:<15} {'Not Found':<12} {'-':<25} ❌ Missing")
                    continue

                try:
                    query = text(f"SELECT COUNT(*), MAX(time) FROM `{table_name}`")
                    result = conn.execute(query).fetchone()
                    count = result[0]
                    last_date = result[1]

                    status = (
                        "✅ OK"
                        if last_date and last_date.year >= 2025
                        else "⚠️ Old Data"
                    )

                    print(f"{table_name:<15} {count:<12} {str(last_date):<25} {status}")

                except Exception as e:
                    print(f"{table_name:<15} Error: {e}")

        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    check_recent_data()
