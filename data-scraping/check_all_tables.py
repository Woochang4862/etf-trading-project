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


def check_all_tables_status():
    db = DatabaseService()

    try:
        db.connect()
        print(f"\n{'=' * 80}")
        print(f"Checking ALL Database Tables Status at {datetime.now()}")
        print(f"{'Table':<25} {'Row Count':<12} {'Last Date':<25} {'Status'}")
        print(f"{'-' * 80}")

        with db.engine.connect() as conn:
            # Get all tables
            tables_query = text("SHOW TABLES")
            tables_result = conn.execute(tables_query).fetchall()
            all_tables = [row[0] for row in tables_result]

            # Sort tables for better readability
            all_tables.sort()

            for table_name in all_tables:
                try:
                    # Check if table has 'time' column
                    col_query = text(f"SHOW COLUMNS FROM `{table_name}` LIKE 'time'")
                    col_result = conn.execute(col_query).fetchone()

                    if not col_result:
                        print(
                            f"{table_name:<25} {'N/A':<12} {'No time column':<25} ℹ️ Info Table"
                        )
                        continue

                    query = text(f"SELECT COUNT(*), MAX(time) FROM `{table_name}`")
                    result = conn.execute(query).fetchone()
                    count = result[0]
                    last_date = result[1]

                    status = "✅ OK"
                    if not last_date:
                        status = "⚠️ Empty"
                    elif last_date.year < 2025:
                        status = "⚠️ Old Data"

                    last_date_str = str(last_date) if last_date else "None"
                    print(f"{table_name:<25} {count:<12} {last_date_str:<25} {status}")

                except Exception as e:
                    print(f"{table_name:<25} Error: {e}")

        print(f"{'=' * 80}\n")
        print(f"Total Tables Checked: {len(all_tables)}")

    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    check_all_tables_status()
