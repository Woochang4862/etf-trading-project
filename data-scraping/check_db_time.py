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


def check_db_time_format():
    db = DatabaseService()
    try:
        db.connect()
        with db.engine.connect() as conn:
            # Check NVDA_D table
            table_name = "NVDA_D"
            if db.table_exists(table_name):
                # Get a sample record to see time format
                query = text(
                    f"SELECT time FROM `{table_name}` ORDER BY time DESC LIMIT 5"
                )
                result = conn.execute(query).fetchall()

                print(f"Sample time values from {table_name}:")
                for row in result:
                    print(f"- {row[0]} (Type: {type(row[0])})")
            else:
                print(f"Table {table_name} does not exist.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    check_db_time_format()
