import sys
import os
from sqlalchemy import text

# Add current directory to sys.path to allow importing db_service
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from db_service import DatabaseService
except ImportError:
    # If running from project root
    sys.path.append(os.path.join(current_dir, "data-scraping"))
    from db_service import DatabaseService


def check_table_schema(table_name="NVDA_D"):
    # Use environment variables if needed, otherwise defaults
    db = DatabaseService()
    try:
        db.connect()
        print(f"Checking schema for table: {table_name}")

        with db.engine.connect() as conn:
            try:
                # DESCRIBE query execution
                result = conn.execute(text(f"DESCRIBE `{table_name}`"))
                columns = result.fetchall()

                print(
                    f"{'Field':<20} {'Type':<20} {'Null':<10} {'Key':<10} {'Default':<20} {'Extra':<20}"
                )
                print("-" * 100)
                for col in columns:
                    # Access by index for tuple results
                    field = col[0]
                    type_ = col[1]
                    null = col[2]
                    key = col[3]
                    default = col[4]
                    extra = col[5]

                    print(
                        f"{str(field):<20} {str(type_):<20} {str(null):<10} {str(key):<10} {str(default):<20} {str(extra):<20}"
                    )
            except Exception as e:
                print(f"Table {table_name} might not exist or error: {e}")

    except Exception as e:
        print(f"Error checking schema: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    check_table_schema()
