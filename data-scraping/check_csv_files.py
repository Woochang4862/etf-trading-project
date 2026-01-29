import pandas as pd
import glob
import os


def check_csv_content():
    download_dir = "data-scraping/downloads"

    # Check NVDA and AAPL files created today
    files = glob.glob(os.path.join(download_dir, "*_20260129.csv"))
    target_files = [f for f in files if "NVDA" in f or "AAPL" in f]

    print(f"{'File Name':<30} {'Last Time':<25} {'Row Count':<10}")
    print("-" * 70)

    for file_path in sorted(target_files):
        try:
            df = pd.read_csv(file_path)

            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()

            time_col = None
            if "time" in df.columns:
                time_col = "time"
            elif "date" in df.columns:
                time_col = "date"

            if time_col:
                last_time = df[time_col].iloc[-1]
                file_name = os.path.basename(file_path)
                print(f"{file_name:<30} {str(last_time):<25} {len(df):<10}")
            else:
                print(f"{os.path.basename(file_path):<30} No time column found")

        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")


if __name__ == "__main__":
    check_csv_content()
