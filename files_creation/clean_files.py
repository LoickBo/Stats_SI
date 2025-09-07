import os
import pandas as pd
from tqdm import tqdm

def clean_csv_file(input_file_path: str, output_file_path: str, ticker: str) -> bool:
    """
    Clean a single CSV:
      - Rename first column to DATE
      - Parse DATE
      - Disregard if only PX_LAST has data (all other columns NA)
      - Filter DATE >= 2006-07-03
      - Drop fully-empty columns
      - Fill remaining NaN with 0
      - Save to output_file_path
    Returns True if cleaned & saved, False if skipped/disregarded.
    """
    if not os.path.exists(input_file_path):
        print(f"⚠️  {ticker} skipped: input file not found: {input_file_path}")
        return False

    try:
        df = pd.read_csv(input_file_path)
    except pd.errors.EmptyDataError:
        print(f"⚠️  {ticker} skipped: CSV is empty.")
        return False

    if df.shape[1] == 0:
        print(f"⚠️  {ticker} skipped: no columns.")
        return False

    # Ensure first column is DATE
    df = df.copy()
    df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
    # Parse dates
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    target_col = f"{ticker}_PX_LAST"
    if target_col in df.columns:
        other_cols = [c for c in df.columns if c not in ("DATE", target_col)]
        if len(other_cols) == 0 or df[other_cols].isna().all().all():
            print(f"ℹ️  {ticker} disregarded: only '{target_col}' is populated.")
            return False

    # Filter by date
    df = df[df["DATE"] >= "2006-07-03"]

    # Drop fully empty columns (keep DATE even if empty)
    keep_cols = ["DATE"]
    if target_col in df.columns:
        keep_cols.append(target_col)
    # Identify columns that are not entirely NA
    non_empty_cols = [c for c in df.columns if c in keep_cols or not df[c].isna().all()]
    df = df[non_empty_cols]

    # Fill NaNs with 0 for non-date columns
    non_date_cols = [c for c in df.columns if c != "DATE"]
    df[non_date_cols] = df[non_date_cols].fillna(0)

    # Save
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)

    print(f"✅ {ticker} cleaned successfully!")
    return True


# === Driver ===
tickers = pd.read_excel("files_creation/data/input/tickers.xlsx")["Ticker"].tolist()

output_dir = "files_creation/data/output_cleaned"
os.makedirs(output_dir, exist_ok=True)

input_dir = "files_creation/data/output"
os.makedirs(input_dir, exist_ok=True)

tickers_cleaned_list = []

for ticker in tqdm(tickers, desc="Processing tickers"):
    output_file = os.path.join(output_dir, f"{ticker}_cleaned.csv")
    input_file = os.path.join(input_dir, f"{ticker}.csv")

    if os.path.exists(output_file):
        print(f"✅ Skipping {ticker} (already cleaned).")
        continue
    else:
        print(f"\nCleaning data for {ticker}...")

    cleaned = clean_csv_file(input_file, output_file, ticker)
    if cleaned:
        tickers_cleaned_list.append(ticker)

# Save tickers that were actually cleaned
df_tickers_cleaned = pd.DataFrame(tickers_cleaned_list, columns=["Ticker"])
df_tickers_cleaned.to_csv("statistics/data/input/tickers_cleaned.csv", index=False)
