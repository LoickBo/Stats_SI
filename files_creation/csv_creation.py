import pandas as pd
import os
from xbbg import blp
import pandas as pd
from tqdm import tqdm

import blpapi

# Check API connection
session = blpapi.Session()
if session.start():
    print("API is accessible")
else:
    print("Cannot connect to Bloomberg API")


tickers = pd.read_excel('files_creation/data/input/tickers.xlsx')['Ticker'].tolist()
fields = pd.read_excel('files_creation/data/input/fields.xlsx')['FLDS_Mnemonic'].tolist()

start_date='1980-01-01'
end_date='2025-08-29'

output_dir = 'files_creation/data/output'
os.makedirs(output_dir, exist_ok=True)

merged_data = pd.DataFrame()

# Loop through fields and merge
for ticker in tqdm(tickers, desc="Processing tickers"):
    output_file = os.path.join(output_dir, f'{ticker}.csv')

    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"✅ Skipping {ticker} (already downloaded).")
        continue
    else :
        print(f"⬇️ Downloading data for {ticker}...")
    
    merged_data = pd.DataFrame()  
    for field in fields:
        data = blp.bdh(
            tickers=ticker,
            flds=field,
            start_date=start_date,
            end_date=end_date
        )
        if data.empty:
            print(f"⚠️ No data for {ticker} - {field}, skipping.")
            continue

        # Flatten multi-index columns into "Ticker_Field"
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        else:
            data.columns = [f"{ticker}_{field}"]

        # Merge on index (date)
        if merged_data.empty:
            merged_data = data
        else:
            merged_data = pd.merge(merged_data, data, left_index=True, right_index=True, how='outer')

    merged_data.to_csv(output_file)
