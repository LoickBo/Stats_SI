import pandas as pd
import os
from tqdm import tqdm

def clean_csv_files(input_file_path, output_file_path, ticker):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)

    df.rename(columns={df.columns[0]: 'DATE'}, inplace=True)

    df['DATE'] = pd.to_datetime(df['DATE'])

    if f'{ticker}_PX_LAST' in df.columns and df.drop(columns=[f'{ticker}_PX_LAST', 'DATE']).isna().all().all():
        print(f"{ticker} disregarded: only 'PX_LAST' is populated.")
        return
    
    filtered_df = df[df['DATE'] >= '2006-07-03']

    df_cleaned = filtered_df.dropna(how='all', axis=1)
    df_cleaned = df_cleaned.fillna(0)
    df_cleaned.to_csv(output_file_path, index=False)

    print(f"{ticker} cleaned successfully!")


tickers = pd.read_excel('files_creation/data/input/tickers.xlsx')['Ticker'].tolist()

output_dir = 'files_creation/data/output_cleaned'
os.makedirs(output_dir, exist_ok=True)

input_dir = 'files_creation/data/output'
os.makedirs(input_dir, exist_ok=True)

for ticker in tqdm(tickers, desc="Processing tickers"):
    output_file = os.path.join(output_dir, f'{ticker}_cleaned.csv')
    input_file = os.path.join(input_dir, f'{ticker}.csv')

    if os.path.exists(output_file):
        print(f"✅ Skipping {ticker} (already cleaned).")
        continue
    else :
        print(f"\nCleaning data for {ticker}...")

    clean_csv_files(input_file, output_file, ticker)



