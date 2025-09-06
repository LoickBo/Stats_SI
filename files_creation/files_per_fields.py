import pandas as pd

def clean_csv_files(input_file_path, output_file_path, ticker):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)

    df.rename(columns={df.columns[0]: 'DATE'}, inplace=True)

    df['DATE'] = pd.to_datetime(df['DATE'])

    if f'{ticker}_PX_LAST' in df.columns and df.drop(columns=['PX_LAST', 'DATE']).isna().all().all():
        print(f"{ticker} disregarded: only 'PX_LAST' is populated.")
        return
    
    filtered_df = df[df['DATE'] >= '2006-07-03']

    df_cleaned = filtered_df.dropna(how='all', axis=1)
    df_cleaned = df_cleaned.fillna(0)
    df_cleaned.to_csv(output_file_path, index=False)

    print("{ticker} cleaned successfully!")


