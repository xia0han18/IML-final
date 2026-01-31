import os

import numpy as np
import pandas as pd

def process_data():
    """Clean raw S&P 500 data and engineer volatility features."""
    input_path = os.path.join("data", "raw", "sp500_raw.csv")
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    print("Processing data...")
    try:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        price_col = df.columns[0]
        print(f"Warning: Using '{price_col}' as price column.")

    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

    df[price_col] = df[price_col].ffill()
    df = df.dropna(subset=[price_col])

    valid_prices = df[price_col] > 0
    df = df.loc[valid_prices].copy()

    df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['Target_Vol'] = df['Log_Return'] ** 2

    for lag in range(1, 6):
        df[f'Vol_Lag_{lag}'] = df['Target_Vol'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Log_Return'].shift(lag)

    df['Vol_Roll_Mean_5'] = df['Target_Vol'].rolling(window=5).mean().shift(1)
    df['Vol_Roll_Mean_21'] = df['Target_Vol'].rolling(window=21).mean().shift(1)

    df_clean = df.dropna()

    if df_clean.empty:
        print("Error: Cleaning resulted in an empty dataset.")
        return

    output_path = os.path.join("data", "processed")
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, "sp500_clean.csv")
    df_clean.to_csv(output_file)
    
    print(f"Processed data saved to {output_file}")
    print(f"Cleaned shape: {df_clean.shape}")

if __name__ == "__main__":
    process_data()