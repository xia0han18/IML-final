import os
import requests
import yfinance as yf

def fetch_sp500_data():
    """
    Downloads historical S&P 500 data from Yahoo Finance and saves it to a raw CSV file.
    Includes session headers to avoid Rate Limiting.
    """
    ticker_symbol = "^GSPC"
    start_date = "1927-12-30"
    end_date = "2025-12-30"

    print(f"Downloading data for {ticker_symbol}...")
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date, session=session)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    if data.empty:
        print("No data found. (You might still be rate-limited. Try Option 2 below).")
        return

    raw_path = os.path.join("data", "raw")
    os.makedirs(raw_path, exist_ok=True)
    
    file_path = os.path.join(raw_path, "sp500_raw.csv")
    data.to_csv(file_path)
    print(f"Data saved to: {file_path}")

if __name__ == "__main__":
    fetch_sp500_data()