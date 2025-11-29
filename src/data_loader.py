import os
import yfinance as yf
import pandas as pd
from datetime import datetime

def download_data(ticker, start_date, end_date, save_dir='data/raw'):
    """
    Downloads historical stock data using yfinance and saves it to a CSV file.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        save_dir (str): The directory to save the CSV file. Defaults to 'data/raw'.

    Returns:
        pd.DataFrame: The downloaded DataFrame.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    # Download data
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            print(f"No data found for {ticker} in the specified range.")
            return pd.DataFrame()

        # Construct filename
        filename = f"{ticker}_{start_date}_{end_date}.csv"
        filepath = os.path.join(save_dir, filename)

        # Save to CSV
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")

        return df

    except Exception as e:
        print(f"An error occurred while downloading data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Test the function with AAPL
    ticker_symbol = 'AAPL'
    start = '2015-01-01'
    end = datetime.today().strftime('%Y-%m-%d')    

    df_aapl = download_data(ticker_symbol, start, end)
    
    if not df_aapl.empty:
        print("\nHead of the downloaded data:")
        print(df_aapl.head())   
        print("\nShape of the downloaded data:")
        print(df_aapl.shape)    