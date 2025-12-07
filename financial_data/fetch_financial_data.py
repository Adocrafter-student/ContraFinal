# fetch_financial_data.py - Financial Data Fetching Module
# Fetches and saves stock price data for Tesla, Amazon, and Meta

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Companies to analyze
COMPANIES = {
    "TSLA": "Tesla",
    "AMZN": "Amazon", 
    "META": "Meta"
}

# Market benchmark
MARKET = "^GSPC"  # S&P 500

# Date range (same as contra.py)
START = "2018-01-01"
END = "2020-12-31"

# Output directory
DATA_DIR = "../datasets"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    print(f"Output directory: {path}")

def save_data(df, filename, format='parquet'):
    """Save dataframe to file (parquet or csv)"""
    ensure_dir(DATA_DIR)
    filepath = os.path.join(DATA_DIR, filename)
    
    if format == 'parquet':
        df.to_parquet(filepath)
        print(f"Saved: {filename} ({len(df)} rows)")
    elif format == 'csv':
        df.to_csv(filepath)
        print(f"Saved: {filename} ({len(df)} rows)")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return filepath

# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_stock_data(ticker, start, end):
    """
    Fetch stock price data for a single ticker.
    Returns DataFrame with adjusted close prices.
    """
    try:
        print(f"  Fetching {ticker}...")
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if data.empty:
            print(f" No data retrieved for {ticker}")
            return None
            
        # Extract close prices
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data[['Close']] if 'Close' in data.columns else data
            
        close = close.to_frame() if isinstance(close, pd.Series) else close
        close.columns = [ticker]
        
        print(f"Got {len(close)} trading days")
        return close
        
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return None

def fetch_all_data(tickers, market, start, end):
    """
    Fetch data for all tickers and market benchmark.
    Returns combined DataFrame with all close prices.
    """
    print(f"\nFetching stock data from {start} to {end}...")
    
    all_data = []
    
    # Fetch market data first
    market_data = fetch_stock_data(market, start, end)
    if market_data is not None:
        all_data.append(market_data)
    
    # Fetch individual company data
    for ticker in tickers:
        ticker_data = fetch_stock_data(ticker, start, end)
        if ticker_data is not None:
            all_data.append(ticker_data)
    
    if not all_data:
        raise RuntimeError("No data retrieved for any ticker")
    
    # Combine all data
    combined = pd.concat(all_data, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()
    
    print(f"\nSuccessfully fetched data for {len(combined.columns)} tickers")
    return combined

def calculate_returns(prices):
    """
    Calculate daily returns from price data.
    Returns DataFrame with percentage returns.
    """
    print(f"\nCalculating daily returns...")
    returns = prices.pct_change().dropna()
    print(f"Calculated returns for {len(returns)} trading days")
    return returns

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print("Financial Data Fetching Module")
    print("=" * 70)
    print(f"\nCompanies: {', '.join([f'{t} ({n})' for t, n in COMPANIES.items()])}")
    print(f"Market: {MARKET}")
    print(f"Period: {START} to {END}")
    
    # Fetch all price data
    tickers = list(COMPANIES.keys())
    prices = fetch_all_data(tickers, MARKET, START, END)
    
    # Calculate returns
    returns = calculate_returns(prices)
    
    # Save data
    print(f"\nSaving data to {DATA_DIR}/...")
    
    # Save prices
    save_data(prices, "stock_prices.parquet", format='parquet')
    save_data(prices, "stock_prices.csv", format='csv')  # Also save CSV for easy viewing
    
    # Save returns
    save_data(returns, "stock_returns.parquet", format='parquet')
    save_data(returns, "stock_returns.csv", format='csv')
    
    # Save metadata
    metadata = {
        "fetch_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": START,
        "end_date": END,
        "tickers": tickers,
        "market": MARKET,
        "num_trading_days": len(prices),
        "columns": list(prices.columns)
    }
    
    metadata_df = pd.DataFrame([metadata])
    save_data(metadata_df, "metadata.csv", format='csv')
    
    # Display summary statistics
    print(f"\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"\nPrice Data:")
    print(f"  Trading days: {len(prices)}")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Tickers: {', '.join(prices.columns)}")
    
    print(f"\nReturns Summary:")
    print(returns.describe().round(6))
    
    print(f"\nData fetching complete!")
    print(f"Files saved in: {DATA_DIR}/")
    print(f"\nFiles created:")
    print(f"stock_prices.parquet - Raw price data (Parquet)")
    print(f"stock_prices.csv - Raw price data (CSV)")
    print(f"stock_returns.parquet - Daily returns (Parquet)")
    print(f"stock_returns.csv - Daily returns (CSV)")
    print(f"metadata.csv - Fetch metadata")
    print()

if __name__ == "__main__":
    main()

