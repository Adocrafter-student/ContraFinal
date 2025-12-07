"""
Fetch Daily Wikipedia and Google Trends Data
Mimics the original contra.py behavior to fetch continuous daily sentiment signals

This fetches:
1. Wikipedia daily pageviews for company and CEO pages (2018-2020)
2. Google Trends data year-by-year for better granularity
3. Saves daily time series aligned to trading days
"""

import os
import pandas as pd
import requests
from urllib.parse import quote
from pytrends.request import TrendReq
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Companies to analyze
COMPANIES = {
    "Tesla": {
        "wiki_pages": ["Tesla,_Inc.", "Elon_Musk"],
        "trends_keywords": ["Tesla", "Elon Musk"]
    },
    "Amazon": {
        "wiki_pages": ["Amazon_(company)", "Jeff_Bezos"],
        "trends_keywords": ["Amazon", "Jeff Bezos"]
    },
    "Meta": {
        "wiki_pages": ["Meta_Platforms", "Mark_Zuckerberg"],
        "trends_keywords": ["Facebook", "Mark Zuckerberg"]  # Facebook brand in 2018-2020
    }
}

# Date range
START = "2018-01-01"
END = "2020-12-31"

# Input/Output
STOCK_PRICES_PATH = "../datasets/stock_prices.csv"
OUTPUT_DIR = "../datasets"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def load_trading_days():
    """Load trading days from stock prices"""
    print("Loading trading days...")
    prices = pd.read_csv(STOCK_PRICES_PATH, index_col=0, parse_dates=True)
    trading_days = prices.index
    print(f"  {len(trading_days)} trading days from {trading_days[0].date()} to {trading_days[-1].date()}")
    return trading_days

# =============================================================================
# WIKIPEDIA FETCHING
# =============================================================================

def fetch_wiki_daily(start, end, titles):
    """
    Fetch Wikipedia daily pageviews using Wikimedia REST API
    Returns DataFrame with daily pageview sentiment
    """
    print(f"\nFetching Wikipedia pageview data...")
    s = pd.to_datetime(start).strftime("%Y%m%d")
    e = pd.to_datetime(end).strftime("%Y%m%d")
    
    series = []
    for title in titles:
        print(f"  Fetching: {title}")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{quote(title)}/daily/{s}/{e}"
        
        try:
            headers = {'User-Agent': 'Some/1.0 (Educational Project)'}
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            js = r.json()
            rows = js.get("items", [])
            
            if not rows:
                print(f"    No data returned for {title}")
                continue
                
            df = pd.DataFrame({
                "date": [pd.to_datetime(x["timestamp"][:8]) for x in rows],
                f"pv_{title}": [x["views"] for x in rows]
            }).set_index("date")
            
            series.append(df)
            print(f"    Got {len(df)} daily pageviews")
            time.sleep(0.5)  # Rate limiting
            
        except Exception as ex:
            print(f"    Failed to fetch {title}: {ex}")
            continue
    
    if not series:
        print(f"    No Wikipedia data retrieved")
        return None
    
    pv = pd.concat(series, axis=1).asfreq("D").ffill()
    pv["sent_wiki_raw"] = pv.mean(axis=1)  # Average across pages
    
    print(f"  Wikipedia: {len(pv)} daily points from {pv.index.min().date()} to {pv.index.max().date()}")
    return pv[["sent_wiki_raw"]]

# =============================================================================
# GOOGLE TRENDS FETCHING
# =============================================================================

def fetch_trends_daily(start, end, keywords):
    """
    Fetch Google Trends data year-by-year for better granularity
    Returns DataFrame with daily trends sentiment
    """
    print(f"\nFetching Google Trends data...")
    pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.2)
    
    frames = []
    y0, y1 = pd.to_datetime(start).year, pd.to_datetime(end).year
    
    for year in range(y0, y1+1):
        s = f"{year}-01-01"
        e = f"{year}-12-31" if year < y1 else end
        print(f"  Fetching {year}...")
        
        try:
            pytrends.build_payload(keywords, timeframe=f"{s} {e}", geo="US")
            part = pytrends.interest_over_time()
            if "isPartial" in part.columns:
                part = part.drop(columns=["isPartial"])
            frames.append(part)
            print(f"    Got {len(part)} data points")
            time.sleep(2)  # Rate limiting
        except Exception as ex:
            print(f"    Failed to fetch {year}: {ex}")
            continue
    
    if not frames:
        print(f"    No Google Trends data retrieved")
        return None
    
    tr = pd.concat(frames).sort_index()
    tr.columns = [c.lower().replace(" ","_") for c in tr.columns]
    tr = tr.resample("D").mean().ffill()  # Convert to daily
    tr["sent_trends_raw"] = tr.mean(axis=1)  # Average across keywords
    
    print(f"  Google Trends: {len(tr)} daily points from {tr.index.min().date()} to {tr.index.max().date()}")
    return tr[["sent_trends_raw"]]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def fetch_company_sentiment(company, config, trading_days):
    """Fetch daily sentiment data for a company"""
    print(f"\n{'='*70}")
    print(f"FETCHING: {company}")
    print(f"{'='*70}")
    
    # Fetch Wikipedia pageviews
    wiki_data = fetch_wiki_daily(START, END, config["wiki_pages"])
    
    # Fetch Google Trends
    trends_data = fetch_trends_daily(START, END, config["trends_keywords"])
    
    # Combine data
    if wiki_data is None and trends_data is None:
        print(f"  Warning: No data retrieved for {company}")
        return None
    
    # Merge and align to trading days
    sentiment_data = pd.DataFrame(index=trading_days)
    
    if wiki_data is not None:
        sentiment_data = sentiment_data.join(wiki_data, how="left")
        sentiment_data["sent_wiki_raw"] = sentiment_data["sent_wiki_raw"].ffill().bfill()
    else:
        sentiment_data["sent_wiki_raw"] = 0
    
    if trends_data is not None:
        sentiment_data = sentiment_data.join(trends_data, how="left")
        sentiment_data["sent_trends_raw"] = sentiment_data["sent_trends_raw"].ffill().bfill()
    else:
        sentiment_data["sent_trends_raw"] = 0
    
    sentiment_data["company"] = company
    
    print(f"\n  Combined sentiment data: {len(sentiment_data)} trading days")
    return sentiment_data

def main():
    print("=" * 70)
    print("DAILY SENTIMENT DATA FETCHER")
    print("=" * 70)
    print("\nFetching continuous daily Wikipedia and Google Trends data")
    print("This mimics the original contra.py behavior")
    print(f"Period: {START} to {END}")
    print()
    
    # Load trading days
    trading_days = load_trading_days()
    
    # Fetch data for each company
    all_sentiment = []
    for company, config in COMPANIES.items():
        try:
            sentiment = fetch_company_sentiment(company, config, trading_days)
            if sentiment is not None:
                all_sentiment.append(sentiment)
        except Exception as e:
            print(f"\nError processing {company}: {e}")
            continue
    
    if not all_sentiment:
        print("\nNo sentiment data retrieved for any company")
        return
    
    # Combine all companies
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    combined = pd.concat(all_sentiment, ignore_index=False)
    
    # Save combined file
    ensure_dir(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, "sentiment_daily_raw.csv")
    combined.to_csv(output_path, index=True)
    print(f"\nSaved: {output_path}")
    print(f"  Total rows: {len(combined)}")
    
    # Save individual company files
    for company in COMPANIES.keys():
        company_data = combined[combined["company"] == company]
        company_path = os.path.join(OUTPUT_DIR, f"sentiment_daily_raw_{company.lower()}.csv")
        company_data.to_csv(company_path, index=True)
        print(f"Saved: {company_path} ({len(company_data)} rows)")
    
    # Summary
    print(f"\n{'='*70}")
    print("DAILY SENTIMENT DATA FETCHED!")
    print(f"{'='*70}")
    print("\nSummary by company:")
    for company in COMPANIES.keys():
        company_data = combined[combined["company"] == company]
        if len(company_data) > 0:
            print(f"\n{company}:")
            print(f"  Wiki mean: {company_data['sent_wiki_raw'].mean():.2f}")
            print(f"  Wiki std:  {company_data['sent_wiki_raw'].std():.2f}")
            print(f"  Trends mean: {company_data['sent_trends_raw'].mean():.2f}")
            print(f"  Trends std:  {company_data['sent_trends_raw'].std():.2f}")
    
    print("\nNext step:")
    print("  Run: python process_sentiment_data.py")
    print("  This will combine daily data with event kernel and z-score normalize")
    print()

if __name__ == "__main__":
    main()

