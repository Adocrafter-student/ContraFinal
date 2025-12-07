"""
CONTRA Sentiment Data Processing Pipeline
Performs z-score normalization and creates composite sentiment index

This script:
1. Loads raw sentiment data (Wikipedia, Google Trends, Event Kernel)
2. Performs z-score normalization on each source independently
3. Creates equal-weighted composite sentiment index
4. Saves normalized data aligned to trading days
"""

import os
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input paths
STOCK_PRICES_PATH = "../datasets/stock_prices.csv"
SENTIMENT_DAILY_PATH = "../datasets/sentiment_daily_raw_{}.csv"  # Daily wiki/trends data
EVENT_KERNEL_PATH = "../datasets/event_kernel.csv"

# Output directory
OUTPUT_DIR = "../datasets"

# Date range (same as financial data)
START = "2018-01-01"
END = "2020-12-31"

# Event-kernel shape: defines sentiment impact around events
# tau = days relative to event (0 = event day)
# Amplified magnitudes to create stronger signals
EVENT_KERNEL = {
    "positive": {-2: 1.5, -1: 3.0, 0: 5.0, +1: 3.0, +2: 1.5},
    "negative": {-2: -1.5, -1: -3.0, 0: -5.0, +1: -3.0, +2: -1.5}
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def z_score(series):
    """
    Calculate z-score normalization (mean=0, std=1)
    """
    mean = series.mean()
    std = series.std()
    if std > 0:
        return (series - mean) / std
    return series - mean

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_trading_days():
    """Load trading days index from stock prices"""
    print(" Loading trading days...")
    prices = pd.read_csv(STOCK_PRICES_PATH, index_col=0, parse_dates=True)
    trading_days = prices.index
    print(f"  {len(trading_days)} trading days from {trading_days[0].date()} to {trading_days[-1].date()}")
    return trading_days

def load_daily_sentiment(company, trading_days):
    """Load daily sentiment data for a company"""
    print(f"\n📊 Loading daily sentiment data for {company}...")
    
    sentiment_path = SENTIMENT_DAILY_PATH.format(company.lower())
    if not os.path.exists(sentiment_path):
        print(f"    ⚠ Daily sentiment file not found: {sentiment_path}")
        print(f"    ⚠ Please run fetch_daily_sentiment.py first!")
        return None, None
    
    df = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
    
    # Align to trading days
    wiki_raw = df["sent_wiki_raw"].reindex(trading_days).ffill().bfill()
    trends_raw = df["sent_trends_raw"].reindex(trading_days).ffill().bfill()
    
    print(f"  Loaded {len(df)} days of daily data")
    print(f"  Aligned to {len(trading_days)} trading days")
    
    return wiki_raw, trends_raw

def load_event_kernel():
    """Load event kernel data"""
    print("\nLoading event kernel...")
    df = pd.read_csv(EVENT_KERNEL_PATH, parse_dates=['date'])
    print(f"  Loaded {len(df)} events")
    return df

# =============================================================================
# SENTIMENT INDEX BUILDERS
# =============================================================================

def build_event_kernel_index(events_df, trading_days, kernel):
    """
    Build event-kernel sentiment index from curated events
    
    For each event, applies a kernel weight pattern around the event date
    (e.g., day -2, -1, 0, +1, +2) based on event type (positive/negative)
    """
    print("\n Building event kernel index...")
    
    # Create time series index
    kernel_series = pd.Series(0.0, index=trading_days, name='sent_event_raw')
    
    event_count = {"positive": 0, "negative": 0}
    
    for _, event in events_df.iterrows():
        # Find nearest trading day
        event_date = pd.to_datetime(event['date'])
        if event_date not in trading_days:
            nearest_idx = trading_days.get_indexer([event_date], method='nearest')[0]
            event_date = trading_days[nearest_idx]
        
        event_type = event['event_type'].lower().strip()
        if event_type not in kernel:
            continue
        
        # Apply kernel weights around event
        for tau, weight in kernel[event_type].items():
            target_date = event_date + pd.tseries.offsets.BDay(tau)
            if target_date in kernel_series.index:
                kernel_series.loc[target_date] += weight
        
        event_count[event_type] += 1
    
    print(f"  Applied kernel to {event_count['positive']} positive and {event_count['negative']} negative events")
    print(f"  Non-zero days: {(kernel_series != 0).sum()}")
    
    return kernel_series

# =============================================================================
# Z-SCORE NORMALIZATION
# =============================================================================

def normalize_sentiment_sources(wiki_raw, trends_raw, event_raw):
    """
    Perform z-score normalization on each sentiment source independently
    """
    print("\n Performing z-score normalization...")
    
    # Z-score each source
    wiki_z = z_score(wiki_raw)
    trends_z = z_score(trends_raw)
    event_z = z_score(event_raw)
    
    print(f"  Wikipedia z-scored: mean={wiki_z.mean():.6f}, std={wiki_z.std():.6f}")
    print(f"  Google Trends z-scored: mean={trends_z.mean():.6f}, std={trends_z.std():.6f}")
    print(f"  Event Kernel z-scored: mean={event_z.mean():.6f}, std={event_z.std():.6f}")
    
    return wiki_z, trends_z, event_z

def create_composite_index(wiki_z, trends_z, event_z):
    """
    Create weighted composite sentiment index
    Event kernel gets more weight since it contains curated controversy signals
    """
    print("\nCreating composite sentiment index...")
    
    # Combine into DataFrame
    sentiment_df = pd.DataFrame({
        'sent_wiki_z': wiki_z,
        'sent_trends_z': trends_z,
        'sent_event_z': event_z
    })
    
    # WEIGHTED average: Event kernel gets 60%, Wiki 20%, Trends 20%
    # This prioritizes curated event signals over smooth daily data
    sentiment_df['sent_composite_raw'] = (
        0.20 * sentiment_df['sent_wiki_z'] + 
        0.20 * sentiment_df['sent_trends_z'] + 
        0.60 * sentiment_df['sent_event_z']
    )
    
    # Z-score the composite
    sentiment_df['sent_composite_z'] = z_score(sentiment_df['sent_composite_raw'])
    
    print(f"  Composite index created (Event 60%, Wiki 20%, Trends 20%)")
    print(f"  Composite mean={sentiment_df['sent_composite_z'].mean():.6f}, std={sentiment_df['sent_composite_z'].std():.6f}")
    
    return sentiment_df

# =============================================================================
# COMPANY-SPECIFIC PROCESSING
# =============================================================================

def process_company_sentiment(company, event_kernel_df, trading_days):
    """
    Process sentiment data for a single company
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {company}")
    print(f"{'='*70}")
    
    # Load daily sentiment data (Wikipedia & Google Trends)
    wiki_raw, trends_raw = load_daily_sentiment(company, trading_days)
    
    if wiki_raw is None or trends_raw is None:
        print(f"  ⚠ Could not load daily sentiment for {company}")
        return None
    
    # Filter event kernel for this company
    company_kernel = event_kernel_df[event_kernel_df['company'] == company].copy()
    print(f"  Event kernel: {len(company_kernel)} events")
    
    # Build event kernel index
    event_raw = build_event_kernel_index(company_kernel, trading_days, EVENT_KERNEL)
    
    # Z-score normalization
    wiki_z, trends_z, event_z = normalize_sentiment_sources(wiki_raw, trends_raw, event_raw)
    
    # Create composite index
    sentiment_df = create_composite_index(wiki_z, trends_z, event_z)
    
    # Add metadata
    sentiment_df['company'] = company
    sentiment_df['date'] = sentiment_df.index
    
    return sentiment_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("CONTRA SENTIMENT DATA PROCESSING PIPELINE")
    print("=" * 70)
    print("\nThis script performs:")
    print("  1. Load daily sentiment data (Wikipedia, Google Trends)")
    print("  2. Load event kernel data")
    print("  3. Z-score normalization (mean=0, std=1) for each source")
    print("  4. Equal-weighted composite sentiment index creation")
    print("  5. Save processed data aligned to trading days")
    
    # Load data
    trading_days = load_trading_days()
    event_kernel = load_event_kernel()
    
    # Define companies
    companies = ["Tesla", "Amazon", "Meta"]
    print(f"\n📋 Processing {len(companies)} companies: {', '.join(companies)}")
    
    # Process each company
    all_sentiment = []
    for company in companies:
        company_sentiment = process_company_sentiment(
            company, 
            event_kernel, 
            trading_days
        )
        if company_sentiment is not None:
            all_sentiment.append(company_sentiment)
    
    if not all_sentiment:
        print("\nNo sentiment data processed!")
        print("Please run fetch_daily_sentiment.py first to get daily data")
        return
    
    # Combine all companies
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    combined_sentiment = pd.concat(all_sentiment, ignore_index=False)
    
    # Save to CSV
    ensure_dir(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, "sentiment_normalized.csv")
    combined_sentiment.to_csv(output_path, index=True)
    print(f"\n Saved: {output_path}")
    print(f"  Total rows: {len(combined_sentiment)}")
    print(f"  Date range: {combined_sentiment.index[0].date()} to {combined_sentiment.index[-1].date()}")
    
    # Save individual company files
    for company in companies:
        company_data = combined_sentiment[combined_sentiment['company'] == company]
        company_path = os.path.join(OUTPUT_DIR, f"sentiment_normalized_{company.lower()}.csv")
        company_data.to_csv(company_path, index=True)
        print(f" Saved: {company_path} ({len(company_data)} rows)")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    print("\nComposite Sentiment by Company:")
    for company in companies:
        company_data = combined_sentiment[combined_sentiment['company'] == company]
        print(f"\n{company}:")
        print(f"  Mean: {company_data['sent_composite_z'].mean():.6f}")
        print(f"  Std:  {company_data['sent_composite_z'].std():.6f}")
        print(f"  Min:  {company_data['sent_composite_z'].min():.6f}")
        print(f"  Max:  {company_data['sent_composite_z'].max():.6f}")
    
    print(f"\n{'='*70}")
    print("SENTIMENT DATA PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print("\nNext Steps:")
    print("  Run: python contra.py (in root directory)")
    print("\nData created:")
    print("  sentiment_normalized.csv - All companies combined")
    print("  sentiment_normalized_[company].csv - Individual files")
    print("  All indices are z-scored (mean≈0, std≈1)")
    print("  Composite = equal-weighted average of Wiki + Trends + Events")
    print()

if __name__ == "__main__":
    main()

