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
SENTIMENT_EVENTS_PATH = "../datasets/sentiment_data_events.csv"
EVENT_KERNEL_PATH = "../datasets/event_kernel.csv"

# Output directory
OUTPUT_DIR = "../datasets"

# Date range (same as financial data)
START = "2018-01-01"
END = "2020-12-31"

# Event-kernel shape: defines sentiment impact around events
# tau = days relative to event (0 = event day)
EVENT_KERNEL = {
    "positive": {-2: 0.5, -1: 1.0, 0: 1.5, +1: 1.0, +2: 0.5},
    "negative": {-2: -0.5, -1: -1.0, 0: -1.5, +1: -1.0, +2: -0.5}
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

def load_sentiment_events():
    """Load sentiment data for specific events"""
    print("\n Loading sentiment events data...")
    df = pd.read_csv(SENTIMENT_EVENTS_PATH, parse_dates=['date'])
    print(f"  Loaded {len(df)} events with Wikipedia and Google Trends data")
    return df

def load_event_kernel():
    """Load event kernel data"""
    print("\n⚡ Loading event kernel...")
    df = pd.read_csv(EVENT_KERNEL_PATH, parse_dates=['date'])
    print(f"  Loaded {len(df)} events")
    return df

# =============================================================================
# SENTIMENT INDEX BUILDERS
# =============================================================================

def build_wiki_sentiment_index(events_df, trading_days):
    """
    Build Wikipedia pageview sentiment index
    Creates daily time series from event-level data
    """
    print("\n Building Wikipedia sentiment index...")
    
    # Create time series index
    wiki_series = pd.Series(0.0, index=trading_days, name='sent_wiki_raw')
    
    # For each event, fill surrounding days with pageview signal
    for _, event in events_df.iterrows():
        event_date = pd.to_datetime(event['date'])
        
        # Find nearest trading day
        if event_date not in trading_days:
            nearest_idx = trading_days.get_indexer([event_date], method='nearest')[0]
            event_date = trading_days[nearest_idx]
        
        # Apply pageview value to event day and surrounding days (decay)
        if event_date in wiki_series.index:
            views = event['wiki_pageviews']
            wiki_series.loc[event_date] += views
            
            # Add decay to adjacent days
            for offset in [-1, 1]:
                adj_date = event_date + pd.tseries.offsets.BDay(offset)
                if adj_date in wiki_series.index:
                    wiki_series.loc[adj_date] += views * 0.5
    
    # Forward fill for non-event days (baseline)
    wiki_series = wiki_series.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
    
    print(f"  ✓ Created Wikipedia index with {(wiki_series > 0).sum()} non-zero days")
    return wiki_series

def build_trends_sentiment_index(events_df, trading_days):
    """
    Build Google Trends sentiment index
    Creates daily time series from event-level data
    """
    print("\n Building Google Trends sentiment index...")
    
    # Create time series index
    trends_series = pd.Series(0.0, index=trading_days, name='sent_trends_raw')
    
    # For each event, fill surrounding days with trends signal
    for _, event in events_df.iterrows():
        event_date = pd.to_datetime(event['date'])
        
        # Find nearest trading day
        if event_date not in trading_days:
            nearest_idx = trading_days.get_indexer([event_date], method='nearest')[0]
            event_date = trading_days[nearest_idx]
        
        # Apply trends value to event day and surrounding days (decay)
        if event_date in trends_series.index:
            trend_val = event['google_trends']
            trends_series.loc[event_date] += trend_val
            
            # Add decay to adjacent days
            for offset in [-1, 1]:
                adj_date = event_date + pd.tseries.offsets.BDay(offset)
                if adj_date in trends_series.index:
                    trends_series.loc[adj_date] += trend_val * 0.5
    
    # Forward fill for non-event days (baseline)
    trends_series = trends_series.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
    
    print(f"  ✓ Created Google Trends index with {(trends_series > 0).sum()} non-zero days")
    return trends_series

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
    
    print(f"  ✓ Applied kernel to {event_count['positive']} positive and {event_count['negative']} negative events")
    print(f"  ✓ Non-zero days: {(kernel_series != 0).sum()}")
    
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
    
    print(f"  ✓ Wikipedia z-scored: mean={wiki_z.mean():.6f}, std={wiki_z.std():.6f}")
    print(f"  ✓ Google Trends z-scored: mean={trends_z.mean():.6f}, std={trends_z.std():.6f}")
    print(f"  ✓ Event Kernel z-scored: mean={event_z.mean():.6f}, std={event_z.std():.6f}")
    
    return wiki_z, trends_z, event_z

def create_composite_index(wiki_z, trends_z, event_z):
    """
    Create equal-weighted composite sentiment index
    """
    print("\n Creating composite sentiment index...")
    
    # Combine into DataFrame
    sentiment_df = pd.DataFrame({
        'sent_wiki_z': wiki_z,
        'sent_trends_z': trends_z,
        'sent_event_z': event_z
    })
    
    # Equal-weighted average
    sentiment_df['sent_composite_raw'] = sentiment_df[['sent_wiki_z', 'sent_trends_z', 'sent_event_z']].mean(axis=1)
    
    # Z-score the composite
    sentiment_df['sent_composite_z'] = z_score(sentiment_df['sent_composite_raw'])
    
    print(f"  ✓ Composite index created")
    print(f"  ✓ Composite mean={sentiment_df['sent_composite_z'].mean():.6f}, std={sentiment_df['sent_composite_z'].std():.6f}")
    
    return sentiment_df

# =============================================================================
# COMPANY-SPECIFIC PROCESSING
# =============================================================================

def process_company_sentiment(company, events_df, event_kernel_df, trading_days):
    """
    Process sentiment data for a single company
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {company}")
    print(f"{'='*70}")
    
    # Filter events for this company
    company_events = events_df[events_df['company'] == company].copy()
    company_kernel = event_kernel_df[event_kernel_df['company'] == company].copy()
    
    print(f"Events: {len(company_events)} with sentiment data, {len(company_kernel)} in kernel")
    
    # Build raw sentiment indices
    wiki_raw = build_wiki_sentiment_index(company_events, trading_days)
    trends_raw = build_trends_sentiment_index(company_events, trading_days)
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
    print("  1. Load raw sentiment data (Wikipedia, Google Trends, Event Kernel)")
    print("  2. Z-score normalization (mean=0, std=1) for each source")
    print("  3. Equal-weighted composite sentiment index creation")
    print("  4. Save processed data aligned to trading days")
    
    # Load data
    trading_days = load_trading_days()
    sentiment_events = load_sentiment_events()
    event_kernel = load_event_kernel()
    
    # Get unique companies
    companies = sentiment_events['company'].unique()
    print(f"\n Processing {len(companies)} companies: {', '.join(companies)}")
    
    # Process each company
    all_sentiment = []
    for company in companies:
        company_sentiment = process_company_sentiment(
            company, 
            sentiment_events, 
            event_kernel, 
            trading_days
        )
        all_sentiment.append(company_sentiment)
    
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
    print(" SENTIMENT DATA PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print("\n Next Steps:")
    print("  • Use sentiment_normalized.csv for CONTRA analysis")
    print("  • Each company has individual sentiment file")
    print("  • All indices are z-scored (mean≈0, std≈1)")
    print("  • Composite index is equal-weighted average of all sources")
    print()

if __name__ == "__main__":
    main()

