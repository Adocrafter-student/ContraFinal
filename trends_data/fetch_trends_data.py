"""
Fetch Wikipedia pageviews and Google Trends data for controversy events
Reads events from event_kernel.csv and outputs sentiment data CSV
"""

import os
import pandas as pd
import requests
from urllib.parse import quote
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import time

# Paths
INPUT_CSV = "../datasets/event_kernel.csv"
OUTPUT_DIR = "../datasets"

# Company to CEO mapping
CEO_MAPPING = {
    "Tesla": "Elon Musk",
    "Amazon": "Jeff Bezos",
    "Meta": "Mark Zuckerberg"
}

# Wikipedia page mappings
WIKI_PAGES = {
    "Tesla": ["Tesla,_Inc.", "Elon_Musk"],
    "Amazon": ["Amazon_(company)", "Jeff_Bezos"],
    "Meta": ["Meta_Platforms", "Mark_Zuckerberg"]
}

# Google Trends keyword mappings  
TRENDS_KEYWORDS = {
    "Tesla": ["Tesla", "Elon Musk"],
    "Amazon": ["Amazon", "Jeff Bezos"],
    "Meta": ["Facebook", "Mark Zuckerberg"]  # Facebook brand name in 2018-2020
}

def fetch_wiki_pageviews(page_title, date_str):
    """Fetch Wikipedia pageviews for a specific date"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_formatted = date_obj.strftime("%Y%m%d")
        
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{quote(page_title)}/daily/{date_formatted}/{date_formatted}"
        
        headers = {'User-Agent': 'CONTRA-Research/1.0 (Educational Project)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("items"):
                return data["items"][0]["views"]
        return None
    except Exception as e:
        print(f"Wiki error for {page_title} on {date_str}: {e}")
        return None

def fetch_trends_for_week(keywords, center_date):
    """Fetch Google Trends for a week around the event date"""
    try:
        date_obj = datetime.strptime(center_date, "%Y-%m-%d")
        start = (date_obj - timedelta(days=3)).strftime("%Y-%m-%d")
        end = (date_obj + timedelta(days=3)).strftime("%Y-%m-%d")
        
        pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.2)
        pytrends.build_payload(keywords, timeframe=f"{start} {end}", geo="US")
        
        df = pytrends.interest_over_time()
        
        if not df.empty and "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        
        if not df.empty:
            # Get value for the specific date
            date_value = df.loc[date_obj.strftime("%Y-%m-%d")] if date_obj.strftime("%Y-%m-%d") in df.index else None
            if date_value is not None:
                return date_value.mean()  # Average across keywords
        
        return None
    except Exception as e:
        print(f"Trends error for {keywords} on {center_date}: {e}")
        return None

def load_events():
    """Load events from CSV file"""
    print(f"Loading events from {INPUT_CSV}...")
    
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Event kernel CSV not found: {INPUT_CSV}")
    
    events_df = pd.read_csv(INPUT_CSV)
    events_df['date'] = pd.to_datetime(events_df['date']).dt.strftime('%Y-%m-%d')
    
    # Add CEO column based on company
    events_df['ceo'] = events_df['company'].map(CEO_MAPPING)
    
    print(f"Loaded {len(events_df)} events")
    return events_df

def main():
    print("=" * 80)
    print("FETCHING SENTIMENT DATA FOR EVENTS")
    print("=" * 80)
    
    # Load events from CSV
    events_df = load_events()
    
    print(f"Total events: {len(events_df)}")
    print(f"  Tesla: {sum(events_df['company'] == 'Tesla')}")
    print(f"  Amazon: {sum(events_df['company'] == 'Amazon')}")
    print(f"  Meta: {sum(events_df['company'] == 'Meta')}")
    print()
    
    results = []
    
    for i, (idx, event) in enumerate(events_df.iterrows(), 1):
        print(f"[{i}/{len(events_df)}] {event['date']} - {event['company']}: {event['event']}")
        
        company = event['company']
        wiki_pages = WIKI_PAGES[company]
        trends_keywords = TRENDS_KEYWORDS[company]
        
        # Fetch Wikipedia pageviews
        wiki_views = []
        for page in wiki_pages:
            views = fetch_wiki_pageviews(page, event['date'])
            if views is not None:
                wiki_views.append(views)
            time.sleep(0.2)  # Rate limiting
        
        avg_wiki_views = sum(wiki_views) / len(wiki_views) if wiki_views else None
        
        # Fetch Google Trends
        trends_value = fetch_trends_for_week(trends_keywords, event['date'])
        time.sleep(1)  # Rate limiting for Trends API
        
        results.append({
            'date': event['date'],
            'company': company,
            'ceo': event['ceo'],
            'event': event['event'],
            'event_type': event['event_type'],
            'wiki_pageviews': int(avg_wiki_views) if avg_wiki_views else '',
            'google_trends': round(float(trends_value), 1) if trends_value is not None else ''
        })
        
        print(f"  Wiki: {avg_wiki_views if avg_wiki_views else 'N/A'} | Trends: {trends_value if trends_value else 'N/A'}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV in datasets folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_filename = os.path.join(OUTPUT_DIR, 'sentiment_data_events.csv')
    df.to_csv(csv_filename, index=False)
    
    print()
    print("=" * 80)
    print("DATA SAVED!")
    print("=" * 80)
    print(f"CSV file: {csv_filename}")
    print(f"Total records: {len(df)}")
    print()
    print("Summary by company:")
    print(df.groupby('company').size())
    print()
    print("Sentiment data saved to datasets folder!")

if __name__ == "__main__":
    main()

