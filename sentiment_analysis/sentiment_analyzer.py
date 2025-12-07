import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

def parse_news_file(input_file):
    """
    Parse the text-based news file format and extract articles.
    Returns a list of dictionaries with 'date', 'company', 'event', and 'text' keys.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by article delimiter
    articles = content.split('=== ARTICLE ===')
    
    parsed_articles = []
    for article in articles:
        article = article.strip()
        if not article:
            continue
        
        # Initialize fields
        date = None
        company = None
        event = None
        text = None
        title = None
        
        # Split into lines
        lines = article.split('\n')
        
        # Parse metadata fields
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('date:'):
                date = line.replace('date:', '').strip()
            elif line.startswith('company:'):
                company = line.replace('company:', '').strip()
            elif line.startswith('event:'):
                event = line.replace('event:', '').strip()
            elif line.startswith('title:'):
                title = line.replace('title:', '').strip()
            elif line.startswith('text:'):
                # Text starts after the "text:" line and continues until the end
                text_lines = []
                i += 1
                while i < len(lines):
                    text_lines.append(lines[i])
                    i += 1
                text = '\n'.join(text_lines).strip()
                break
            
            i += 1
        
        # Add article if we have required fields
        if date and company and text:
            parsed_articles.append({
                'date': date,
                'company': company,
                'event': event or '',
                'text': text,
                'title': title or ''
            })
    
    return parsed_articles

def analyze_sentiment(input_file="news.txt", output_file="event_kernel.csv"):
    """
    Reads news data from text file, performs sentiment analysis, and generates an events CSV.
    """
    print(f"Reading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    try:
        articles = parse_news_file(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not articles:
        print("Error: No articles found in the input file.")
        return

    print(f"Found {len(articles)} articles.")
    print("Initializing VADER sentiment analyzer...")
    sia = SentimentIntensityAnalyzer()

    events = []

    print("Analyzing news items...")
    for article in articles:
        text = article["text"]
        date = article["date"]
        company = article["company"]
        event = article["event"]
        title = article["title"]

        analysis_text = f"{event}. {title}"
        scores = sia.polarity_scores(analysis_text)
        compound = scores["compound"]
        
        # Apply keyword-based adjustments for common financial news terms
        text_lower = analysis_text.lower()
        
        # Negative indicators
        negative_keywords = ['lawsuit', 'investigation', 'scandal', 'crash', 'drop', 'controversy', 
                           'contempt', 'extortion', 'divorce', 'cancellation', 'outage', 
                           'antitrust', 'testimony', 'hearing', 'filed', 'sued', 'fine', 
                           'falls', 'plunge', 'too high']
        
        # Positive indicators  
        positive_keywords = ['beat', 'profit', 'secured', 'settlement', 'hq2 announcement',
                           'inclusion', 'surprise', 'crosses', 'earnings beat', 's&p 500',
                           'investor day', 'autonomy day', 'highlights']
        
        # Check for keyword overrides
        has_negative = any(keyword in text_lower for keyword in negative_keywords)
        has_positive = any(keyword in text_lower for keyword in positive_keywords)
        
        # Adjust compound score based on keywords
        if has_negative and not has_positive:
            compound = compound - 0.3  # Push negative
        elif has_positive and not has_negative:
            compound = compound + 0.3  # Push positive
        
        event_type = "neutral"
        if compound > 0.05:
            event_type = "positive"
        elif compound < -0.05:
            event_type = "negative"
        
        # We only care about positive or negative events for the model
        if event_type != "neutral":
            events.append({
                "date": date,
                "company": company,
                "event": event,
                "event_type": event_type
            })

    if not events:
        print("No significant events found.")
        return

    events_df = pd.DataFrame(events)
    
    
    print(f"Saving {len(events_df)} events to {output_file}...")
    events_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    analyze_sentiment()
