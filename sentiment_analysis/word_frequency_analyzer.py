import re
import os
from collections import Counter
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

from nltk.corpus import stopwords

def parse_news_file(input_file):
    """
    Parse the text-based news file format and extract all article text.
    Returns a concatenated string of all article text.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by article delimiter
    articles = content.split('=== ARTICLE ===')
    
    all_text = []
    for article in articles:
        article = article.strip()
        if not article:
            continue
        
        # Initialize fields
        text = None
        
        # Split into lines
        lines = article.split('\n')
        
        # Parse to find text content
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('text:'):
                # Text starts after the "text:" line and continues until the end
                text_lines = []
                i += 1
                while i < len(lines):
                    text_lines.append(lines[i])
                    i += 1
                text = '\n'.join(text_lines).strip()
                break
            
            i += 1
        
        # Add text if found
        if text:
            all_text.append(text)
    
    return ' '.join(all_text)

def clean_text(text):
    """
    Clean and preprocess text for word frequency analysis.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_word_frequencies(text, min_length=3, top_n=100):
    """
    Extract word frequencies from text, excluding stopwords.
    """
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords for financial/news context
    custom_stopwords = {'said', 'would', 'could', 'also', 'one', 'two', 'three', 
                       'first', 'last', 'year', 'years', 'month', 'months', 
                       'day', 'days', 'time', 'times', 'new', 'old', 'may',
                       'might', 'must', 'shall', 'will', 'can', 'get', 'got',
                       'make', 'made', 'take', 'took', 'see', 'saw', 'know',
                       'knew', 'think', 'thought', 'say', 'tell', 'told',
                       'come', 'came', 'go', 'went', 'use', 'used', 'way',
                       'ways', 'much', 'many', 'more', 'most', 'very', 'well',
                       'good', 'great', 'little', 'long', 'small', 'large',
                       'big', 'high', 'low', 'right', 'left', 'back', 'front',
                       'top', 'bottom', 'next', 'previous', 'last', 'first'}
    stop_words.update(custom_stopwords)
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Split into words
    words = cleaned_text.split()
    
    # Filter words: remove stopwords, short words, and keep only alphabetic
    filtered_words = [
        word for word in words 
        if len(word) >= min_length 
        and word.isalpha() 
        and word not in stop_words
    ]
    
    # Count frequencies
    word_freq = Counter(filtered_words)
    
    # Return top N words
    return word_freq.most_common(top_n)

def save_word_frequencies(word_freq, output_file="word_frequencies.txt"):
    """
    Save word frequencies to a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Word Frequency Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total unique words: {len(word_freq)}\n")
        f.write(f"Total word count: {sum(count for _, count in word_freq)}\n\n")
        f.write("Top Words (by frequency):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Rank':<8} {'Word':<20} {'Frequency':<12} {'Percentage':<12}\n")
        f.write("-" * 50 + "\n")
        
        total_count = sum(count for _, count in word_freq)
        for rank, (word, count) in enumerate(word_freq, 1):
            percentage = (count / total_count) * 100
            f.write(f"{rank:<8} {word:<20} {count:<12} {percentage:.2f}%\n")

def generate_wordcloud(word_freq, output_file="wordcloud.png", width=1200, height=800):
    """
    Generate a word cloud image from word frequencies.
    """
    # Convert to dictionary format for WordCloud
    word_dict = dict(word_freq)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=100,
        colormap='viridis',  # Green/brown color scheme similar to the example
        relative_scaling=0.5,
        random_state=42
    ).generate_from_frequencies(word_dict)
    
    # Create figure
    plt.figure(figsize=(width/100, height/100), facecolor='white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save the image
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Word cloud saved to {output_file}")

def analyze_word_frequency(input_file="news.txt", 
                          freq_output="word_frequencies.txt",
                          cloud_output="wordcloud.png",
                          top_n=100):
    """
    Main function to analyze word frequencies and generate outputs.
    """
    print(f"Reading news articles from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    try:
        # Parse and extract all text
        all_text = parse_news_file(input_file)
        
        if not all_text:
            print("Error: No text found in articles.")
            return
        
        print(f"Extracted {len(all_text)} characters of text.")
        print("Analyzing word frequencies...")
        
        # Get word frequencies
        word_freq = get_word_frequencies(all_text, top_n=top_n)
        
        if not word_freq:
            print("Error: No words found after processing.")
            return
        
        print(f"Found {len(word_freq)} unique words.")
        
        # Save word frequencies to file
        print(f"Saving word frequencies to {freq_output}...")
        save_word_frequencies(word_freq, freq_output)
        print(f"Word frequencies saved to {freq_output}")
        
        # Generate word cloud
        print(f"Generating word cloud image...")
        generate_wordcloud(word_freq, cloud_output)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_word_frequency()
