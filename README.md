# CONTRA:

**CO**ntroversy-Augmented **N**atural-languague **TR**ading **A**lgorithm

A sentiment-augmented stock prediction framework that quantifies how CEO-driven controversies and public attention impact stock returns beyond traditional market factors.

---

## 📊 Overview

CONTRA tests the hypothesis that CEO-centric public sentiment—captured through Wikipedia pageviews, Google Trends, and curated news events explains stock return variation beyond traditional asset pricing models. By constructing a weighted composite sentiment index and estimating a "sentiment-alpha" coefficient (θ), the model measures how controversy-driven behavioral factors generate short-term price movements.

---

## 🎯 Features

- **Multi-Source Sentiment Index**: Combines Wikipedia pageviews (20%), Google Trends (20%), and event kernel (60%)
- **Event Kernel System**: Amplified temporal decay patterns around curated controversy events (±5.0 peak weight)
- **Statistical Validation**: OLS regression with HC1 robust standard errors, Granger causality tests, event study analysis
- **Interactive Visualizations**: Plotly-based prediction comparisons, sentiment overlays, and event markers
- **Multi-Company Analysis**: Tesla, Amazon, and Meta (2018-2020 period)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Internet connection for data fetching

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Adocrafter-student/ContraFinal.git
cd ContraFinal
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=2.2
numpy>=1.26
yfinance>=0.2.40
statsmodels>=0.14
scikit-learn>=1.4
pytrends>=4.9.2
matplotlib>=3.8
plotly>=5.22
requests>=2.31
pyarrow>=14.0.0
```

---

## 📁 Project Structure

```
ContraFinal/
├── datasets/                      # Processed data storage
│   ├── event_kernel.csv          # Curated CEO events (28 events)
│   ├── sentiment_data_events.csv # Wikipedia/Trends for events
│   ├── sentiment_daily_raw_*.csv # Daily sentiment time series
│   ├── sentiment_normalized_*.csv# Z-scored composite indices
│   ├── stock_prices.csv          # Historical price data
│   └── stock_returns.csv         # Daily return calculations
│
├── financial_data/
│   ├── fetch_financial_data.py   # Download stock data (yfinance)
│   └── contra.py                 # Original reference implementation
│
├── trends_data/
│   ├── fetch_daily_sentiment.py  # Daily Wikipedia/Trends fetcher
│   ├── process_sentiment_data.py # Z-score normalization & compositing
│   └── fetch_trends_data.py      # Event-level sentiment fetcher
│
├── sentiment_analysis/
│   ├── sentiment_analyzer.py     # VADER sentiment analysis
│   ├── word_frequency_analyzer.py# Corpus validation
│   └── wordcloud.png             # Visual word frequency
│
├── results/                       # Model outputs
│   ├── summary_*.txt             # Performance metrics per company
│   ├── regression_*.txt          # Full OLS regression output
│   ├── predictions_*.html        # Interactive prediction plots
│   ├── sentiment_overlay_*.html  # Sentiment correlation plots
│   └── granger_*.csv             # Causality test results
│
├── contra.py                      # Main CONTRA analysis script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔄 Usage Workflow

### Step 1: Fetch Financial Data

Download historical stock prices and calculate returns for Tesla, Amazon, and Meta:

```bash
cd financial_data
python fetch_financial_data.py
```

**Output**: `datasets/stock_prices.csv`, `datasets/stock_returns.csv`

### Step 2: Fetch Daily Sentiment Data

Retrieve continuous daily Wikipedia pageviews and Google Trends (2018-2020):

```bash
cd trends_data
python fetch_daily_sentiment.py
```

**Output**: `datasets/sentiment_daily_raw_*.csv` (one per company)

**Note**: This step may take 10-15 minutes due to API rate limiting. The script fetches year-by-year to preserve daily resolution.

### Step 3: Process & Normalize Sentiment

Combine daily sentiment with event kernel and create weighted composite:

```bash
python process_sentiment_data.py
```

**Output**: `datasets/sentiment_normalized_*.csv` with z-scored indices

**Key Parameters**:
- Event kernel weights: {-2: ±1.5, -1: ±3.0, 0: ±5.0, +1: ±3.0, +2: ±1.5}
- Composite weighting: 60% events, 20% Wikipedia, 20% Trends

### Step 4: Run CONTRA Analysis

Train baseline and sentiment-augmented models for all companies:

```bash
cd ..
python contra.py
```

**Output**: 
- `results/summary_*.txt` - Performance metrics
- `results/regression_*.txt` - Full regression output
- `results/*.html` - Interactive Plotly visualizations
- `results/granger_*.csv` - Causality test results

### Step 5: View Results

Open the HTML files in `results/` directory with your browser:
- `predictions_tesla.html` - Actual vs predicted returns with event markers
- `sentiment_overlay_tesla.html` - Returns vs sentiment dual-axis plot
- `sentiment_components_tesla.html` - Individual sentiment source breakdown
- `comparison_*.html` - Cross-company performance comparison

---

## 🔬 Methodology

### Regression Model

**Baseline**: R(i,t) = α + β R(m,t) + ε(i,t)  
**CONTRA**: R(i,t) = α + β R(m,t) + θ S(t) + ε(i,t)

Where:
- R(i,t) = Company daily return
- R(m,t) = S&P 500 market return
- S(t) = Weighted composite sentiment index (z-scored)
- θ = Sentiment-alpha coefficient
- ε(i,t) = Residual error term

### Composite Sentiment Index

```
S(t) = 0.60 × EventKernel(t) + 0.20 × Wikipedia(t) + 0.20 × Trends(t)
```

Each component is independently z-score normalized before weighting.

### Event Kernel Pattern

For each curated event (positive/negative):
```
Day relative to event:  -2    -1     0    +1    +2
Weight:                ±1.5  ±3.0  ±5.0  ±3.0  ±1.5
```

Amplified magnitudes (3.3x versus preliminary PoC) ensure sentiment signals match empirically observed controversy-driven return volatility (~5-10%).

---

## 📊 Event Dataset

The `datasets/event_kernel.csv` file contains 28 curated CEO-centric events:
- **Tesla**: 13 events (7 positive, 6 negative)
- **Amazon**: 7 events (2 positive, 5 negative)
- **Meta**: 8 events (0 positive, 8 negative)

Event types include:
- SEC investigations & settlements
- Congressional testimonies
- Viral social media incidents
- Earnings surprises
- Major corporate announcements

---

## 🛠️ Configuration

Key parameters can be adjusted in respective files:

**Event Kernel Weights** (`trends_data/process_sentiment_data.py`):
```python
EVENT_KERNEL = {
    "positive": {-2: 1.5, -1: 3.0, 0: 5.0, +1: 3.0, +2: 1.5},
    "negative": {-2: -1.5, -1: -3.0, 0: -5.0, +1: -3.0, +2: -1.5}
}
```

**Composite Weighting** (`trends_data/process_sentiment_data.py`):
```python
sentiment_df['sent_composite_raw'] = (
    0.60 * sentiment_df['sent_event_z'] + 
    0.20 * sentiment_df['sent_wiki_z'] + 
    0.20 * sentiment_df['sent_trends_z']
)
```

**Train/Test Split** (`contra.py`):
```python
TRAIN_RATIO = 0.7  # 70% training, 30% testing
```

---

## 🔍 Validation Methods

1. **Statistical Significance**: OLS with HC1 heteroskedasticity-robust standard errors
2. **Out-of-Sample Testing**: 70/30 chronological train-test split
3. **Granger Causality**: Tests whether sentiment predicts future returns (lags 1-5)
4. **Event Study Analysis**: Cumulative abnormal returns (CAR) around event windows
5. **Cross-Company Validation**: Consistent methodology applied to 3 firms

---

## ⚠️ Limitations & Caveats

1. **Data Period**: Limited to 2018-2020 (754 trading days)
2. **Company Selection**: Results may not generalize beyond high-profile tech CEOs
3. **Event Curation**: Manual event selection introduces potential researcher bias
4. **Rate Limits**: Google Trends API has usage restrictions; daily fetching recommended
5. **Causality**: Statistical significance ≠ causal proof; sentiment may reflect rather than predict returns

---

## 📚 Dependencies & APIs

- **yfinance**: Historical stock data (Yahoo Finance)
- **Wikimedia REST API**: Daily pageview statistics
- **pytrends**: Unofficial Google Trends API wrapper
- **VADER**: Sentiment analysis lexicon (not used in production pipeline)
- **statsmodels**: Regression analysis with robust standard errors
- **plotly**: Interactive HTML visualizations

---

## 🤝 Contributing

This project was developed as an academic research implementation. Key areas for extension:

1. **Longer Time Periods**: Extend beyond 2018-2020 for robustness
2. **More balanced and richer sentiment dataset**: Extend beyond 8 events for Meta, needed more positive events
3. **More Companies**: Test across broader firm universe
4. **Real-Time Pipeline**: Automate daily sentiment fetching
5. **Alternative Sources**: Incorporate Twitter/Reddit sentiment
6. **Machine Learning**: Test LSTM/transformer models versus OLS

---


## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: December 2025

