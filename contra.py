"""
CONTRA: Sentiment-Alpha Stock Prediction Model
Production implementation using pre-processed sentiment and financial data

This script performs complete CONTRA analysis for Tesla, Amazon, and Meta:
1. Loads pre-processed financial data and normalized sentiment indices
2. Trains baseline (market-only) and CONTRA (market + sentiment) models
3. Evaluates predictive performance and statistical significance
4. Generates comprehensive visualizations and reports
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =============================================================================
# CONFIGURATION
# =============================================================================

# Company ticker mappings
COMPANIES = {
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Meta": "META"
}

# Market benchmark
MARKET = "^GSPC"

# Data paths
STOCK_RETURNS_PATH = "datasets/stock_returns.csv"
STOCK_PRICES_PATH = "datasets/stock_prices.csv"
SENTIMENT_BASE_PATH = "datasets/sentiment_normalized_{}.csv"
EVENT_KERNEL_PATH = "datasets/event_kernel.csv"

# Output directory
OUTPUT_DIR = "results"

# Train/test split ratio
TRAIN_RATIO = 0.7

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def save_html(fig, fname):
    """Save Plotly figure as HTML"""
    ensure_dir(OUTPUT_DIR)
    filepath = os.path.join(OUTPUT_DIR, fname)
    fig.write_html(filepath, include_plotlyjs="cdn")
    return filepath

def save_text(content, fname):
    """Save text content to file"""
    ensure_dir(OUTPUT_DIR)
    filepath = os.path.join(OUTPUT_DIR, fname)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath

# =============================================================================
# DATA LOADING
# =============================================================================

def load_stock_data(ticker):
    """Load stock returns and prices for a ticker"""
    print(f"  Loading stock data for {ticker}...")
    
    # Load returns
    returns = pd.read_csv(STOCK_RETURNS_PATH, index_col=0, parse_dates=True)
    returns = returns[[ticker, MARKET]].copy()
    returns.columns = ["ret_stk", "ret_mkt"]
    
    # Load prices
    prices = pd.read_csv(STOCK_PRICES_PATH, index_col=0, parse_dates=True)
    prices = prices[[ticker, MARKET]].copy()
    
    print(f"    ✓ {len(returns)} trading days of returns")
    return returns, prices

def load_sentiment_data(company):
    """Load normalized sentiment data for a company"""
    print(f"  Loading sentiment data for {company}...")
    
    sentiment_path = SENTIMENT_BASE_PATH.format(company.lower())
    sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
    
    # Select relevant columns
    sent_cols = ["sent_wiki_z", "sent_trends_z", "sent_event_z", "sent_composite_z"]
    sentiment = sentiment[sent_cols].copy()
    
    print(f"    ✓ {len(sentiment)} days of sentiment data")
    return sentiment

def load_events(company=None):
    """Load event kernel data, optionally filtered by company"""
    events = pd.read_csv(EVENT_KERNEL_PATH, parse_dates=['date'])
    if company:
        events = events[events['company'] == company].copy()
    return events

def prepare_data(company, ticker):
    """Prepare complete dataset for a company"""
    print(f"\n{'='*70}")
    print(f"PREPARING DATA: {company} ({ticker})")
    print(f"{'='*70}")
    
    # Load data
    returns, prices = load_stock_data(ticker)
    sentiment = load_sentiment_data(company)
    
    # Merge returns and sentiment
    data = returns.join(sentiment, how="inner").dropna()
    
    print(f"\n  Final dataset: {len(data)} observations")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    return data, prices

# =============================================================================
# MODELING
# =============================================================================

def fit_ols_robust(df, with_sent):
    """Fit OLS regression with robust standard errors"""
    X = df[["ret_mkt"]].copy()
    if with_sent:
        X["sent_composite_z"] = df["sent_composite_z"]
    X = sm.add_constant(X)
    y = df["ret_stk"]
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return model

def predict(model, df):
    """Generate predictions from fitted model"""
    feature_names = [col for col in model.params.index if col != 'const']
    X = df[feature_names].copy()
    X = sm.add_constant(X, has_constant='add')
    X = X[model.params.index]
    return model.predict(X)

def metrics(y_true, y_pred):
    """Calculate MAE and RMSE"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return mae, rmse

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_granger_test(data, company):
    """Run Granger causality test"""
    print(f"\n  Running Granger causality test...")
    
    gc_df = data[["ret_stk", "sent_composite_z"]].dropna()
    
    try:
        gtest = grangercausalitytests(gc_df, maxlag=5, verbose=False)
        
        rows = []
        for L in range(1, 6):
            p = gtest[L][0]["ssr_ftest"][1]
            rows.append({"lag": L, "p_value": p})
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"    Lag {L}: p={p:.4f} {sig}")
        
        gc_tbl = pd.DataFrame(rows)
        gc_tbl.to_csv(os.path.join(OUTPUT_DIR, f"granger_{company.lower()}.csv"), index=False)
        
        return gc_tbl
        
    except Exception as e:
        print(f"    ⚠ Granger test failed: {e}")
        return None

def create_results_summary(company, ticker, base, contra, mae_b, rmse_b, mae_c, rmse_c, test_size):
    """Create comprehensive results summary text"""
    
    summary = []
    summary.append("=" * 70)
    summary.append(f"CONTRA MODEL RESULTS: {company} ({ticker})")
    summary.append("=" * 70)
    summary.append("")
    
    # Model comparison
    summary.append("TRAINING SET PERFORMANCE (Adjusted R²):")
    summary.append(f"  Baseline (Market only):  {base.rsquared_adj:.6f}")
    summary.append(f"  CONTRA (+ Sentiment):    {contra.rsquared_adj:.6f}")
    improvement = contra.rsquared_adj - base.rsquared_adj
    pct_improvement = (improvement / base.rsquared_adj * 100) if base.rsquared_adj > 0 else 0
    summary.append(f"  Improvement:             {improvement:.6f} ({pct_improvement:+.2f}%)")
    summary.append("")
    
    # Test set performance
    summary.append(f"TEST SET PERFORMANCE ({test_size} observations):")
    mae_improvement = (mae_b - mae_c) / mae_b * 100
    rmse_improvement = (rmse_b - rmse_c) / rmse_b * 100
    summary.append(f"  MAE  — Baseline: {mae_b:.6e}  |  CONTRA: {mae_c:.6e}  |  Δ: {mae_improvement:+.2f}%")
    summary.append(f"  RMSE — Baseline: {rmse_b:.6e}  |  CONTRA: {rmse_c:.6e}  |  Δ: {rmse_improvement:+.2f}%")
    summary.append("")
    
    # Sentiment coefficient
    if "sent_composite_z" in contra.params:
        theta = contra.params["sent_composite_z"]
        p_val = contra.pvalues["sent_composite_z"]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        summary.append("SENTIMENT-ALPHA COEFFICIENT (θ):")
        summary.append(f"  θ = {theta:.6f}  (p={p_val:.4f}) {sig}")
        summary.append(f"  Interpretation: 1 SD ↑ in sentiment → {theta*100:.3f}% Δ in daily return")
        summary.append("")
    
    # Model parameters
    summary.append("BASELINE MODEL PARAMETERS:")
    summary.append(f"  α (intercept): {base.params['const']:.6e} (p={base.pvalues['const']:.4f})")
    summary.append(f"  β (market):    {base.params['ret_mkt']:.6f} (p={base.pvalues['ret_mkt']:.4f})")
    summary.append("")
    
    summary.append("CONTRA MODEL PARAMETERS:")
    summary.append(f"  α (intercept):  {contra.params['const']:.6e} (p={contra.pvalues['const']:.4f})")
    summary.append(f"  β (market):     {contra.params['ret_mkt']:.6f} (p={contra.pvalues['ret_mkt']:.4f})")
    if "sent_composite_z" in contra.params:
        summary.append(f"  θ (sentiment):  {contra.params['sent_composite_z']:.6f} (p={contra.pvalues['sent_composite_z']:.4f})")
    summary.append("")
    
    return "\n".join(summary)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_predictions(data, yhat_base, yhat_contra, train_size, company, ticker, events=None):
    """Create prediction comparison plot"""
    print(f"  Creating prediction plot...")
    
    plot_df = data.copy()
    plot_df["pred_base"] = yhat_base
    plot_df["pred_contra"] = yhat_contra
    split_date = data.index[train_size]
    
    fig = go.Figure()
    
    # Actual returns
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["ret_stk"],
        mode="lines", name="Actual Return",
        line=dict(color="black", width=1.5)
    ))
    
    # Baseline predictions
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["pred_base"],
        mode="lines", name="Baseline (Market Only)",
        line=dict(color="blue", dash="dot", width=1.5)
    ))
    
    # CONTRA predictions
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["pred_contra"],
        mode="lines", name="CONTRA (+ Sentiment)",
        line=dict(color="red", width=2)
    ))
    
    # Mark events
    if events is not None and not events.empty:
        for _, event in events.iterrows():
            event_date = pd.to_datetime(event['date'])
            if event_date in plot_df.index:
                color = "lightgreen" if event['event_type'] == "positive" else "lightcoral"
                fig.add_shape(
                    type="rect",
                    x0=event_date - pd.Timedelta(days=0.5),
                    x1=event_date + pd.Timedelta(days=0.5),
                    y0=0, y1=1, yref="paper",
                    fillcolor=color, opacity=0.2, line_width=0,
                    layer="below"
                )
    
    # Mark train/test split
    fig.add_shape(
        type="line",
        x0=split_date, x1=split_date,
        y0=0, y1=1, yref="paper",
        line=dict(color="green", width=2, dash="dash")
    )
    fig.add_annotation(
        x=split_date, y=1, yref="paper",
        text="Train/Test Split", showarrow=False,
        yshift=10, font=dict(color="green")
    )
    
    fig.update_layout(
        title=f"{company} ({ticker}): Actual vs Predicted Returns (Events: Green=Positive, Red=Negative)",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    
    return fig

def plot_sentiment_overlay(data, company, ticker, events=None):
    """Create sentiment overlay plot"""
    print(f"  Creating sentiment overlay plot...")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["ret_stk"],
            mode="lines", name="Stock Returns",
            line=dict(color="black", width=1)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["sent_composite_z"],
            mode="lines", name="Composite Sentiment",
            line=dict(color="red", width=1.5)
        ),
        secondary_y=True
    )
    
    # Mark events
    if events is not None and not events.empty:
        for _, event in events.iterrows():
            event_date = pd.to_datetime(event['date'])
            if event_date in data.index:
                color = "green" if event['event_type'] == "positive" else "red"
                fig.add_shape(
                    type="line",
                    x0=event_date, x1=event_date,
                    y0=0, y1=1, yref="paper",
                    line=dict(color=color, width=1, dash="dash"),
                    opacity=0.5, layer="below"
                )
    
    fig.update_layout(
        title=f"{company} ({ticker}): Returns vs Composite Sentiment (Events Marked)",
        hovermode="x unified",
        height=500
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Daily Return", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Z-Score", secondary_y=True)
    
    return fig

def plot_sentiment_components(data, company, ticker, events=None):
    """Create plot showing all sentiment components"""
    print(f"  Creating sentiment components plot...")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data["sent_wiki_z"],
        mode="lines", name="Wikipedia",
        line=dict(width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data["sent_trends_z"],
        mode="lines", name="Google Trends",
        line=dict(width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data["sent_event_z"],
        mode="lines", name="Event Kernel",
        line=dict(width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data["sent_composite_z"],
        mode="lines", name="Composite",
        line=dict(color="black", width=2)
    ))
    
    # Mark events
    if events is not None and not events.empty:
        for _, event in events.iterrows():
            event_date = pd.to_datetime(event['date'])
            if event_date in data.index:
                color = "green" if event['event_type'] == "positive" else "red"
                fig.add_shape(
                    type="line",
                    x0=event_date, x1=event_date,
                    y0=0, y1=1, yref="paper",
                    line=dict(color=color, width=1, dash="dash"),
                    opacity=0.3, layer="below"
                )
    
    fig.update_layout(
        title=f"{company} ({ticker}): Sentiment Components (Z-Scored, Events Marked)",
        xaxis_title="Date",
        yaxis_title="Z-Score",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    
    return fig

def plot_granger_causality(gc_tbl, company, ticker):
    """Create Granger causality plot"""
    print(f"  Creating Granger causality plot...")
    
    if gc_tbl is None:
        return None
    
    fig = px.bar(gc_tbl, x="lag", y="p_value",
                 title=f"{company} ({ticker}): Granger Causality (Sentiment → Returns)")
    fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                  annotation_text="α = 0.05", annotation_position="right")
    fig.update_layout(
        xaxis_title="Lag (days)",
        yaxis_title="p-value",
        height=400
    )
    
    return fig

def plot_residuals(base_model, contra_model, train_data, company, ticker):
    """Create residual comparison plot"""
    print(f"  Creating residuals plot...")
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Baseline Model", "CONTRA Model"))
    
    # Baseline residuals
    fig.add_trace(
        go.Scatter(x=base_model.fittedvalues, y=base_model.resid,
                   mode="markers", name="Baseline",
                   marker=dict(size=4, color="blue", opacity=0.5)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    
    # CONTRA residuals
    fig.add_trace(
        go.Scatter(x=contra_model.fittedvalues, y=contra_model.resid,
                   mode="markers", name="CONTRA",
                   marker=dict(size=4, color="red", opacity=0.5)),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_xaxes(title_text="Fitted Values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    
    fig.update_layout(
        title=f"{company} ({ticker}): Residual Analysis",
        showlegend=False,
        height=400
    )
    
    return fig

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_company(company, ticker):
    """Run complete CONTRA analysis for a company"""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {company} ({ticker})")
    print(f"{'='*70}")
    
    # Prepare data
    data, prices = prepare_data(company, ticker)
    
    # Load events for this company
    events = load_events(company)
    
    # Train/test split
    split_idx = int(len(data) * TRAIN_RATIO)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    print(f"\n  Train: {len(train)} obs ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"  Test:  {len(test)} obs ({test.index[0].date()} to {test.index[-1].date()})")
    
    # Fit models
    print(f"\n  Training models...")
    base = fit_ols_robust(train, with_sent=False)
    contra = fit_ols_robust(train, with_sent=True)
    print(f"    ✓ Baseline R²: {base.rsquared_adj:.6f}")
    print(f"    ✓ CONTRA R²:   {contra.rsquared_adj:.6f}")
    
    # Generate predictions
    print(f"\n  Generating predictions...")
    yhat_base = pd.concat([predict(base, train), predict(base, test)])
    yhat_contra = pd.concat([predict(contra, train), predict(contra, test)])
    
    # Calculate metrics
    mae_b, rmse_b = metrics(test["ret_stk"], predict(base, test))
    mae_c, rmse_c = metrics(test["ret_stk"], predict(contra, test))
    
    print(f"    ✓ Test MAE:  Baseline={mae_b:.6e}, CONTRA={mae_c:.6e}")
    print(f"    ✓ Test RMSE: Baseline={rmse_b:.6e}, CONTRA={rmse_c:.6e}")
    
    # Create results summary
    summary = create_results_summary(
        company, ticker, base, contra,
        mae_b, rmse_b, mae_c, rmse_c, len(test)
    )
    print("\n" + summary)
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save summary
    save_text(summary, f"summary_{company.lower()}.txt")
    print(f"  Saved summary")
    
    # Save regression details
    save_text(contra.summary().as_text(), f"regression_{company.lower()}.txt")
    print(f"  Saved regression details")
    
    # Generate and save plots
    print(f"\n  Generating visualizations...")
    
    fig1 = plot_predictions(data, yhat_base, yhat_contra, split_idx, company, ticker, events)
    save_html(fig1, f"predictions_{company.lower()}.html")
    print(f"  Saved predictions plot")
    
    fig2 = plot_sentiment_overlay(data, company, ticker, events)
    save_html(fig2, f"sentiment_overlay_{company.lower()}.html")
    print(f"  Saved sentiment overlay")
    
    fig3 = plot_sentiment_components(data, company, ticker, events)
    save_html(fig3, f"sentiment_components_{company.lower()}.html")
    print(f"  Saved sentiment components")
    
    fig4 = plot_residuals(base, contra, train, company, ticker)
    save_html(fig4, f"residuals_{company.lower()}.html")
    print(f"  Saved residuals plot")
    
    # Granger causality
    gc_tbl = run_granger_test(data, company)
    if gc_tbl is not None:
        fig5 = plot_granger_causality(gc_tbl, company, ticker)
        if fig5:
            save_html(fig5, f"granger_{company.lower()}.html")
            print(f"  Saved Granger causality plot")
    
    return {
        "company": company,
        "ticker": ticker,
        "base_r2": base.rsquared_adj,
        "contra_r2": contra.rsquared_adj,
        "r2_improvement": contra.rsquared_adj - base.rsquared_adj,
        "mae_base": mae_b,
        "mae_contra": mae_c,
        "rmse_base": rmse_b,
        "rmse_contra": rmse_c,
        "theta": contra.params.get("sent_composite_z", np.nan),
        "theta_pval": contra.pvalues.get("sent_composite_z", np.nan)
    }

# =============================================================================
# COMPARATIVE ANALYSIS
# =============================================================================

def create_comparison_plots(results):
    """Create comparative plots across all companies"""
    print(f"\n{'='*70}")
    print("CREATING COMPARATIVE ANALYSIS")
    print(f"{'='*70}")
    
    df = pd.DataFrame(results)
    
    # R² comparison
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df["company"], y=df["base_r2"],
        name="Baseline", marker_color="blue"
    ))
    fig1.add_trace(go.Bar(
        x=df["company"], y=df["contra_r2"],
        name="CONTRA", marker_color="red"
    ))
    fig1.update_layout(
        title="Model Performance Comparison (Adjusted R²)",
        xaxis_title="Company",
        yaxis_title="Adjusted R²",
        barmode="group",
        height=400
    )
    save_html(fig1, "comparison_r2.html")
    print("  Saved R² comparison")
    
    # RMSE comparison
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["company"], y=df["rmse_base"],
        name="Baseline", marker_color="blue"
    ))
    fig2.add_trace(go.Bar(
        x=df["company"], y=df["rmse_contra"],
        name="CONTRA", marker_color="red"
    ))
    fig2.update_layout(
        title="Test Set RMSE Comparison",
        xaxis_title="Company",
        yaxis_title="RMSE",
        barmode="group",
        height=400
    )
    save_html(fig2, "comparison_rmse.html")
    print("  Saved RMSE comparison")
    
    # Theta coefficients
    fig3 = go.Figure()
    colors = ["green" if x > 0 else "red" for x in df["theta"]]
    fig3.add_trace(go.Bar(
        x=df["company"], y=df["theta"],
        marker_color=colors,
        text=df["theta"].round(4),
        textposition="outside"
    ))
    fig3.update_layout(
        title="Sentiment-Alpha Coefficients (θ)",
        xaxis_title="Company",
        yaxis_title="θ (Sentiment Coefficient)",
        height=400
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="black")
    save_html(fig3, "comparison_theta.html")
    print("  Saved theta comparison")
    
    return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("CONTRA: SENTIMENT-ALPHA STOCK PREDICTION MODEL")
    print("=" * 70)
    print("\nAnalyzing Tesla, Amazon, and Meta (2018-2020)")
    print("Using pre-processed financial and sentiment data\n")
    
    # Analyze each company
    results = []
    for company, ticker in COMPANIES.items():
        result = analyze_company(company, ticker)
        results.append(result)
    
    # Create comparison plots
    comparison_df = create_comparison_plots(results)
    
    # Save comparison summary
    comparison_summary = []
    comparison_summary.append("=" * 70)
    comparison_summary.append("CONTRA ANALYSIS: COMPARATIVE SUMMARY")
    comparison_summary.append("=" * 70)
    comparison_summary.append("")
    
    for _, row in comparison_df.iterrows():
        comparison_summary.append(f"{row['company']} ({row['ticker']}):")
        comparison_summary.append(f"  R² Improvement: {row['r2_improvement']:.6f}")
        comparison_summary.append(f"  Sentiment Coefficient (θ): {row['theta']:.6f} (p={row['theta_pval']:.4f})")
        comparison_summary.append(f"  RMSE Reduction: {(row['rmse_base'] - row['rmse_contra'])/row['rmse_base']*100:+.2f}%")
        comparison_summary.append("")
    
    save_text("\n".join(comparison_summary), "comparison_summary.txt")
    
    # Final summary
    print(f"\n{'='*70}")
    print("CONTRA ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nFiles generated per company:")
    print("  summary_[company].txt - Performance metrics")
    print("  regression_[company].txt - Full regression output")
    print("  predictions_[company].html - Actual vs predicted")
    print("  sentiment_overlay_[company].html - Sentiment correlation")
    print("  sentiment_components_[company].html - Individual sources")
    print("  residuals_[company].html - Model diagnostics")
    print("  granger_[company].html - Causality tests")
    print("\nComparative analysis:")
    print("  comparison_r2.html - R² comparison")
    print("  comparison_rmse.html - RMSE comparison")
    print("  comparison_theta.html - Sentiment coefficients")
    print("  comparison_summary.txt - Summary statistics")
    print()

if __name__ == "__main__":
    main()

