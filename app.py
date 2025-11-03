"""
Enhanced Streamlit Stock Dashboard v2.6 - MODULAR VERSION
Main UI Application - All logic extracted to modules
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Import all modules
from modules.data_fetcher import get_data_optimized, get_spy_data, get_fear_greed_index
from modules.indicators import calc_indicators, validate_indicators
from modules.news_scraper import get_news_from_source, scrape_finviz_news, scrape_google_news, scrape_yahoo_news
from modules.sentiment import analyze_news_sentiment_bert_only
from modules.strategy import rule_based_signal_v2
from modules.risk_metrics import calculate_risk_metrics
from modules.backtester import backtest_strategy
from modules.ml_models import run_ml_analysis, calculate_ensemble_recommendation
from modules.monte_carlo import estimate_days_to_target_advanced, monte_carlo_price_simulation

st.set_page_config(layout="wide", page_title="Enhanced Stock Dashboard v2.6 (by Sadiq)")

# ==================== UI LAYOUT ====================

st.title("📈 Enhanced Stock Dashboard v2.6 (by Sadiq)")
st.caption("Advanced technical analysis with ML models, multi-source news, and strategy backtesting")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

tickers_input = st.sidebar.text_area("Tickers (comma separated)", value="AAPL, AMZN, TSLA", height=100)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
lookback = st.sidebar.selectbox("Lookback Period", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Data Interval", ["1d", "1wk"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Sentiment Analysis")
use_finbert = st.sidebar.checkbox("Use FinBERT (slower, more accurate)", value=False, 
                                   help="FinBERT is 60% more accurate but takes ~10 seconds")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Indicator Parameters")
with st.sidebar.expander("RSI Settings"):
    rsi_period = st.slider("RSI Period", 10, 30, 14)
    rsi_oversold = st.slider("RSI Oversold", 10, 40, 30)
    rsi_overbought = st.slider("RSI Overbought", 60, 85, 70)

with st.sidebar.expander("SMA Settings"):
    sma_short = st.slider("SMA Short Window", 10, 40, 20)
    sma_long = st.slider("SMA Long Window", 30, 200, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Signal Weights")
with st.sidebar.expander("Adjust Signal Weights"):
    w_rsi = st.slider("RSI weight", 0.0, 5.0, 2.0, 0.1)
    w_macd = st.slider("MACD weight", 0.0, 5.0, 1.5, 0.1)
    w_sma = st.slider("SMA weight", 0.0, 5.0, 1.0, 0.1)
    w_bb = st.slider("BB weight", 0.0, 5.0, 1.0, 0.1)
    w_stoch = st.slider("Stochastic weight", 0.0, 3.0, 0.8, 0.1)
    w_vol = st.slider("Volume weight", 0.0, 2.0, 0.5, 0.1)
    w_adx = st.slider("ADX weight", 0.0, 3.0, 1.0, 0.1)

weights = {'RSI': w_rsi, 'MACD': w_macd, 'SMA': w_sma, 'BB': w_bb, 'Stoch': w_stoch, 'Volume': w_vol, 'ADX': w_adx}

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Algorithm Weights for Ensemble")
with st.sidebar.expander("ML Model Weight Distribution"):
    w_rf = st.slider("Random Forest", 0.0, 1.0, 0.20, 0.05)
    w_xgb = st.slider("XGBoost", 0.0, 1.0, 0.20, 0.05)
    w_arima = st.slider("ARIMA + GARCH", 0.0, 1.0, 0.15, 0.05)
    w_lstm = st.slider("LSTM", 0.0, 1.0, 0.20, 0.05)
    w_rnn = st.slider("RNN", 0.0, 1.0, 0.20, 0.05)
    w_mc = st.slider("Monte Carlo", 0.0, 1.0, 0.10, 0.05)
    
    total_weight = w_rf + w_xgb + w_arima + w_lstm + w_rnn + w_mc
    st.info(f"Total Weight: {total_weight:.2f} (should sum to ~1.0)")

ml_weights = {
    'Random Forest': w_rf,
    'XGBoost': w_xgb,
    'ARIMA + GARCH': w_arima,
    'LSTM': w_lstm,
    'RNN': w_rnn,
    'Monte Carlo': w_mc
}

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Simulation & Backtest")
with st.sidebar.expander("Monte Carlo Settings"):
    sim_count = st.select_slider("Simulation count", options=[500, 1000, 2500, 5000, 10000], value=2500)
    max_days = st.slider("Max days for sim", 90, 730, 365, 30)

with st.sidebar.expander("Backtest Settings"):
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
    confidence_threshold = st.slider("Min confidence for trade (%)", 0, 50, 20, 5)
    stop_loss_pct = st.slider("Stop Loss (%)", 1, 20, 5, 1) / 100
    take_profit_pct = st.slider("Take Profit (%)", 5, 50, 15, 5) / 100

st.sidebar.markdown("---")
refresh_button = st.sidebar.button("🔄 Refresh Data", type="primary")

# Get market data
fg_score, fg_rating, fg_color = get_fear_greed_index()
spy_hist = get_spy_data(period=lookback, interval=interval)

# Session State
if "data_cache" not in st.session_state:
    st.session_state["data_cache"] = {}
if "ml_cache" not in st.session_state:
    st.session_state["ml_cache"] = {}

# Market Overview
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.subheader("🌍 Market Sentiment")
    c1, c2 = st.columns(2)
    with c1:
        if fg_score is not None:
            st.metric("Fear & Greed Score", f"{fg_score:.2f}")
            # Sentiment Meter-style indicator
            sentiment_color = "#ff4444" if fg_score < 25 else "#ffaa00" if fg_score < 45 else "#ffff00" if fg_score < 55 else "#aaff00" if fg_score < 75 else "#44ff44"
            st.markdown(f"""
            <div style="background: linear-gradient(to right, #ff4444 0%, #ffaa00 25%, #ffff00 50%, #aaff00 75%, #44ff44 100%); 
                        height: 30px; border-radius: 15px; position: relative; margin: 10px 0;">
                <div style="position: absolute; left: {fg_score}%; top: 50%; transform: translate(-50%, -50%); 
                           width: 20px; height: 20px; background: white; border: 3px solid black; 
                           border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Fear & Greed Score", "N/A")
    with c2:
        st.write(f"**{fg_rating}**")
        st.write(fg_color)

with col2:
    st.subheader("📈 SPY")
    if not spy_hist.empty:
        spy_change = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-2]) / spy_hist['Close'].iloc[-2] * 100
        st.metric("SPY", f"${spy_hist['Close'].iloc[-1]:.2f}", f"{spy_change:+.2f}%")

with col3:
    st.subheader("📍 Select Stock")
    selected = st.selectbox("Choose ticker to analyze", options=tickers if tickers else ["AAPL"], label_visibility="collapsed")

# Single Stock Analysis
st.markdown("---")
st.header(f"🔍 Deep Dive: {selected}")

cache_key = f"{selected}_{lookback}_{interval}"
if cache_key in st.session_state["data_cache"] and not refresh_button:
    hist, info = st.session_state["data_cache"][cache_key]
else:
    with st.spinner(f"Loading {selected}..."):
        hist, info = get_data_optimized(selected, period=lookback, interval=interval)
        st.session_state["data_cache"][cache_key] = (hist, info)
        # Clear ML cache when data refreshes
        if cache_key in st.session_state["ml_cache"]:
            del st.session_state["ml_cache"][cache_key]

if hist.empty:
    st.error(f"❌ No data for {selected}. Reason: {info.get('_error', 'Unknown')}")
    st.stop()

df = calc_indicators(hist, rsi_period=rsi_period, sma_short=sma_short, sma_long=sma_long)
is_valid, validation = validate_indicators(df)
if not is_valid:
    st.warning(f"⚠️ Data quality issue: {validation}")

latest = df.iloc[-1]

# Key Metrics
st.subheader("💰 Key Metrics")
m1, m2, m3, m4, m5 = st.columns(5)

price_str = f"${latest['Close']:.2f}"
vol_str = f"{latest['Volume'] / 1_000_000:.2f}M"
market_cap = info.get("marketCap")
mc_str = f"${market_cap/1_000_000_000:.2f}B" if market_cap else "N/A"
pe_val = info.get("forwardPE") or info.get("trailingPE") or "N/A"
pe_str = f"{pe_val:.1f}x" if isinstance(pe_val, (int, float)) else pe_val

m1.metric("Price", price_str)
m2.metric("Volume", vol_str)
m3.metric("Market Cap", mc_str)
m4.metric("Fwd P/E", pe_str)

corr = 0.0
if not spy_hist.empty:
    try:
        min_len = min(len(spy_hist), len(df))
        corr = df['Close'].iloc[-min_len:].corr(spy_hist['Close'].iloc[-min_len:])
        corr = 0.0 if np.isnan(corr) else corr
    except:
        corr = 0.0
m5.metric("SPY Correlation", f"{corr:.2f}")

# Risk Metrics
st.subheader("⚠️ Risk Analysis")
risk_metrics = calculate_risk_metrics(df)

if risk_metrics:
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Annual Return", f"{risk_metrics['annual_return']*100:.2f}%")
    r2.metric("Volatility", f"{risk_metrics['volatility']*100:.2f}%")
    r3.metric("Sharpe Ratio", f"{risk_metrics['sharpe']:.2f}")
    r4.metric("Sortino Ratio", f"{risk_metrics['sortino']:.2f}")
    r5.metric("Max Drawdown", f"{risk_metrics['max_drawdown']*100:.2f}%")

# Charts
st.markdown("---")
st.subheader("📉 Price Chart with Technical Indicators")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    row_heights=[0.5, 0.25, 0.25], subplot_titles=('Price & Indicators', 'RSI', 'MACD'))

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

if 'SMA_short' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_short'], mode='lines', name=f'SMA {sma_short}', line=dict(width=1, color='orange')), row=1, col=1)
if 'SMA_long' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_long'], mode='lines', name=f'SMA {sma_long}', line=dict(width=1, color='blue')), row=1, col=1)
if 'BB_upper' in df and 'BB_lower' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB Upper', line=dict(dash='dot', width=1, color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB Lower', line=dict(dash='dot', width=1, color='gray')), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='Signal', line=dict(color='red')), row=3, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram'), row=3, col=1)

fig.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ==================== Multi-Source News Section ====================

st.markdown("---")
st.header(f"📰 Latest News & Sentiment for {selected}")

# Header with source selector and refresh button
col_source, col_spacer, col_refresh = st.columns([2, 2, 1])

with col_source:
    news_source = st.selectbox(
        "📡 News Source:",
        options=["Finviz", "Google News", "Yahoo Finance"],
        index=0,
        key="news_source_selector",
        help="Choose where to fetch news from"
    )

with col_refresh:
    st.write("")  # Spacer
    st.write("")  # Spacer
    if st.button("🔄 Refresh News", key="refresh_news", type="secondary"):
        # Clear all news caches
        scrape_finviz_news.clear()
        scrape_google_news.clear()
        scrape_yahoo_news.clear()
        st.success("✅ News refreshed!")
        st.rerun()

st.markdown("---")

# Fetch news from selected source
st.subheader(f"📰 Latest News from {news_source}")

with st.spinner(f"Fetching latest news from {news_source}..."):
    news_data = get_news_from_source(selected, news_source, max_news=10)

if news_data:
    # Analyze sentiment - FinBERT only
    if use_finbert:
        with st.spinner("Analyzing with FinBERT... (this may take 10-15 seconds)"):
            news_data = analyze_news_sentiment_bert_only(news_data, use_bert=True)
    else:
        news_data = analyze_news_sentiment_bert_only(news_data, use_bert=False)
    
    st.caption(f"📊 Showing {len(news_data)} most recent news items from {news_source}")
    
    # Create DataFrame for table display
    news_table_data = []
    for item in news_data:
        # Format timestamp
        timestamp = f"{item.get('Date', 'N/A')} {item.get('Time', 'N/A')}"
        
        # Create clickable link
        title = item.get('Title', 'No title')
        link = item.get('Link', '#')
        
        # Store raw sentiment data for later use in HTML table
        news_table_data.append({
            'Timestamp': timestamp,
            'Headline': title,
            'Link': link,
            'sentiment_emoji': item.get('sentiment_emoji', '⚪'),
            'sentiment_label': item.get('sentiment_label', 'N/A'),
            'sentiment_score': item.get('sentiment_score', 0.0),
            'bert_score': item.get('bert_score', 0.0),
            'bert_label': item.get('bert_label', 'N/A'),
            'bert_available': item.get('bert_available', False),
            'Source': item.get('Source', news_source)
        })
    
    # Create DataFrame
    news_df = pd.DataFrame(news_table_data)
    
    # Display as interactive table with clickable links
    st.markdown("**Click on headlines to read full articles:**")
    
    # Create HTML table with clickable links and FinBERT-only sentiment
    html_table = '<table style="width:100%; border-collapse: collapse;">'
    html_table += '<tr style="background-color: #f0f2f6; font-weight: bold;">'
    html_table += '<th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Time</th>'
    html_table += '<th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Headline</th>'
    html_table += '<th style="padding: 10px; text-align: center; border-bottom: 2px solid #ddd;">Sentiment</th>'
    html_table += '</tr>'
    
    for idx, row in news_df.iterrows():
        html_table += f'<tr style="border-bottom: 1px solid #ddd;">'
        html_table += f'<td style="padding: 8px; vertical-align: top; white-space: nowrap;">{row["Timestamp"]}</td>'
        html_table += f'<td style="padding: 8px;"><a href="{row["Link"]}" target="_blank" style="color: #0066cc; text-decoration: none;">{row["Headline"]}</a><br><small style="color: #666;">📡 {row["Source"]}</small></td>'
        
        # Build sentiment display - FinBERT only
        sentiment_emoji = row['sentiment_emoji']
        sentiment_label = row['sentiment_label']
        sentiment_score = row['sentiment_score']
        
        if row['bert_available']:
            bert_score = row['bert_score']
            sentiment_display = f'{sentiment_emoji} <strong>{sentiment_label}</strong> (FinBERT: {bert_score:.2f})'
        else:
            sentiment_display = f'{sentiment_emoji} <strong>{sentiment_label}</strong>'
        
        html_table += f'<td style="padding: 8px; text-align: center; white-space: nowrap;">{sentiment_display}</td>'
        html_table += '</tr>'
    
    html_table += '</table>'
    
    # Display HTML table
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Overall sentiment summary
    if news_data and len(news_data) > 0:
        st.markdown("---")
        st.markdown("**📊 Overall Sentiment Summary:**")
    
        sentiment_scores = [item.get('sentiment_score', 0) for item in news_data]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
        positive_count = sum(1 for s in sentiment_scores if s >= 0.05)
        neutral_count = sum(1 for s in sentiment_scores if -0.05 < s < 0.05)
        negative_count = sum(1 for s in sentiment_scores if s <= -0.05)
    
        col_s1.metric("🟢 Positive", f"{positive_count}")
        col_s2.metric("🟡 Neutral", f"{neutral_count}")
        col_s3.metric("🔴 Negative", f"{negative_count}")
        col_s4.metric("Average Score", f"{avg_sentiment:.2f}")
    
        # Show method used
        if news_data[0].get('bert_available', False):
            st.caption("📊 Using FinBERT sentiment analysis (financial domain-specific)")
        else:
            st.caption("⚠️ FinBERT not available - showing neutral sentiment")
    
        # Overall indicator
        if avg_sentiment >= 0.05:
            st.success(f"📈 Overall Positive News Sentiment on {news_source}")
        elif avg_sentiment <= -0.05:
            st.error(f"📉 Overall Negative News Sentiment on {news_source}")
        else:
            st.info(f"➡️ Overall Neutral News Sentiment on {news_source}")
else:
    st.info(f"📭 No recent news available from {news_source} at this time")
    st.caption(f"This could be due to network issues or {selected} not being covered on {news_source}")
    
    # Suggest trying other sources
    if news_source == "Finviz":
        st.info("💡 Try switching to **Google News** for more coverage")
    else:
        st.info("💡 Try switching to **Finviz** for ticker-specific news")

# ==================== Rule-Based Recommendation ====================

st.markdown("---")
st.subheader("🎯 Rule-Based Trading Signals")

recommendation, signals, confidence, raw_scores = rule_based_signal_v2(df, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought, weights=weights)

col1, col2 = st.columns([1, 2])
with col1:
    color = "green" if "BUY" in recommendation else "red" if "SELL" in recommendation else "orange"
    st.markdown(f"<h2 style='color:{color}; text-align:center;'>{recommendation}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center;'>Confidence: {confidence:.1f}%</h4>", unsafe_allow_html=True)
    st.metric("Buy Score", f"{raw_scores['buy']:.2f}")
    st.metric("Sell Score", f"{raw_scores['sell']:.2f}")
    st.metric("Net Score", f"{raw_scores['net']:.2f}")

with col2:
    st.write("**Signal Breakdown:**")
    for signal_text, signal_type, weight, extra in signals:
        emoji = {"BUY": "🟢", "SELL": "🔴", "CONFIRM": "✅", "AMPLIFY": "📈", "DAMPEN": "📉"}.get(signal_type, "⚪")
        display_text = f"{emoji} {signal_text}"
        if extra:
            display_text += f" ({extra})"
        if weight > 0:
            display_text += f" [w={weight:.2f}]"
        st.write(display_text)

# ==================== ML Analysis Section ====================

st.markdown("---")
st.subheader("🤖 Machine Learning Models Analysis")

ml_button = st.button("🚀 Run ML Analysis", type="primary", use_container_width=True)

if ml_button or cache_key in st.session_state["ml_cache"]:
    if ml_button or cache_key not in st.session_state["ml_cache"]:
        ml_results = run_ml_analysis(df)
        if ml_results:
            st.session_state["ml_cache"][cache_key] = ml_results
    else:
        ml_results = st.session_state["ml_cache"][cache_key]
    
    if ml_results:
        # Display results table
        ml_df = pd.DataFrame(ml_results, columns=["Algorithm", "Key Parameters", "Performance Metrics", "Recommendation", "Confidence"])
        st.dataframe(ml_df, use_container_width=True, height=300)
        
        # Ensemble Recommendation
        st.markdown("---")
        st.subheader("🎯 Ensemble Recommendation")
        ensemble_rec, ensemble_conf, agreement = calculate_ensemble_recommendation(ml_results, weights=ml_weights)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ens_color = "green" if ensemble_rec == "BUY" else "red" if ensemble_rec == "SELL" else "orange"
            st.markdown(f"<h2 style='color:{ens_color}; text-align:center;'>{ensemble_rec}</h2>", unsafe_allow_html=True)
        with col2:
            st.metric("Avg Confidence", ensemble_conf)
        with col3:
            st.metric("Model Agreement", agreement)
        
        st.info("💡 Ensemble uses weighted voting across all models based on the configured weights in the sidebar")

# ==================== Backtest Results ====================

st.markdown("---")
st.subheader("📊 Strategy Backtest Performance")

with st.spinner("Running backtest..."):
    backtest_results = backtest_strategy(df, weights=weights, initial_capital=initial_capital,
                                        confidence_threshold=confidence_threshold,
                                        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("Final Capital", f"${backtest_results['final_capital']:.2f}")
b2.metric("Strategy Return", f"{backtest_results['total_return']*100:.2f}%")
b3.metric("Buy & Hold Return", f"{backtest_results['buy_hold_return']*100:.2f}%")
b4.metric("Alpha", f"{backtest_results['alpha']*100:.2f}%", delta=f"{backtest_results['alpha']*100:.2f}%")
b5.metric("Number of Trades", backtest_results['num_trades'])

b6, b7, b8 = st.columns(3)
b6.metric("Win Rate", f"{backtest_results['win_rate']*100:.1f}%")
b7.metric("Avg Win", f"{backtest_results['avg_win']*100:.2f}%")
b8.metric("Avg Loss", f"{backtest_results['avg_loss']*100:.2f}%")

if backtest_results['equity_curve']:
    st.subheader("📈 Equity Curve")
    equity_df = pd.DataFrame(backtest_results['equity_curve'])
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=equity_df['date'], y=equity_df['value'], mode='lines',
                                    name='Portfolio Value', line=dict(color='green', width=2)))
    
    if backtest_results['buy_signals']:
        buy_dates = [df.index[i] for i in backtest_results['buy_signals']]
        fig_equity.add_trace(go.Scatter(
            x=buy_dates,
            y=[equity_df[equity_df['date'] == d]['value'].iloc[0] if len(equity_df[equity_df['date'] == d]) > 0 else 0 for d in buy_dates],
            mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))
    
    if backtest_results['sell_signals']:
        sell_dates = [df.index[i] for i in backtest_results['sell_signals']]
        fig_equity.add_trace(go.Scatter(
            x=sell_dates,
            y=[equity_df[equity_df['date'] == d]['value'].iloc[0] if len(equity_df[equity_df['date'] == d]) > 0 else 0 for d in sell_dates],
            mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))
    
    fig_equity.update_layout(height=400, xaxis_title="Date", yaxis_title="Portfolio Value ($)", hovermode='x unified')
    st.plotly_chart(fig_equity, use_container_width=True)

with st.expander("📋 View Trade History"):
    if backtest_results['positions']:
        trades_df = pd.DataFrame(backtest_results['positions'], columns=['Action', 'Date', 'Price', 'Value/Shares', 'Return/Conf'])
        st.dataframe(trades_df, use_container_width=True)

# ==================== Monte Carlo Projections ====================

st.markdown("---")
st.subheader("🎲 Monte Carlo Price Target Projections")

targets = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
sim_results = []
current_price = float(latest['Close'])

with st.spinner("Running Monte Carlo simulations..."):
    for t in targets:
        res = estimate_days_to_target_advanced(df, current_price, target_return=t, sims=sim_count, max_days=max_days)
        sim_results.append({
            "Target (%)": int(t*100),
            "Target Price": f"${current_price * (1+t):.2f}",
            "Probability (%)": f"{res['probability']*100:.1f}",
            "Median Days": res['median_days'],
            "90th Pctl Days": res['90pct_days'],
            "10th Pctl Days": res['10pct_days']
        })

mc_df = pd.DataFrame(sim_results)
st.dataframe(mc_df, use_container_width=True)
st.info(f"💡 Based on {sim_count:,} simulations with {max_days} day horizon using Student's t-distribution")

# ==================== NEW: Monte Carlo Price Simulation ====================

st.markdown("---")
st.subheader("🎲 Monte Carlo Price Simulation")

with st.spinner("Running price simulations across timeframes..."):
    mc_price_sim = monte_carlo_price_simulation(df, current_price, sims=sim_count)

if not mc_price_sim.empty:
    st.dataframe(mc_price_sim, use_container_width=True, hide_index=True)
    
    st.info(f"💡 Based on {sim_count:,} simulations using Student's t-distribution with volatility clustering. "
            f"Shows 5th percentile (worst case), median (most likely), and 95th percentile (best case) prices.")
else:
    st.warning("⚠️ Insufficient data for price simulation")

# ==================== Footer ====================

st.markdown("---")
st.subheader("📝 Notes & Disclaimers")

st.write("""
### Improvements in v2.6:
- ✅ **Simplified Ticker Input** - Direct ticker entry (default: AAPL, AMZN, TSLA) without market cap groups
- ✅ **Enhanced Sentiment Indicator** - Meter-style visual indicator for Fear & Greed Score (2 decimal precision)
- ✅ **Yahoo Finance News** - Added as third news source alongside Finviz and Google News
- ✅ **Weighted Ensemble Voting** - ML algorithms now use configurable weights (RF: 0.2, XGB: 0.2, ARIMA: 0.15, LSTM: 0.2, RNN: 0.2, MC: 0.1)
- ✅ **Customizable ML Weights** - Adjust algorithm importance via sidebar sliders

### Previous Features (v2.3 - MODULAR VERSION):
- ✅ **Modular Architecture** - Code organized into 9 specialized modules
- ✅ **Better Performance** - Improved caching and code organization
- ✅ **Easier Maintenance** - Each module handles one responsibility
- ✅ **Multi-Source News** - Finviz, Google News, Yahoo Finance with dropdown selector
- ✅ **News Sentiment Analysis** - FinBERT ONLY (no VADER in table) with table display
- ✅ **Machine Learning Integration** - Random Forest, XGBoost, ARIMA+GARCH, LSTM, RNN, Monte Carlo
- ✅ **Ensemble Recommendations** - Weighted voting across all models
- ✅ **Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score, AUC for each model
- ✅ **Strategy Backtesting** - With risk management (stop loss, take profit)
- ✅ **Risk Analytics** - Sharpe, Sortino, Max Drawdown, Calmar ratio

### Module Structure:
- `data_fetcher.py` - API calls and data retrieval
- `indicators.py` - Technical indicator calculations
- `news_scraper.py` - Multi-source news aggregation (Finviz, Google News, Yahoo Finance)
- `sentiment.py` - FinBERT sentiment analysis
- `strategy.py` - Rule-based trading signals
- `risk_metrics.py` - Risk calculations
- `backtester.py` - Strategy backtesting
- `ml_models.py` - All 6 ML algorithms with weighted ensemble
- `monte_carlo.py` - Monte Carlo simulations

### Limitations & Disclaimers:
- ⚠️ **NOT FINANCIAL ADVICE** - This tool is for educational purposes only
- ⚠️ Past performance does not guarantee future results
- ⚠️ ML models can overfit to historical patterns that may not persist
- ⚠️ Market conditions change - models trained on past data may not predict future well
- ⚠️ Sentiment analysis is based on headlines only, not full article content
- ⚠️ Always do your own research and consult a financial advisor
- ⚠️ Consider paper trading before using real capital

### Dependencies:
- `pip install feedparser --break-system-packages` (for Google News)
- `pip install transformers torch --break-system-packages` (for FinBERT)

### Recommended Next Steps:
1. Compare sentiment across different news sources (Finviz, Google News, Yahoo Finance)
2. Monitor ensemble agreement - high disagreement suggests uncertainty
3. Test different weight configurations for both rule-based signals and ML algorithms
4. Paper trade the strategy for at least 3 months before deploying real capital
5. Consider adding fundamental analysis (earnings, revenue growth, debt ratios)
""")

st.markdown("---")
st.caption("Enhanced Stock Dashboard v2.6 - MODULAR | Built with Streamlit | Data: Yahoo Finance, Finviz, Google News | ML: scikit-learn, XGBoost, TensorFlow, FinBERT")
