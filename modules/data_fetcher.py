"""
Data Fetching Module
Handles all external API calls for stock data, market indicators, and Fear & Greed index
"""
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import streamlit as st


@st.cache_data(ttl=3600)
def get_data_optimized(ticker: str, period: str = "1y", interval: str = "1d", fetch_info: bool = True):
    """
    Optimized data fetch with selective info retrieval and robust error handling
    Returns (hist_df, info_dict) or (empty_df, error_dict)
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if hist.empty:
            raise ValueError("Empty history returned")
        
        missing = set(required_cols) - set(hist.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if len(hist) < 50:
            raise ValueError("Need at least 50 data points")
        
        # Fetch comprehensive info with multiple fallback attempts
        info = {}
        if fetch_info:
            try:
                # First attempt: standard info
                raw_info = tk.info
                
                # Extract with fallbacks for each field
                info = {
                    'forwardPE': raw_info.get('forwardPE') or raw_info.get('trailingPE'),
                    'trailingPE': raw_info.get('trailingPE'),
                    'marketCap': raw_info.get('marketCap'),
                    'shortName': raw_info.get('shortName') or raw_info.get('longName') or ticker,
                    'currentPrice': raw_info.get('currentPrice') or raw_info.get('regularMarketPrice')
                }
                
                # If marketCap is missing, try to calculate from shares outstanding
                if not info['marketCap']:
                    shares = raw_info.get('sharesOutstanding')
                    price = info['currentPrice'] or hist['Close'].iloc[-1]
                    if shares and price:
                        info['marketCap'] = shares * price
                
                # Additional attempts for missing P/E
                if not info['forwardPE'] and not info['trailingPE']:
                    # Try alternative fields
                    info['forwardPE'] = raw_info.get('forwardEps')
                    info['trailingPE'] = raw_info.get('trailingEps')
                
            except Exception as e:
                # Fallback: create minimal info with calculated values
                try:
                    info = {
                        'shortName': ticker,
                        'currentPrice': hist['Close'].iloc[-1],
                        'marketCap': None,
                        'forwardPE': None,
                        'trailingPE': None
                    }
                    
                    # Try fast_info as alternative
                    try:
                        fast_info = tk.fast_info
                        if hasattr(fast_info, 'market_cap'):
                            info['marketCap'] = fast_info.market_cap
                        if hasattr(fast_info, 'forward_pe'):
                            info['forwardPE'] = fast_info.forward_pe
                        if hasattr(fast_info, 'trailing_pe'):
                            info['trailingPE'] = fast_info.trailing_pe
                    except:
                        pass
                        
                except:
                    info = {'shortName': ticker}
        
        return hist, info
    except Exception as e:
        return pd.DataFrame(), {"_error": str(e)}


@st.cache_data(ttl=3600)
def get_spy_data(period="1y", interval="1d"):
    """Cache SPY data for correlation and beta calculations"""
    hist, _ = get_data_optimized("SPY", period=period, interval=interval, fetch_info=False)
    return hist


@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Fetch CNN Fear & Greed Index with fallback"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}
    base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    
    for days_back in range(0, 3):
        d = (date.today() - timedelta(days=days_back)).isoformat()
        try:
            resp = requests.get(base_url + d, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            fg = data.get("fear_and_greed", {})
            score = fg.get("score")
            rating = fg.get("rating", "N/A")
            
            if score is None:
                continue
            
            if score < 25:
                color = "🟥 Extreme Fear"
            elif score < 45:
                color = "🔴 Fear"
            elif score < 55:
                color = "🟡 Neutral"
            elif score < 75:
                color = "🟢 Greed"
            else:
                color = "🟩 Extreme Greed"
            
            return score, rating, color
        except Exception:
            continue
    
    return None, "N/A", "N/A"
