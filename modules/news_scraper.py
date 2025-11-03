"""
News Scraping Module
Multi-source news aggregation from Finviz and Google News
"""
from bs4 import BeautifulSoup
import requests
import streamlit as st
from datetime import datetime

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except:
    FEEDPARSER_AVAILABLE = False


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def scrape_finviz_news(ticker: str, max_news: int = 10):
    """
    Scrape latest news from Finviz
    Returns: List of dicts with {date, time, title, link, source}
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        news_table = soup.find('table', {'id': 'news-table'})
        
        if not news_table:
            return []
        
        news_list = []
        current_date = None
        
        for row in news_table.find_all('tr')[:max_news]:
            td_timestamp = row.find('td', {'align': 'right', 'width': '130'})
            td_content = row.find('td', {'align': 'left'})
            
            if not td_timestamp or not td_content:
                continue
            
            # Parse timestamp
            timestamp_text = td_timestamp.get_text().strip()
            timestamp_parts = timestamp_text.split()
            
            if len(timestamp_parts) == 2:
                current_date = timestamp_parts[0]
                time_str = timestamp_parts[1]
            else:
                time_str = timestamp_parts[0]
            
            # Extract news info
            link_tag = td_content.find('a')
            if link_tag:
                title = link_tag.get_text().strip()
                link = link_tag.get('href', '')
                
                # Get source
                source_span = td_content.find('span', {'class': 'news-link-right'})
                source = source_span.get_text().strip() if source_span else 'Finviz'
                
                # Make link absolute
                if link and not link.startswith('http'):
                    link = 'https://finviz.com/' + link
                
                news_list.append({
                    'Date': current_date,
                    'Time': time_str,
                    'Source': source,
                    'Title': title,
                    'Link': link
                })
        
        return news_list
    
    except Exception as e:
        st.warning(f"Could not fetch Finviz news for {ticker}: {str(e)[:100]}")
        return []


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def scrape_google_news(ticker: str, max_news: int = 10):
    """
    Scrape latest news from Google News using RSS
    Returns: List of dicts with {date, time, title, link, source}
    """
    if not FEEDPARSER_AVAILABLE:
        st.warning("feedparser not installed. Run: pip install feedparser --break-system-packages")
        return []
    
    try:
        # Google News RSS feed for stock ticker
        query = f"{ticker} stock"
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse RSS feed
        feed = feedparser.parse(response.content)
        
        if not feed.entries:
            return []
        
        news_list = []
        
        for entry in feed.entries[:max_news]:
            # Parse timestamp
            published = entry.get('published_parsed', None)
            if published:
                dt = datetime(*published[:6])
                date_str = dt.strftime('%b-%d-%y')
                time_str = dt.strftime('%I:%M%p')
            else:
                date_str = 'Today'
                time_str = 'N/A'
            
            # Extract info
            title = entry.get('title', 'No title')
            link = entry.get('link', '#')
            
            # Extract source from title (Google News format: "Title - Source")
            source = 'Google News'
            if ' - ' in title:
                parts = title.rsplit(' - ', 1)
                if len(parts) == 2:
                    title = parts[0]
                    source = parts[1]
            
            news_list.append({
                'Date': date_str,
                'Time': time_str,
                'Source': source,
                'Title': title,
                'Link': link
            })
        
        return news_list
    
    except Exception as e:
        st.warning(f"Could not fetch Google News for {ticker}: {str(e)[:100]}")
        return []


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def scrape_yahoo_news(ticker: str, max_news: int = 10):
    """
    Scrape latest news from Yahoo Finance
    Returns: List of dicts with {date, time, title, link, source}
    """
    try:
        url = f"https://finance.yahoo.com/quote/{ticker.upper()}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Yahoo Finance uses various container classes for news
        news_items = soup.find_all('li', {'class': lambda x: x and 'stream-item' in x})
        
        if not news_items:
            # Try alternative selector
            news_items = soup.find_all('div', {'class': lambda x: x and 'Ov(h)' in x})
        
        if not news_items:
            return []
        
        news_list = []
        
        for item in news_items[:max_news]:
            try:
                # Find title and link
                title_tag = item.find('a')
                if not title_tag:
                    continue
                    
                title = title_tag.get_text().strip()
                link = title_tag.get('href', '')
                
                # Make link absolute if needed
                if link and not link.startswith('http'):
                    link = 'https://finance.yahoo.com' + link
                
                # Find timestamp (usually in a time tag or span)
                time_tag = item.find('time')
                if time_tag:
                    timestamp_text = time_tag.get_text().strip()
                else:
                    time_span = item.find('span', {'class': lambda x: x and 'C(#959595)' in x})
                    timestamp_text = time_span.get_text().strip() if time_span else 'Today'
                
                # Parse timestamp
                if '•' in timestamp_text:
                    parts = timestamp_text.split('•')
                    date_str = parts[0].strip() if len(parts) > 0 else 'Today'
                    time_str = parts[1].strip() if len(parts) > 1 else ''
                else:
                    date_str = timestamp_text
                    time_str = ''
                
                # Find source (if available)
                source_tag = item.find('div', {'class': lambda x: x and 'C(#959595)' in x})
                source = source_tag.get_text().strip() if source_tag else 'Yahoo Finance'
                
                news_list.append({
                    'Date': date_str,
                    'Time': time_str,
                    'Source': source,
                    'Title': title,
                    'Link': link
                })
            except Exception as inner_e:
                continue
        
        return news_list
    
    except Exception as e:
        st.warning(f"Could not fetch Yahoo Finance news for {ticker}: {str(e)[:100]}")
        return []


def get_news_from_source(ticker: str, source: str, max_news: int = 10):
    """
    Unified function to get news from selected source
    
    Args:
        ticker: Stock ticker symbol
        source: News source ('Finviz', 'Google News', 'Yahoo Finance')
        max_news: Maximum number of news items to fetch
    
    Returns:
        List of news items
    """
    if source == "Finviz":
        return scrape_finviz_news(ticker, max_news)
    elif source == "Google News":
        return scrape_google_news(ticker, max_news)
    elif source == "Yahoo Finance":
        return scrape_yahoo_news(ticker, max_news)
    else:
        return []
