"""
Stock Dashboard Modules
Modular components for stock analysis dashboard
"""

from .data_fetcher import get_data_optimized, get_spy_data, get_fear_greed_index
from .indicators import calc_indicators, validate_indicators
from .news_scraper import scrape_finviz_news, scrape_google_news, get_news_from_source
from .sentiment import load_finbert_model, get_finbert_sentiment, analyze_news_sentiment_bert_only
from .strategy import rule_based_signal_v2
from .risk_metrics import calculate_risk_metrics
from .backtester import backtest_strategy
from .ml_models import run_ml_analysis, calculate_ensemble_recommendation
from .monte_carlo import estimate_days_to_target_advanced, monte_carlo_price_simulation

__all__ = [
    'get_data_optimized',
    'get_spy_data',
    'get_fear_greed_index',
    'calc_indicators',
    'validate_indicators',
    'scrape_finviz_news',
    'scrape_google_news',
    'get_news_from_source',
    'load_finbert_model',
    'get_finbert_sentiment',
    'analyze_news_sentiment_bert_only',
    'rule_based_signal_v2',
    'calculate_risk_metrics',
    'backtest_strategy',
    'run_ml_analysis',
    'calculate_ensemble_recommendation',
    'estimate_days_to_target_advanced',
    'monte_carlo_price_simulation'
]
